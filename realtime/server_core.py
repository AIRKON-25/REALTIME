import asyncio
import heapq
import math
import json
import os
import queue
import random
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from comms.command_server import CommandServer
from comms.udp_receiver import UDPReceiverSingle
from comms.ws_hub import WebSocketHub
from utils.colors import (
    color_label_to_hex,
    hex_to_color_label,
    normalize_color_hex,
    normalize_color_label,
)
from utils.tracking.cluster import ClassMergePolicy, ClusterConfig, cluster_by_aabb_iou
from utils.tracking.fusion import fuse_cluster_weighted
from utils.tracking.geometry import aabb_iou_axis_aligned
from utils.tracking.lane_matcher import LaneMatcher
from utils.tracking.tracker import SortTracker, TrackerConfigCar, TrackerConfigObstacle, TrackState
from utils.tracking._constants import ASSOC_CENTER_NORM, ASSOC_CENTER_WEIGHT, IOU_CLUSTER_THR
from realtime.runtime_constants import (
    COLOR_BIAS_STRENGTH,
    COLOR_BIAS_MIN_VOTES,
    EARLY_RELEASE_LOST_FRAMES,
    EARLY_REID_DIST_M,
    EARLY_REID_YAW_DEG,
    PATH_REID_DIST_M,
    PATH_REID_YAW_DEG,
    ROUTE_CHANGE_DISTANCE_M,
    VELOCITY_DT_MAX,
    VELOCITY_DT_MIN,
    VELOCITY_EWMA_ALPHA,
    VELOCITY_MAX_MPS,
    VELOCITY_SPEED_WINDOW,
    VELOCITY_ZERO_THRESH,
    vehicle_fixed_length,
    vehicle_fixed_width,
    rubberCone_fixed_length,
    rubberCone_fixed_width,
    barricade_fixed_length,
    barricade_fixed_width,
)
from realtime.status_core import StatusState


class FusionDashboard:
    """
    간단한 리치 기반 대시보드. 최신 스냅샷만 덮어쓰며 stdout을 깔끔하게 유지한다.
    """

    def __init__(self, refresh_hz: float = 4.0) -> None:
        self.refresh_hz = max(1.0, float(refresh_hz) if refresh_hz else 4.0)
        self._latest: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available = True

    def start(self) -> bool:
        try:
            # 지연 import로 rich 미설치 시도 실패를 런타임에만 노출
            import rich  # noqa: F401
        except Exception as exc:
            print(f"[Dashboard] rich not available: {exc}")
            self._available = False
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)

    def update(self, snapshot: dict) -> None:
        if not self._available:
            return
        with self._lock:
            self._latest = snapshot or {}

    @staticmethod
    def _fmt_float(val: Optional[float], digits: int = 2) -> str:
        try:
            return f"{float(val):.{digits}f}"
        except Exception:
            return "-"

    def _color_str(self, meta: dict) -> str:
        if not meta:
            return "-"
        label = meta.get("color")
        if label:
            return str(label)
        hex_val = meta.get("color_hex")
        if hex_val:
            try:
                from utils.colors import hex_to_color_label
                lbl = hex_to_color_label(hex_val)
                return lbl or str(hex_val)
            except Exception:
                return str(hex_val)
        return "-"

    @staticmethod
    def _color_text(label: Optional[str], hex_val: Optional[str], prefer_hex: bool = False):
        from rich.text import Text
        import re

        # 표시할 텍스트 우선순위: (1) hex 표시 요청 시 hex, (2) 라벨, (3) hex를 라벨로 변환, (4) hex 원문, (5) "-"
        disp = None
        if prefer_hex and hex_val:
            disp = hex_val
        if disp is None and label:
            disp = label
        if disp is None and hex_val:
            try:
                from utils.colors import hex_to_color_label
                disp = hex_to_color_label(hex_val) or hex_val
            except Exception:
                disp = hex_val
        disp = disp or "-"

        # 색상 스타일은 유효한 hex(#RRGGBB)나 안전한 라벨만 적용하고, 그 외엔 평문으로 둔다.
        safe_labels = {"red", "green", "yellow", "blue", "cyan", "magenta", "white", "black", "purple", "orange"}
        style = ""
        hex_str = None
        if isinstance(hex_val, str):
            m = re.match(r"^#(?:[0-9a-fA-F]{6})$", hex_val.strip())
            if m:
                hex_str = m.group(0)
        lbl_lower = str(label).lower() if label else None
        if hex_str and (prefer_hex or not label):
            style = hex_str
        elif lbl_lower in safe_labels:
            style = lbl_lower
        elif hex_str:
            style = hex_str

        return Text(str(disp), style=style)

    @staticmethod
    def _fmt_cams(cams: List[str]) -> str:
        ids = []
        for c in cams:
            s = str(c)
            if s.startswith("cam-"):
                s = s[4:]
            elif s.startswith("cam"):
                s = s[3:]
            ids.append(s)
        return ", ".join(ids)

    def _render(self, snap: dict):
        from rich import box
        from rich.console import Group
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        ts = snap.get("ts")
        raw_total = snap.get("raw_total", 0)
        raw_cls = snap.get("raw_per_cls") or {}
        raw_cam = snap.get("raw_per_cam") or {}
        clusters = snap.get("clusters") or []
        tracks = snap.get("tracks") or []
        track_meta = snap.get("track_meta") or {}
        ext_to_int = snap.get("ext_to_int") or {}
        timings = snap.get("timings") or {}
        metrics = snap.get("tracker_metrics") or {}
        car_status = snap.get("car_status") or {}

        def _fmt_progress(info: dict) -> str:
            if not info:
                return "-"
            out = "-"
            try:
                ratio = info.get("route_progress_ratio")
                if ratio is not None:
                    out = f"{float(ratio) * 100.0:.0f}%"
            except Exception:
                out = "-"
            try:
                idx = info.get("route_progress_idx")
                total = info.get("route_progress_total")
                if total:
                    idx_val = int(idx or 0)
                    total_val = int(total)
                    suffix = f" ({out})" if out != "-" else ""
                    out = f"{idx_val}/{total_val}{suffix}"
            except Exception:
                pass
            return out

        def _fmt_s_start(info: dict) -> str:
            if not info:
                return "-"
            s_start = info.get("s_start")
            if not s_start:
                return "-"
            try:
                first = s_start[0]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    return f"{float(first[0]):.1f},{float(first[1]):.1f}"
                if isinstance(first, dict) and "x" in first and "y" in first:
                    return f"{float(first['x']):.1f},{float(first['y']):.1f}"
            except Exception:
                pass
            try:
                return str(s_start)[:24]
            except Exception:
                return "-"

        def _fmt_path(info: dict) -> str:
            if not info:
                return "-"
            try:
                future_len = int(info.get("path_future_len") or 0)
                total_len = int(info.get("path_len") or future_len)
                if future_len == 0 and total_len == 0:
                    return "-"
                if total_len:
                    return f"{future_len}/{total_len}"
                return str(future_len)
            except Exception:
                return "-"

        summary_lines = [
            f"ts={self._fmt_float(ts, 3)}  raw={raw_total} (c0:{raw_cls.get(0,0)} c1:{raw_cls.get(1,0)} c2:{raw_cls.get(2,0)})",
            "cams=" + (", ".join([f"{k}:{v}" for k, v in sorted(raw_cam.items())]) or "-"),
            f"clusters={len(clusters)} tracks={len(tracks)} timings(ms) "
            f"g={self._fmt_float(timings.get('gather'))} f={self._fmt_float(timings.get('fuse'))} t={self._fmt_float(timings.get('track'))}",
        ]
        summary = Panel(Text("\n".join(summary_lines)), title="Fusion", border_style="cyan")

        fused_table = Table(title="Fused", box=box.SIMPLE, expand=True, show_lines=False)
        for col in ["#", "cls", "cx", "cy", "yaw", "score", "cams", "color"]:
            fused_table.add_column(col)
        for idx, det in enumerate(clusters):
            cams = det.get("source_cams") or []
            color_label = det.get("color")
            color_hex = det.get("color_hex")
            fused_table.add_row(
                str(idx),
                str(det.get("cls", "-")),
                self._fmt_float(det.get("cx")),
                self._fmt_float(det.get("cy")),
                self._fmt_float(det.get("yaw")),
                self._fmt_float(det.get("score")),
                self._fmt_cams(cams),
                self._color_text(color_label, color_hex, prefer_hex=True),
            )
        if not clusters:
            fused_table.add_row("-", "-", "-", "-", "-", "-", "-", "-")

        cls_groups: Dict[object, List[list]] = {}
        for row in tracks:
            try:
                cls_key = int(row[1])
            except Exception:
                cls_key = "?"
            cls_groups.setdefault(cls_key, []).append(row)

        def build_track_table(cls_key, rows):
            table = Table(title=f"Tracks cls={cls_key} ({len(rows)})", box=box.SIMPLE, expand=True, show_lines=False)
            for col in ["ext_id", "int_id", "cx", "cy", "yaw", "speed", "color", "batt", "path", "prog", "s_start", "route_chg"]:
                table.add_column(col)
            for row in rows:
                try:
                    ext_id = int(row[0])
                except Exception:
                    ext_id = None
                ext_disp = "-" if ext_id is None else ext_id
                int_id = ext_to_int.get(ext_id, "-") if ext_id is not None else "-"
                meta = track_meta.get(ext_id, {}) if ext_id is not None else {}
                color_label = meta.get("color")
                color_hex = meta.get("color_hex")
                car_info = car_status.get(ext_id) if ext_id is not None else None
                battery_str = self._fmt_float(car_info.get("battery"), 0) if car_info and car_info.get("battery") is not None else "-"
                path_str = _fmt_path(car_info) if car_info else "-"
                progress_str = _fmt_progress(car_info) if car_info else "-"
                s_start_str = _fmt_s_start(car_info) if car_info else "-"
                route_chg = car_info.get("route_change_count") if car_info else None
                route_chg_str = str(route_chg) if route_chg is not None else "-"
                table.add_row(
                    str(ext_disp),
                    str(int_id),
                    self._fmt_float(row[2] if len(row) > 2 else None),
                    self._fmt_float(row[3] if len(row) > 3 else None),
                    self._fmt_float(row[6] if len(row) > 6 else None),
                    self._fmt_float(meta.get("speed")),
                    self._color_text(color_label, color_hex),
                    battery_str,
                    path_str,
                    progress_str,
                    s_start_str,
                    route_chg_str,
                )
            if not rows:
                table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
            return table

        track_tables: List[Table] = []
        for cls_key in sorted(cls_groups.keys(), key=lambda v: (v == "?", v)):
            track_tables.append(build_track_table(cls_key, cls_groups.get(cls_key, [])))
        if not track_tables:
            track_tables.append(build_track_table("all", []))
        track_panel = Panel(Group(*track_tables), title="Tracks", border_style="magenta")

        metrics_table = Table(title="TrackerMetrics", box=box.SIMPLE, expand=True, show_lines=False)
        metrics_table.add_column("metric")
        metrics_table.add_column("value", justify="right")
        for key in [
            "num_tracks",
            "num_measurements",
            "num_matched",
            "num_matched_stage1",
            "num_matched_stage2",
            "num_matched_last_chance",
            "num_unmatched_tracks",
            "num_unmatched_measurements",
        ]:
            if key in metrics:
                metrics_table.add_row(key, str(metrics.get(key)))
        if not metrics:
            metrics_table.add_row("-", "-")

        layout = Layout()
        layout.split_column(
            Layout(summary, name="summary", size=5),
            Layout(fused_table, name="fused", ratio=3),
            Layout(track_panel, name="tracks", ratio=4),
            Layout(metrics_table, name="metrics", size=10),
        )
        return layout

    def _run(self) -> None:
        from rich.live import Live
        refresh = self.refresh_hz
        with Live(self._render({}), refresh_per_second=refresh, screen=False) as live:
            while not self._stop.is_set():
                with self._lock:
                    snap = dict(self._latest)
                live.update(self._render(snap))
                self._stop.wait(1.0 / refresh)

class RealtimeServer:
    def __init__(
        self,
        cam_ports: Dict[str, int],
        cam_positions_path: Optional[str] = None,
        fps: float = 10.0,
        iou_cluster_thr: float = IOU_CLUSTER_THR,
        single_port: int = 50050,
        tx_host: Optional[str] = None,
        tx_port: int = 60050,
        tx_protocol: str = "udp",
        carla_host: Optional[str] = None,
        carla_port: int = 61000,
        car_id_count: int = 5,
        # tracker_fixed_length: Optional[float] = None,
        # tracker_fixed_width: Optional[float] = None,
        command_host: Optional[str] = None,
        command_port: Optional[int] = None,
        ws_host: Optional[str] = "0.0.0.0",
        ws_port: int = 9001,
        cluster_config: Optional[ClusterConfig] = None,
        class_merge_policy: ClassMergePolicy = ClassMergePolicy.ALLOW_CAR_OBSTACLE, # 차-장애물 머지 허용
        enable_cluster_split: bool = False, # 클러스터 분할 허용하려면 True
        tracker_config_car: Optional[TrackerConfigCar] = None,
        tracker_config_obstacle: Optional[TrackerConfigObstacle] = None,
        assoc_center_weight: Optional[float] = None,
        assoc_center_norm: Optional[float] = None,
        debug_assoc_logging: bool = False,
        log_pipeline: bool = True,
        log_udp_packets: bool = False,
        lane_map_path: Optional[str] = "utils/make_H/lanes.json",
        status_state: Optional[StatusState] = None,
        debug_http_host: Optional[str] = "0.0.0.0",
        debug_http_port: Optional[int] = 18110,
        dashboard: bool = True,
        dashboard_refresh_hz: float = 4.0,
    ):
        self.fps = fps
        self.dt = 1.0 / max(1e-3, fps) # 루프 주기 초
        self.buffer_ttl = max(self.dt * 2.0, 1.0) # 버퍼 유효 시간 초
        self.cluster_config = cluster_config or ClusterConfig(
            iou_cluster_thr=iou_cluster_thr,
            merge_policy=class_merge_policy,
            allow_split_multi_modal=enable_cluster_split,
        )
        # 클래스가 다르게 찍힌 동일 물체(차/장애물)도 합쳐지도록 미스클래스 게이트를 완화
        try:
            self.cluster_config.merge_policy = ClassMergePolicy.ALLOW_CAR_OBSTACLE
            self.cluster_config.misclass_min_iou = min(self.cluster_config.misclass_min_iou, 0.1)
            self.cluster_config.misclass_max_shape_ratio = max(self.cluster_config.misclass_max_shape_ratio, 2.0)
            self.cluster_config.misclass_max_yaw_diff = max(self.cluster_config.misclass_max_yaw_diff, 45.0)
        except Exception:
            pass
        if self.cluster_config.iou_gate is None:
            self.cluster_config.iou_gate = self.cluster_config.iou_cluster_thr
        self.iou_thr = self.cluster_config.iou_cluster_thr
        self.track_meta: Dict[int, dict] = {}
        # External ID mapping: use 1..car_id_count for cars, overflow IDs start at 6.
        self.car_id_min = 1
        self.car_id_ceiling = 5
        if car_id_count is None:
            car_id_count = self.car_id_ceiling
        try:
            car_id_count = int(car_id_count)
        except (TypeError, ValueError):
            car_id_count = self.car_id_ceiling
        if not (self.car_id_min <= car_id_count <= self.car_id_ceiling):
            raise ValueError(
                f"car_id_count must be between {self.car_id_min} and {self.car_id_ceiling}"
            )
        self.car_id_max = car_id_count
        self._car_id_pool = list(range(self.car_id_min, self.car_id_max + 1))
        heapq.heapify(self._car_id_pool)
        self._other_id_pool: List[int] = []
        self._other_id_start = self.car_id_ceiling + 1
        self._next_other_id = self._other_id_start
        self._track_id_map: Dict[int, int] = {}
        self._external_to_internal: Dict[int, int] = {}
        self._released_external: Dict[int, dict] = {}
        self._track_state_cache: Dict[int, dict] = {}
        self._external_history: Dict[int, dict] = {}
        self._reid_keep_secs = 2.0
        self._reid_dist_gate_m = 4.0
        self._reid_color_bonus = 0.6
        self._reid_color_penalty = 1.0
        self._reid_dir_weight = 1.5
        self._reid_speed_weight = 0.5
        self._reid_yaw_weight = 0.5
        self._reid_min_speed_mps = 0.3
        self.active_cams = set()
        self.color_bias_strength = COLOR_BIAS_STRENGTH
        self.color_bias_min_votes = COLOR_BIAS_MIN_VOTES
        self._prev_tracks_for_cluster: Optional[np.ndarray] = None
        self.last_cluster_debug: Dict[str, int] = {}
        self.debug_assoc_logging = debug_assoc_logging
        self.log_pipeline = log_pipeline
        self.dashboard_enabled = bool(dashboard)
        self.dashboard_refresh_hz = dashboard_refresh_hz
        self.dashboard: Optional[FusionDashboard] = FusionDashboard(dashboard_refresh_hz) if self.dashboard_enabled else None
        self.status_state = status_state
        self._velocity_state: Dict[int, dict] = {}  # ext_id -> {cx, cy, ts, vx, vy, history}
        self.obstacle_stop_speed_max = 0.3  # m/s 이하만 UI에 표시
        self.obstacle_stop_window = 5  # 최근 n 프레임 평균 속도로 정지 판정
        self._obstacle_stop_history: Dict[int, deque] = {}
        self._debug_http_host = debug_http_host
        self._debug_http_port = debug_http_port
        self._debug_http_enabled = bool(debug_http_host and debug_http_port)
        self._debug_httpd: Optional[ThreadingHTTPServer] = None
        self._debug_http_thread: Optional[threading.Thread] = None
        self._debug_lock = threading.Lock()
        self._last_track_payload: dict = {}
        self._last_ui_snapshot: dict = {}
        self._route_change_sig: Dict[int, str] = {}
        self._route_change_sent_sig: Dict[int, str] = {}
        self._last_s_start_sig: Dict[str, str] = {}
        self._latest_route_versions: Dict[str, str] = {}
        self._latest_routes: Dict[str, List[dict]] = {}
        self._latest_route_visibility: Dict[str, bool] = {}
        self._ws_heartbeat_stop = threading.Event()
        self._ws_heartbeat_thread: Optional[threading.Thread] = None
        # 클러스터 단계별 스냅샷(WS/대시보드 공유)에 사용
        self._cluster_stage_snapshots: Dict[str, dict] = {}

        self._ui_cameras_on_map = [
            {"id": "camMarker-1", "cameraId": "cam-1", "x": -0.015, "y": 0.51},
            {"id": "camMarker-2", "cameraId": "cam-2", "x": 0.25, "y": -0.02},
            {"id": "camMarker-3", "cameraId": "cam-3", "x": 0.75, "y": -0.02},
            {"id": "camMarker-4", "cameraId": "cam-4", "x": 1.015, "y": 0.51},
            {"id": "camMarker-5", "cameraId": "cam-5", "x": 0.75, "y": 1.02},
            {"id": "camMarker-6", "cameraId": "cam-6", "x": 0.25, "y": 1.02},
        ]
        self._ui_cameras_status = [
            {"id": "cam-1", "name": "Camera 1", "streamUrl": "http://192.168.0.107:8080/stream", "streamBEVUrl": "http://192.168.0.107:8081/stream"},
            {"id": "cam-2", "name": "Camera 2", "streamUrl": "http://192.168.0.105:8080/stream", "streamBEVUrl": "http://192.168.0.105:8081/stream"},
            {"id": "cam-3", "name": "Camera 3", "streamUrl": "http://192.168.0.102:8080/stream", "streamBEVUrl": "http://192.168.0.102:8081/stream"},
            {"id": "cam-4", "name": "Camera 4", "streamUrl": "http://192.168.0.106:8080/stream", "streamBEVUrl": "http://192.168.0.106:8081/stream"},
            {"id": "cam-5", "name": "Camera 5", "streamUrl": "http://192.168.0.104:8080/stream", "streamBEVUrl": "http://192.168.0.104:8081/stream"},
            {"id": "cam-6", "name": "Camera 6", "streamUrl": "http://192.168.0.103:8080/stream", "streamBEVUrl": "http://192.168.0.103:8081/stream"},
        ]
        self._ui_cam_status: Optional[dict] = None
        _raw_traffic_lights = [
            {"id": "tl-1", "trafficLightId": 1, "x": -5.5, "y": -13.1, "yaw": 90.0},
            {"id": "tl-2", "trafficLightId": 2, "x": 5.6, "y": -17.1, "yaw": -90.0},
            {"id": "tl-3", "trafficLightId": 3, "x": 2.1, "y": -9.8, "yaw": 0.0},
            {"id": "tl-4", "trafficLightId": 4, "x": -19.5, "y": 6, "yaw": 0.0},
            {"id": "tl-5", "trafficLightId": 5, "x": -23.5, "y": -5.9, "yaw": 180.0},
            {"id": "tl-6", "trafficLightId": 6, "x": -16.1, "y": -1.9, "yaw": -90.0},
            {"id": "tl-7", "trafficLightId": 7, "x": 19.5, "y": -5.8, "yaw": 180.0},
            {"id": "tl-8", "trafficLightId": 8, "x": 23.7, "y": 6.2, "yaw": 0.0},
            {"id": "tl-9", "trafficLightId": 9, "x": 16.1, "y": 2.2, "yaw": 90.0},
        ]
        self._ui_traffic_lights_on_map = []
        for item in _raw_traffic_lights:
            try:
                x_map, y_map = self._world_to_map_xy(float(item["x"]), -float(item["y"]))
            except Exception:
                continue
            self._ui_traffic_lights_on_map.append(
                {
                    "id": item["id"],
                    "trafficLightId": int(item["trafficLightId"]),
                    "x": x_map,
                    "y": y_map,
                    "yaw": float(item["yaw"]),
                }
            )
        self._ui_traffic_light_status = {
            int(item["trafficLightId"]): {
                "trafficLightId": int(item["trafficLightId"]),
                "light": "red",
                "left_green": None,
            }
            for item in self._ui_traffic_lights_on_map
        }
        self._ui_traffic_light_map: Dict[str, dict] = {}
        self._ui_traffic_light_status_cache: Dict[int, dict] = {}
        self._last_traffic_light_msg: Optional[dict] = None
        self.ws_hub: Optional[WebSocketHub] = None
        if ws_host:
            try:
                self.ws_hub = WebSocketHub(
                    ws_host,
                    ws_port,
                    path="/monitor",
                    on_message=self._handle_ws_message,
                )
                self.ws_hub.start()
                # 초기 빈 스냅샷을 등록해 클라이언트 로딩을 빠르게 끝낸다.
                self._initialize_ui_snapshots()
                self._start_ws_heartbeat()
            except Exception as exc:
                print(f"[WebSocketHub] init failed: {exc}")
                self.ws_hub = None
        self._ui_obstacle_map: Dict[str, dict] = {}
        self._ui_obstacle_status_map: Dict[str, dict] = {}

        self.inference_receiver = UDPReceiverSingle(single_port, log_packets=log_udp_packets)

        self.cam_xy: Dict[str, Tuple[float, float]] = {} # 카메라 위치 로드... json으로 뺄까?

        self.buffer: Dict[str, deque] = {cam: deque(maxlen=1) for cam in cam_ports.keys()}
        """ 아래처럼 생김
        self.buffer = {
            "cam1": deque(maxlen=1), 최대길이 1인 덱
            "cam2": deque(maxlen=1) 즉, 카메라별로 가장 최신 데이터 1개만 유지하는 버퍼
        }
        """
        self.lane_matcher: Optional[LaneMatcher] = None
        self.lane_map_path = lane_map_path
        if lane_map_path and os.path.exists(lane_map_path):
            try:
                self.lane_matcher = LaneMatcher.from_json(lane_map_path)
                print(f"[LaneMatcher] loaded {lane_map_path}")
            except Exception as exc:
                print(f"[LaneMatcher] load failed: {exc}")
        elif lane_map_path:
            print(f"[LaneMatcher] path not found: {lane_map_path}")

        # 디버깅용 최근 지표
        self._last_rx_qsize: int = 0
        self._last_input_latency_ms: Optional[float] = None
        self._last_tx_send_ms: Optional[float] = None
        self._last_tx_item_count: int = 0
        self._last_tx_payload_bytes: Optional[int] = None
        self._tx_warn_ms = 20.0  # 송신이 이보다 길면 경고 출력
        self._last_tx_ts: Optional[float] = None
        self._last_tx_interval_ms: Optional[float] = None

        # 외부 전송(UDP/TCP) 설정
        self.track_tx = None
        self.carla_tx = None
        if tx_host:
            try:
                from comms.track_broadcaster import TrackBroadcaster
                self.track_tx = TrackBroadcaster(tx_host, tx_port, tx_protocol)
                print(f"[TrackBroadcaster] sending to {tx_host}:{tx_port} ({tx_protocol})")
            except Exception as exc:
                print(f"[TrackBroadcaster] init failed: {exc}")
                self.track_tx = None

        car_cfg = tracker_config_car or TrackerConfigCar()
        obs_cfg = tracker_config_obstacle or TrackerConfigObstacle()
        assoc_w = assoc_center_weight if assoc_center_weight is not None else ASSOC_CENTER_WEIGHT
        assoc_norm = assoc_center_norm if assoc_center_norm is not None else ASSOC_CENTER_NORM
        self.tracker = SortTracker(
            car_config=car_cfg,
            obstacle_config=obs_cfg,
            assoc_center_weight=assoc_w,
            assoc_center_norm=assoc_norm,
            debug_logging=self.debug_assoc_logging,
        ) # 기본 파라미터로 SortTracker 초기화
        self.last_tracker_metrics: Dict[str, int] = {}
    
        self._next_log_ts = 0.0

        self.command_queue: Optional[queue.Queue] = None
        self.command_server: Optional[CommandServer] = None
        if command_host and command_port:
            self.command_queue = queue.Queue()
            self.command_server = CommandServer(command_host, command_port, self.command_queue)

        self.tracker_fixed_length = vehicle_fixed_length
        self.tracker_fixed_width = vehicle_fixed_width

    def _world_to_map_xy(self, cx: float, cy: float) -> Tuple[float, float]:
        x_min, x_max = -25.72432848384547, 25.744828483845467
        y_min, y_max = -18.432071780223303, 19.171471780223307

        def _norm(v, vmin, vmax):
            if vmax <= vmin:
                return 0.5
            t = (v - vmin) / (vmax - vmin)
            return float(max(0.0, min(1.0, t)))

        u = _norm(cx, x_min, x_max)
        v = 1.0 - _norm(cy, y_min, y_max)
        return u, v

    def _compute_reid_cost(self, track_info: dict, prev_info: dict, now: float) -> Tuple[float, float]:
        ts = prev_info.get("ts")
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            ts = now
        dt = max(0.0, now - ts)
        pred_cx = float(prev_info.get("cx", 0.0))
        pred_cy = float(prev_info.get("cy", 0.0))
        vx = prev_info.get("vx")
        vy = prev_info.get("vy")
        if vx is not None and vy is not None:
            pred_cx += float(vx) * dt
            pred_cy += float(vy) * dt
        cx = float(track_info.get("cx", 0.0))
        cy = float(track_info.get("cy", 0.0))
        dist = float(np.hypot(cx - pred_cx, cy - pred_cy))
        cost = dist

        nvx = track_info.get("vx")
        nvy = track_info.get("vy")
        if vx is not None and vy is not None and nvx is not None and nvy is not None:
            speed_a = float(np.hypot(float(vx), float(vy)))
            speed_b = float(np.hypot(float(nvx), float(nvy)))
            if speed_a > self._reid_min_speed_mps and speed_b > self._reid_min_speed_mps:
                cos = (float(vx) * float(nvx) + float(vy) * float(nvy)) / (speed_a * speed_b)
                cos = max(-1.0, min(1.0, cos))
                cost += (1.0 - cos) * self._reid_dir_weight
        else:
            yaw_a = prev_info.get("yaw")
            yaw_b = track_info.get("yaw")
            if yaw_a is not None and yaw_b is not None:
                diff = abs(((float(yaw_a) - float(yaw_b) + 180.0) % 360.0) - 180.0)
                cost += (diff / 180.0) * self._reid_yaw_weight

        speed_a = prev_info.get("speed")
        speed_b = track_info.get("speed")
        if speed_a is None and vx is not None and vy is not None:
            speed_a = float(np.hypot(float(vx), float(vy)))
        if speed_b is None and nvx is not None and nvy is not None:
            speed_b = float(np.hypot(float(nvx), float(nvy)))
        if speed_a is not None and speed_b is not None:
            cost += abs(float(speed_a) - float(speed_b)) * self._reid_speed_weight

        color = track_info.get("color")
        info_color = prev_info.get("color")
        if color and info_color:
            if color == info_color:
                cost *= self._reid_color_bonus
            else:
                cost += self._reid_color_penalty
        return cost, dist

    def _update_velocity(self, ext_id: int, cx: float, cy: float, ts: float) -> Optional[Dict[str, float]]:
        """
        외부 ID 기준 위치 차분 + EMA로 속도를 계산한다. SORT 내부는 건드리지 않는다.
        """
        try:
            cx = float(cx)
            cy = float(cy)
            ts = float(ts)
        except Exception:
            return None
        state = self._velocity_state.get(ext_id)
        if state is None:
            self._velocity_state[ext_id] = {
                "cx": cx,
                "cy": cy,
                "ts": ts,
                "vx": 0.0,
                "vy": 0.0,
                "history": deque(maxlen=VELOCITY_SPEED_WINDOW),
            }
            return {"vx": 0.0, "vy": 0.0, "speed": 0.0}
        dt = ts - float(state.get("ts", ts))
        if dt < VELOCITY_DT_MIN:
            speed = float(np.hypot(state.get("vx", 0.0), state.get("vy", 0.0)))
            return {"vx": state.get("vx", 0.0), "vy": state.get("vy", 0.0), "speed": speed}
        if dt > VELOCITY_DT_MAX:
            self._velocity_state[ext_id] = {
                "cx": cx,
                "cy": cy,
                "ts": ts,
                "vx": 0.0,
                "vy": 0.0,
                "history": deque(maxlen=VELOCITY_SPEED_WINDOW),
            }
            return {"vx": 0.0, "vy": 0.0, "speed": 0.0}
        raw_vx = (cx - float(state.get("cx", cx))) / max(dt, 1e-6)
        raw_vy = (cy - float(state.get("cy", cy))) / max(dt, 1e-6)
        alpha = float(VELOCITY_EWMA_ALPHA)
        vx = alpha * raw_vx + (1.0 - alpha) * float(state.get("vx", 0.0))
        vy = alpha * raw_vy + (1.0 - alpha) * float(state.get("vy", 0.0))
        speed = float(np.hypot(vx, vy))
        if VELOCITY_ZERO_THRESH > 0 and speed < VELOCITY_ZERO_THRESH:
            vx = 0.0
            vy = 0.0
            speed = 0.0
            hist = deque([0.0], maxlen=VELOCITY_SPEED_WINDOW)
            self._velocity_state[ext_id] = {"cx": cx, "cy": cy, "ts": ts, "vx": vx, "vy": vy, "history": hist}
            return {"vx": vx, "vy": vy, "speed": speed}
        if VELOCITY_MAX_MPS > 0:
            speed = float(min(speed, VELOCITY_MAX_MPS))
        hist: deque = state.get("history")
        if hist is None:
            hist = deque(maxlen=VELOCITY_SPEED_WINDOW)
        hist.append(speed)
        smooth_speed = float(np.median(hist)) if hist else speed
        if VELOCITY_ZERO_THRESH > 0 and smooth_speed < VELOCITY_ZERO_THRESH:
            smooth_speed = 0.0
        self._velocity_state[ext_id] = {"cx": cx, "cy": cy, "ts": ts, "vx": vx, "vy": vy, "history": hist}
        return {"vx": vx, "vy": vy, "speed": smooth_speed}

    @staticmethod
    def _deg_diff(a: float, b: float) -> float:
        """Return absolute minimal difference of two angles in degrees."""
        try:
            diff = float(a) - float(b)
        except Exception:
            return 180.0
        return abs((diff + 180.0) % 360.0 - 180.0)

    def _path_reid_cost(self, ext_id: int, track_info: dict) -> Optional[float]:
        """
        경로 정보를 이용한 재-ID 비용 계산.
        반환값: 매칭 가능 시 비용(낮을수록 좋음), 불가 시 None.
        """
        if not self.status_state:
            return None
        route = self.status_state.get_route(ext_id)
        if not route:
            return None
        path_future = route.get("path_future") or []
        path_full = route.get("path") or []
        path = path_future if path_future else path_full
        if not path:
            return None
        try:
            px = float(track_info.get("cx"))
            py = float(track_info.get("cy"))
        except Exception:
            return None

        best_idx = None
        best_dist = float("inf")
        for idx, pt in enumerate(path):
            try:
                dx = px - float(pt[0])
                dy = py - float(pt[1])
            except Exception:
                continue
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None or best_dist > PATH_REID_DIST_M:
            return None

        expected_yaw = None
        if len(path) >= 2:
            i0 = max(0, min(best_idx, len(path) - 2))
            try:
                dx = float(path[i0 + 1][0]) - float(path[i0][0])
                dy = float(path[i0 + 1][1]) - float(path[i0][1])
                expected_yaw = math.degrees(math.atan2(dy, dx))
            except Exception:
                expected_yaw = None

        track_yaw = track_info.get("yaw")
        if track_yaw is not None and expected_yaw is not None:
            yaw_diff = self._deg_diff(track_yaw, expected_yaw)
            if yaw_diff > PATH_REID_YAW_DEG:
                return None
            cost = best_dist + yaw_diff / max(PATH_REID_YAW_DEG, 1e-3)
        else:
            cost = best_dist
        return float(cost)

    def _select_car_id(
        self,
        track_info: dict,
        now: Optional[float],
        used_ext_ids: Optional[set] = None,
    ) -> Optional[int]:
        if not self._car_id_pool:
            return None
        wall_now = time.time()
        used = used_ext_ids or set()
        candidates = list(self._car_id_pool)
        best_id = None
        best_cost = None
        for ext_id in candidates:
            if ext_id in used:
                continue
            history = self._external_history.get(ext_id)
            if not history:
                continue
            try:
                hist_ts = float(history.get("ts", 0.0))
                hist_age = max(0.0, wall_now - hist_ts)
            except Exception:
                hist_age = 0.0
            if hist_age > max(self._reid_keep_secs, 1.0):
                continue  # 오래된 히스토리는 무시하고 풀 최솟값으로 돌아가기
            cost, _ = self._compute_reid_cost(track_info, history, wall_now)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_id = ext_id
        if best_id is None:
            for ext_id in candidates:
                if ext_id not in used:
                    best_id = ext_id
                    break
        if best_id is None:
            return None
        self._car_id_pool.remove(best_id)
        heapq.heapify(self._car_id_pool)
        return best_id

    def _assign_external_id(
        self,
        prefer_car: bool,
        track_info: Optional[dict] = None,
        now: Optional[float] = None,
        used_ext_ids: Optional[set] = None,
    ) -> int:
        used = used_ext_ids or set()
        if prefer_car and self._car_id_pool:
            if track_info:
                selected = self._select_car_id(track_info, now, used)
                if selected is not None:
                    return selected
            while self._car_id_pool:
                ext_id = heapq.heappop(self._car_id_pool)
                if ext_id not in used:
                    return ext_id
        while self._other_id_pool:
            ext_id = heapq.heappop(self._other_id_pool)
            if ext_id >= self._other_id_start and ext_id not in used:
                return ext_id
        ext_id = self._next_other_id
        while ext_id in used:
            ext_id += 1
        self._next_other_id = ext_id + 1
        return ext_id

    def _recycle_external_id(self, ext_id: int) -> None:
        if ext_id <= self.car_id_max:
            heapq.heappush(self._car_id_pool, ext_id)
        elif ext_id >= self._other_id_start:
            heapq.heappush(self._other_id_pool, ext_id)

    def _refresh_id_pools(self, used_ext_ids: Optional[set] = None) -> None:
        used = used_ext_ids or set()
        blocked = used | set(self._released_external.keys())
        reserved = range(self.car_id_min, self.car_id_max + 1)
        self._car_id_pool = [ext_id for ext_id in reserved if ext_id not in blocked]
        heapq.heapify(self._car_id_pool)
        if self._other_id_pool:
            self._other_id_pool = [
                ext_id for ext_id in self._other_id_pool
                if ext_id >= self._other_id_start and ext_id not in used
            ]
            heapq.heapify(self._other_id_pool)
        if used:
            while self._next_other_id in used:
                self._next_other_id += 1

    def _dedupe_external_ids(
        self,
        active_internal_ids: List[int],
        track_info_by_id: Dict[int, dict],
        now: float,
    ) -> None:
        ext_to_internal: Dict[int, List[int]] = {}
        for tid in active_internal_ids:
            ext_id = self._track_id_map.get(tid)
            if ext_id is None:
                continue
            ext_to_internal.setdefault(ext_id, []).append(tid)

        used_ext_ids = set()
        to_reassign: List[int] = []
        for ext_id, tids in ext_to_internal.items():
            if len(tids) == 1:
                used_ext_ids.add(ext_id)
                continue
            history = self._external_history.get(ext_id)
            if history:
                best_tid = min(
                    tids,
                    key=lambda tid: self._compute_reid_cost(
                        track_info_by_id.get(tid, {}),
                        history,
                        now,
                    )[0],
                )
            else:
                best_tid = min(tids)
            used_ext_ids.add(ext_id)
            for tid in tids:
                if tid != best_tid:
                    self._track_id_map.pop(tid, None)
                    to_reassign.append(tid)

        for tid in active_internal_ids:
            if self._track_id_map.get(tid) is None and tid not in to_reassign:
                to_reassign.append(tid)

        self._refresh_id_pools(used_ext_ids)
        for tid in to_reassign:
            info = track_info_by_id.get(tid, {})
            prefer_car = int(info.get("cls", 1)) == 0
            ext_id = self._assign_external_id(prefer_car, info, now, used_ext_ids)
            self._track_id_map[tid] = ext_id
            used_ext_ids.add(ext_id)

    def _release_external_id(self, internal_id: int) -> None:
        ext_id = self._track_id_map.pop(internal_id, None)
        if ext_id is None:
            return
        self._external_to_internal.pop(ext_id, None)
        self._recycle_external_id(ext_id)

    def _stash_released_id(self, internal_id: int, now: float) -> None:
        ext_id = self._track_id_map.pop(internal_id, None)
        if ext_id is None:
            return
        self._external_to_internal.pop(ext_id, None)
        info = self._track_state_cache.pop(internal_id, None)
        if info:
            info = dict(info)
            info["ts"] = float(now)
        is_car = bool(info) and int(info.get("cls", 1)) == 0
        if info:
            self._external_history[ext_id] = info
        if ext_id <= self.car_id_max and is_car:
            self._released_external[ext_id] = {
                "ts": float(now),
                "cls": 0,
                "cx": float(info.get("cx", 0.0)),
                "cy": float(info.get("cy", 0.0)),
                "color": info.get("color"),
                "vx": info.get("vx"),
                "vy": info.get("vy"),
                "speed": info.get("speed"),
                "yaw": info.get("yaw"),
            }
            return
        self._recycle_external_id(ext_id)

    def _prune_released_ids(self, now: float) -> None:
        if self._reid_keep_secs <= 0.0:
            for ext_id in list(self._released_external.keys()):
                self._recycle_external_id(ext_id)
            self._released_external.clear()
            return
        for ext_id, info in list(self._released_external.items()):
            try:
                ts_val = float(info.get("ts", 0.0))
            except (TypeError, ValueError):
                ts_val = now
            age = now - ts_val
            if age < 0:
                age = 0.0
            if age <= self._reid_keep_secs:
                continue
            self._released_external.pop(ext_id, None)
            self._recycle_external_id(ext_id)

    def _force_detach_external_id(self, ext_id: int, keep_internal: Optional[int] = None) -> None:
        """
        강제로 외부 ID를 모든 상태에서 떼어낸다.
        keep_internal로 지정된 내부 ID는 보호한다.
        """
        if ext_id is None:
            return
        # drop released/history caches
        self._released_external.pop(ext_id, None)
        self._external_history.pop(ext_id, None)
        # drop mapping for other tracks
        for tid, eid in list(self._track_id_map.items()):
            if eid == ext_id and tid != keep_internal:
                self._track_id_map.pop(tid, None)
                self._track_state_cache.pop(tid, None)
        mapped_internal = self._external_to_internal.get(ext_id)
        if mapped_internal is not None and mapped_internal != keep_internal:
            self._external_to_internal.pop(ext_id, None)
        if keep_internal is None:
            self._external_to_internal.pop(ext_id, None)
        # also remove from pool; refresh will rebuild
        if ext_id in self._car_id_pool:
            try:
                self._car_id_pool.remove(ext_id)
                heapq.heapify(self._car_id_pool)
            except ValueError:
                pass

    def _repair_car_id_pool(self) -> None:
        """Ensure reserved car IDs are neither lost nor blocked forever."""
        reserved = set(range(self.car_id_min, self.car_id_max + 1))
        blocked = set(self._released_external.keys()) | set(self._track_id_map.values())
        # Drop blocked IDs from the pool
        if self._car_id_pool:
            self._car_id_pool = [cid for cid in self._car_id_pool if cid not in blocked]
            heapq.heapify(self._car_id_pool)
        # Put back any unblocked reserved IDs that are missing from the pool
        missing = reserved - blocked - set(self._car_id_pool)
        if missing:
            self._car_id_pool.extend(missing)
            heapq.heapify(self._car_id_pool)

    def _enforce_car_id_cap(self, car_internal_ids: List[int], track_info_by_id: Dict[int, dict], now: float) -> None:
        if len(car_internal_ids) > self.car_id_max:
            return
        reserved_ids = set(range(self.car_id_min, self.car_id_max + 1))
        active_reserved = {
            self._track_id_map.get(tid)
            for tid in car_internal_ids
            if self._track_id_map.get(tid) in reserved_ids
        }
        available = sorted(reserved_ids - active_reserved)
        needs = [
            tid for tid in car_internal_ids
            if self._track_id_map.get(tid) not in reserved_ids
        ]
        if not needs:
            self._car_id_pool = list(available)
            heapq.heapify(self._car_id_pool)
            return
        pairs: List[Tuple[float, int, int]] = []
        for tid in needs:
            info = track_info_by_id.get(tid)
            if not info:
                continue
            for ext_id in available:
                history = self._external_history.get(ext_id)
                if not history:
                    continue
                cost, _ = self._compute_reid_cost(info, history, now)
                pairs.append((cost, tid, ext_id))
        pairs.sort(key=lambda v: v[0])
        used_internal = set()
        used_external = set()
        for cost, internal_id, ext_id in pairs:
            if internal_id in used_internal or ext_id in used_external:
                continue
            self._track_id_map[internal_id] = ext_id
            used_internal.add(internal_id)
            used_external.add(ext_id)
        available = [ext_id for ext_id in available if ext_id not in used_external]
        remaining = [tid for tid in needs if tid not in used_internal]
        for internal_id, ext_id in zip(remaining, available):
            self._track_id_map[internal_id] = ext_id
        for internal_id in needs:
            ext_id = self._track_id_map.get(internal_id)
            if ext_id in self._released_external:
                self._released_external.pop(ext_id, None)
        assigned_now = {
            self._track_id_map.get(tid)
            for tid in car_internal_ids
            if self._track_id_map.get(tid) in reserved_ids
        }
        self._car_id_pool = list(reserved_ids - assigned_now)
        heapq.heapify(self._car_id_pool)

    def _map_track_ids(self, tracks: np.ndarray, ts: float) -> Tuple[np.ndarray, Dict[int, dict]]:
        wall_now = time.time()
        now = wall_now  # ID 매핑/재식별용 기준 시간은 항상 벽시계로 맞춘다.
        self._prune_released_ids(wall_now)
        self._repair_car_id_pool()
        if tracks is None or len(tracks) == 0:
            for tid in list(self._track_id_map.keys()):
                self._stash_released_id(tid, wall_now)
            self._external_to_internal = {}
            return tracks, {}

        active_ids = {int(row[0]) for row in tracks}
        for tid in list(self._track_id_map.keys()):
            if tid not in active_ids:
                self._stash_released_id(tid, wall_now)

        # 조기 해제: LOST가 일정 프레임 이상이며 예약 ID 범위 밖을 쓰는 차량의 외부 ID를 미리 풀어준다.
        early_release_ids: set = set()
        if EARLY_RELEASE_LOST_FRAMES > 0:
            try:
                snapshots = self.tracker.get_tracks_snapshot()
            except Exception:
                snapshots = []
            for snap in snapshots:
                if snap.get("state") != "lost":
                    continue
                if snap.get("time_since_update", 0) < EARLY_RELEASE_LOST_FRAMES:
                    continue
                internal_id = int(snap.get("id"))
                cls_val = int(snap.get("cls", 0))
                if cls_val != 0:
                    continue
                ext_id = self._track_id_map.get(internal_id)
                if ext_id is None:
                    continue
                if ext_id <= self.car_id_max:
                    continue  # 예약 ID는 조기 해제하지 않음
                info = {
                    "ts": now,
                    "cls": 0,
                    "cx": float(snap.get("cx", 0.0)),
                    "cy": float(snap.get("cy", 0.0)),
                    "color": snap.get("color"),
                    "vx": None,
                    "vy": None,
                    "speed": snap.get("speed"),
                    "yaw": snap.get("yaw"),
                }
                self._external_history[ext_id] = info
                released = dict(info)
                released["early"] = True
                self._released_external[ext_id] = released
                self._track_id_map.pop(internal_id, None)
                self._external_to_internal.pop(ext_id, None)
                self._track_state_cache.pop(internal_id, None)
                try:
                    for t in self.tracker.tracks:
                        if int(t.id) == internal_id:
                            t.state = TrackState.DELETED
                            break
                except Exception:
                    pass
                early_release_ids.add(internal_id)

        if early_release_ids and tracks is not None and len(tracks) > 0:
            tracks = np.array([row for row in tracks if int(row[0]) not in early_release_ids], dtype=float)

        track_info_by_id: Dict[int, dict] = {}
        new_track_rows: List[dict] = []
        for row in tracks:
            internal_id = int(row[0])
            cls = int(row[1]) if len(row) > 1 else 0
            meta = self.track_meta.get(internal_id, {})
            color = normalize_color_label(meta.get("color"))
            if not color:
                color = hex_to_color_label(meta.get("color_hex"))
            vel = meta.get("velocity")
            vx = vy = None
            if vel is not None and len(vel) == 2:
                try:
                    vx = float(vel[0])
                    vy = float(vel[1])
                except (TypeError, ValueError):
                    vx = None
                    vy = None
            speed = meta.get("speed")
            try:
                speed = float(speed) if speed is not None else None
            except (TypeError, ValueError):
                speed = None
            yaw = float(row[6]) if len(row) > 6 else None
            ext_id_existing = self._track_id_map.get(internal_id)
            info = {
                "internal_id": internal_id,
                "ts": wall_now,
                "cls": cls,
                "cx": float(row[2]),
                "cy": float(row[3]),
                "color": color,
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "yaw": yaw,
            }
            track_info_by_id[internal_id] = info
            if self.status_state and ext_id_existing is not None and cls == 0:
                try:
                    self.status_state.advance_path_progress(
                        ext_id_existing,
                        (info["cx"], info["cy"]),
                    )
                except Exception:
                    pass
            if internal_id in self._track_id_map:
                continue
            new_track_rows.append(info)

        if self._released_external:
            pairs = []
            relax_gate = not self._car_id_pool
            for track_info in new_track_rows:
                if track_info.get("cls") != 0:
                    continue
                for ext_id, released_info in self._released_external.items():
                    if released_info.get("cls") != 0:
                        continue
                    early_flag = bool(released_info.get("early"))
                    path_cost = self._path_reid_cost(ext_id, track_info)
                    if path_cost is not None:
                        pairs.append((path_cost, track_info["internal_id"], ext_id))
                        continue
                    cost, dist = self._compute_reid_cost(track_info, released_info, now)
                    gate_dist = EARLY_REID_DIST_M if early_flag else self._reid_dist_gate_m
                    if dist is not None and not relax_gate and dist > gate_dist:
                        continue
                    if early_flag:
                        try:
                            yaw_prev = released_info.get("yaw")
                            yaw_new = track_info.get("yaw")
                            if yaw_prev is not None and yaw_new is not None:
                                if self._deg_diff(yaw_new, yaw_prev) > EARLY_REID_YAW_DEG:
                                    continue
                        except Exception:
                            pass
                    pairs.append((cost, track_info["internal_id"], ext_id))
            pairs.sort(key=lambda v: v[0])
            used_internal = set()
            used_external = set()
            for cost, internal_id, ext_id in pairs:
                if internal_id in used_internal or ext_id in used_external:
                    continue
                self._track_id_map[internal_id] = ext_id
                used_internal.add(internal_id)
                used_external.add(ext_id)
                self._released_external.pop(ext_id, None)

        id_map: Dict[int, int] = {}
        for row in tracks:
            internal_id = int(row[0])
            cls = int(row[1]) if len(row) > 1 else 0
            is_car = cls == 0
            ext_id = self._track_id_map.get(internal_id)
            track_info = track_info_by_id.get(internal_id)
            if ext_id is None:
                ext_id = self._assign_external_id(is_car, track_info, now)
                self._track_id_map[internal_id] = ext_id
            elif not is_car and ext_id < self._other_id_start:
                # Keep obstacles out of the reserved car ID range.
                self._release_external_id(internal_id)
                ext_id = self._assign_external_id(False, None, now)
                self._track_id_map[internal_id] = ext_id
            elif is_car and ext_id > self.car_id_max and self._car_id_pool:
                # Prefer reserved IDs for cars when slots open up.
                self._release_external_id(internal_id)
                ext_id = self._assign_external_id(True, track_info, now)
                self._track_id_map[internal_id] = ext_id
            id_map[internal_id] = ext_id

        car_internal_ids = [
            int(row[0]) for row in tracks
            if int(row[1]) == 0
        ]
        if car_internal_ids:
            self._enforce_car_id_cap(car_internal_ids, track_info_by_id, now)
        active_internal_ids = [int(row[0]) for row in tracks]
        self._dedupe_external_ids(active_internal_ids, track_info_by_id, now)

        id_map = {
            int(row[0]): self._track_id_map.get(int(row[0]), int(row[0]))
            for row in tracks
        }
        self._external_to_internal = {ext: internal for internal, ext in id_map.items()}

        new_cache: Dict[int, dict] = {}
        for row in tracks:
            internal_id = int(row[0])
            info = track_info_by_id.get(internal_id, {})
            new_cache[internal_id] = info
            ext_id = id_map.get(internal_id)
            if ext_id is not None:
                self._external_history[ext_id] = info
        self._track_state_cache = new_cache

        mapped = tracks.copy()
        for idx, row in enumerate(tracks):
            internal_id = int(row[0])
            mapped[idx, 0] = float(id_map.get(internal_id, row[0]))
        mapped_meta: Dict[int, dict] = {}
        for row in tracks:
            internal_id = int(row[0])
            ext_id = id_map.get(internal_id)
            if ext_id is None:
                continue
            meta = dict(self.track_meta.get(internal_id, {}))
            vel = self._update_velocity(ext_id, float(row[2]), float(row[3]), now)
            if vel:
                meta["velocity"] = [float(vel["vx"]), float(vel["vy"])]
                meta["speed"] = float(vel["speed"])
            mapped_meta[ext_id] = meta
        return mapped, mapped_meta

    def _build_cam_status_message(self, ts: float) -> dict:
        return {
            "type": "camStatus",
            "ts": ts,
            "data": {
                "camerasOnMap": list(self._ui_cameras_on_map),
                "camerasStatus": list(self._ui_cameras_status),
            },
        }

    def _build_traffic_light_status_message(self, ts: float) -> Optional[dict]:
        if self.status_state:
            for item in self._ui_traffic_lights_on_map:
                try:
                    tid = int(item.get("trafficLightId"))
                except Exception:
                    continue
                try:
                    latest = self.status_state.get_traffic(tid)
                except Exception:
                    latest = None
                if not latest:
                    continue
                prev = self._ui_traffic_light_status.get(
                    tid, {"trafficLightId": tid, "light": "red", "left_green": None}
                )
                updated = dict(prev)
                updated["trafficLightId"] = tid
                light_val = latest.get("light")
                if light_val is not None:
                    updated["light"] = str(light_val)
                if "left_green" in latest:
                    left = latest.get("left_green")
                    updated["left_green"] = bool(left) if left is not None else None
                self._ui_traffic_light_status[tid] = updated

        current_map = {str(item["trafficLightId"]): item for item in self._ui_traffic_lights_on_map}
        current_status = {int(tid): status for tid, status in self._ui_traffic_light_status.items()}

        if not self._ui_traffic_light_map and not self._ui_traffic_light_status_cache:
            self._ui_traffic_light_map = current_map
            self._ui_traffic_light_status_cache = current_status
            message = {
                "type": "trafficLightStatus",
                "ts": ts,
                "data": {
                    "mode": "snapshot",
                    "trafficLightsOnMap": list(current_map.values()),
                    "trafficLightsStatus": list(current_status.values()),
                },
            }
            self._last_traffic_light_msg = message
            return message

        upserts = [
            item
            for tid, item in current_map.items()
            if self._ui_traffic_light_map.get(tid) != item
        ]
        deletes = [tid for tid in self._ui_traffic_light_map.keys() if tid not in current_map]
        status_upserts = [
            item
            for tid, item in current_status.items()
            if self._ui_traffic_light_status_cache.get(tid) != item
        ]
        status_deletes = [
            tid for tid in self._ui_traffic_light_status_cache.keys()
            if tid not in current_status
        ]

        if not (upserts or deletes or status_upserts or status_deletes):
            return None

        self._ui_traffic_light_map = current_map
        self._ui_traffic_light_status_cache = current_status

        data: Dict[str, object] = {"mode": "delta"}
        if upserts:
            data["trafficLightsOnMapUpserts"] = upserts
        if deletes:
            data["trafficLightsOnMapDeletes"] = [int(tid) for tid in deletes]
        if status_upserts:
            data["trafficLightsStatusUpserts"] = status_upserts
        if status_deletes:
            data["trafficLightsStatusDeletes"] = [int(tid) for tid in status_deletes]

        message = {"type": "trafficLightStatus", "ts": ts, "data": data}
        self._last_traffic_light_msg = message
        return message

    @staticmethod
    def _normalize_car_color(label: Optional[str]) -> str:
        allowed = {"red", "green", "yellow", "purple", "white"}
        if label in allowed:
            return label
        return "red"

    @staticmethod
    def _normalize_camera_id(cam_val: Optional[object]) -> Optional[str]:
        if cam_val is None:
            return None
        cam_str = str(cam_val)
        if cam_str.startswith("cam-"):
            return cam_str
        if cam_str.startswith("cam") and len(cam_str) > 3:
            return f"cam-{cam_str[3:]}"
        return cam_str

    @staticmethod
    def _normalize_camera_ids(cam_vals: Optional[object]) -> List[str]:
        if cam_vals is None:
            return []
        if isinstance(cam_vals, (list, tuple, set, deque)):
            candidates = cam_vals
        else:
            candidates = [cam_vals]
        normalized: List[str] = []
        for cam_val in candidates:
            cid = RealtimeServer._normalize_camera_id(cam_val)
            if cid:
                normalized.append(cid)
        seen = set()
        unique_ids: List[str] = []
        for cid in normalized:
            if cid in seen:
                continue
            seen.add(cid)
            unique_ids.append(cid)
        return unique_ids

    @staticmethod
    def _obstacle_kind(cls: int) -> str:
        if cls == 2:
            return "barricade"
        return "rubberCone"

    def _build_obstacle_delta_message(
        self,
        obstacles_on_map: List[dict],
        obstacles_status: List[dict],
        ts: float,
    ) -> Optional[dict]:
        current_map = {item["obstacleId"]: item for item in obstacles_on_map}
        current_status = {item["id"]: item for item in obstacles_status}

        upserts = [
            item
            for oid, item in current_map.items()
            if self._ui_obstacle_map.get(oid) != item
        ]
        deletes = [oid for oid in self._ui_obstacle_map.keys() if oid not in current_map]
        status_upserts = [
            item
            for oid, item in current_status.items()
            if self._ui_obstacle_status_map.get(oid) != item
        ]
        status_deletes = [
            oid for oid in self._ui_obstacle_status_map.keys() if oid not in current_status
        ]

        if not (upserts or deletes or status_upserts or status_deletes):
            return None

        self._ui_obstacle_map = current_map
        self._ui_obstacle_status_map = current_status

        data: Dict[str, object] = {"mode": "delta"}
        if upserts:
            data["upserts"] = upserts
        if deletes:
            data["deletes"] = deletes
        if status_upserts:
            data["statusUpserts"] = status_upserts
        if status_deletes:
            data["statusDeletes"] = status_deletes

        return {"type": "obstacleStatus", "ts": ts, "data": data}

    def _build_cluster_stage_message(
        self,
        stage: str,
        dets: Optional[List[dict]],
        ts: float,
    ) -> dict:
        cars_on_map: List[dict] = []
        cars_status: List[dict] = []
        obstacles_on_map: List[dict] = []
        obstacles_status: List[dict] = []

        for idx, det in enumerate(dets or []):
            try:
                cls = int(det.get("cls", det.get("class", 1)))
            except Exception:
                cls = 1
            try:
                cx = float(det.get("cx", 0.0))
                cy = float(det.get("cy", 0.0))
                x_norm, y_norm = self._world_to_map_xy(cx, -cy)
            except Exception:
                continue
            try:
                yaw_val = float(det.get("yaw", 0.0) or 0.0)
            except Exception:
                yaw_val = 0.0
            color_label = det.get("color") or hex_to_color_label(det.get("color_hex"))
            car_color = self._normalize_car_color(color_label)
            camera_ids = self._normalize_camera_ids(det.get("cam") or det.get("source_cams"))

            if cls == 0:
                car_id = f"{stage}-car-{idx}"
                map_id = f"{stage}-mcar-{idx}"
                cars_on_map.append({
                    "id": map_id,
                    "carId": car_id,
                    "x": x_norm,
                    "y": y_norm,
                    "yaw": yaw_val,
                    "color": car_color,
                    "status": "normal",
                })
                cars_status.append({
                    "class": cls,
                    "car_id": car_id,
                    "color": car_color,
                    "speed": 0.0,
                    "battery": 0.0,
                    "cameraIds": camera_ids,
                    "path_future": [],
                    "category": "",
                    "resolution": "",
                    "routeChanged": False,
                })
            else:
                obstacle_id = f"{stage}-ob-{idx}"
                obstacles_on_map.append({
                    "id": obstacle_id,
                    "obstacleId": obstacle_id,
                    "x": x_norm,
                    "y": y_norm,
                    "kind": self._obstacle_kind(cls),
                })
                obstacles_status.append({
                    "id": obstacle_id,
                    "class": cls,
                    "cameraIds": camera_ids,
                })

        return {
            "type": "clusterStage",
            "ts": ts,
            "data": {
                "stage": stage,
                "mode": "snapshot",
                "carsOnMap": cars_on_map,
                "carsStatus": cars_status,
                "obstaclesOnMap": obstacles_on_map,
                "obstaclesStatus": obstacles_status,
            },
        }

    def _broadcast_cluster_stage(self, stage: str, dets: Optional[List[dict]], ts: float):
        message = self._build_cluster_stage_message(stage, dets or [], ts)
        with self._debug_lock:
            self._cluster_stage_snapshots[stage] = message
        if self.ws_hub:
            self.ws_hub.broadcast(message)

    def _build_ui_messages(
        self,
        tracks: np.ndarray,
        track_meta: Dict[int, dict],
        ts: float,
    ) -> List[dict]:
        messages: List[dict] = []
        route_change_msgs: List[dict] = []
        active_car_ids: set = set()

        traffic_light_msg = self._build_traffic_light_status_message(ts)

        if self._ui_cam_status is None:
            cam_msg = self._build_cam_status_message(ts)
            self._ui_cam_status = cam_msg
            initial_messages = [cam_msg]
            if traffic_light_msg:
                initial_messages.append(traffic_light_msg)
            if self.ws_hub:
                self.ws_hub.set_initial_messages(initial_messages)
            messages.extend(initial_messages)
        elif traffic_light_msg:
            messages.append(traffic_light_msg)

        cars_on_map: List[dict] = []
        cars_status: List[dict] = []
        obstacles_on_map: List[dict] = []
        obstacles_status: List[dict] = []
        active_obstacle_ids: set = set()

        def _first_xy(obj) -> Optional[Tuple[float, float]]:
            try:
                if isinstance(obj, dict):
                    return float(obj.get("x")), float(obj.get("y"))
                if isinstance(obj, (list, tuple)) and len(obj) >= 2:
                    return float(obj[0]), float(obj[1])
            except Exception:
                return None
            return None

        def _closest_index(target: Tuple[float, float], path_list) -> Optional[int]:
            best_idx = None
            best_dist = float("inf")
            for idx, pt in enumerate(path_list):
                xy = _first_xy(pt)
                if xy is None:
                    continue
                dx = xy[0] - target[0]
                dy = xy[1] - target[1]
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            return best_idx

        def _normalize_route_points(raw_points) -> List[dict]:
            points: List[dict] = []
            if not raw_points:
                return points
            for pt in raw_points:
                try:
                    if isinstance(pt, dict):
                        px, py = float(pt.get("x")), float(pt.get("y"))
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        px, py = float(pt[0]), float(pt[1])
                    else:
                        continue
                except Exception:
                    continue
                try:
                    x_route, y_route = self._world_to_map_xy(px, -py)
                except Exception:
                    continue
                points.append({"x": x_route, "y": y_route})
            return points

        if tracks is not None and len(tracks):
            for row in tracks:
                ext_id = int(row[0])  # mapped_tracks에서 이미 외부 ID가 들어있음
                cls = int(row[1])
                cx = float(row[2])
                cy = float(-row[3])
                yaw = float(row[6])
                meta = track_meta.get(ext_id, {})
                color_label = meta.get("color") or hex_to_color_label(meta.get("color_hex"))
                car_color = self._normalize_car_color(color_label)

                x_norm, y_norm = self._world_to_map_xy(cx, cy)

                if cls == 0:
                    source_cams = meta.get("source_cams") or []
                    camera_ids = self._normalize_camera_ids(source_cams)
                    car_key = ext_id
                    car_id = f"car-{car_key}"
                    map_id = f"mcar-{car_key}"
                    active_car_ids.add(car_id)
                    car_state = None
                    if self.status_state:
                        try:
                            car_state = self.status_state.get_car(ext_id)
                        except Exception:
                            car_state = None
                    battery_val: float = 100
                    path_future_raw = []
                    category_val = ""
                    resolution_val = None
                    route_changed = False
                    route_sig = None
                    s_start_sig = None
                    s_start_changed = False
                    path_full_raw = []
                    s_start = None
                    s_end = None
                    allow_route = False
                    if car_state:
                        try:
                            battery_raw = car_state.get("battery")
                            if battery_raw is not None:
                                battery_val = float(battery_raw)
                        except Exception:
                            pass
                        path_full_raw = car_state.get("path") or []
                        path_future_raw = car_state.get("path_future")
                        if path_future_raw is None:
                            path_future_raw = path_full_raw or []
                        category_val = car_state.get("category") or ""
                        resolution_val = car_state.get("resolution")
                        s_start = car_state.get("s_start")
                        s_end = car_state.get("s_end")
                        route_sig = car_state.get("route_version")
                        allow_route = bool(s_start) or bool(s_end)
                        if s_start:
                            try:
                                s_start_sig = json.dumps(s_start, ensure_ascii=False, sort_keys=True)
                            except Exception:
                                try:
                                    s_start_sig = str(s_start)
                                except Exception:
                                    s_start_sig = None
                            prev_s_start_sig = self._last_s_start_sig.get(car_id)
                            if s_start_sig and s_start_sig != prev_s_start_sig:
                                s_start_changed = True
                        if not route_sig and s_start:
                            try:
                                route_sig = json.dumps(
                                    {"s_start": s_start, "path": path_full_raw or path_future_raw},
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                            except Exception:
                                route_sig = str(s_start)
                            prev_sig = self._route_change_sig.get(car_key)
                            if route_sig != prev_sig:
                                route_changed = True
                                self._route_change_sig[car_key] = route_sig
                                try:
                                    s_start_xy = _first_xy(s_start[0]) if s_start else None
                                    path_first_xy = _first_xy(path_future_raw[0]) if path_future_raw else None
                                    res_val = resolution_val
                                    try:
                                        res_val = float(res_val) if res_val is not None else 1.0
                                    except Exception:
                                        res_val = 1.0
                                    res_val = max(res_val, 1e-3)
                                    resolution_val = res_val
                                    if s_start_xy and path_first_xy:
                                        idx_in_path = _closest_index(s_start_xy, path_future_raw)
                                        dist = math.hypot(s_start_xy[0] - path_first_xy[0], s_start_xy[1] - path_first_xy[1])
                                        threshold_steps = dist / res_val
                                        if idx_in_path is not None and idx_in_path >= threshold_steps:
                                            route_changed = False
                                    if route_changed and s_start_xy and path_future_raw:
                                        idx_from_current = _closest_index(s_start_xy, path_future_raw)
                                        step_gate = float(ROUTE_CHANGE_DISTANCE_M) / res_val
                                        if idx_from_current is None or idx_from_current >= step_gate:
                                            route_changed = False
                                except Exception:
                                    pass
                        else:
                            self._route_change_sig.pop(car_key, None)
                    route_points = _normalize_route_points(path_future_raw)
                    if route_sig is None and route_points and (s_start or s_end):
                        try:
                            route_sig = json.dumps(path_full_raw or route_points, ensure_ascii=False, sort_keys=True)
                        except Exception:
                            route_sig = None
                    if route_sig is None and route_points:
                        try:
                            route_sig = json.dumps(route_points, ensure_ascii=False, sort_keys=True)
                        except Exception:
                            route_sig = None
                    map_status = "routeChanged" if route_changed else "normal"
                    if route_points:
                        self._latest_route_visibility[car_id] = bool(allow_route)
                        last_sent_sig = self._route_change_sent_sig.get(car_id)
                        should_emit_route_change = False
                        if route_sig and last_sent_sig != route_sig:
                            should_emit_route_change = True
                        elif not last_sent_sig and route_sig:
                            should_emit_route_change = True
                        if should_emit_route_change and route_sig:
                            route_change_msgs.append({
                                "carId": car_id,
                                "newRoute": route_points,
                                "routeVersion": route_sig,
                                "visible": bool(s_start_changed) if allow_route else False,
                            })
                            self._route_change_sent_sig[car_id] = route_sig
                        if s_start_sig and s_start_changed:
                            self._last_s_start_sig[car_id] = s_start_sig
                        if route_sig and route_sig != self._route_change_sent_sig.get(car_id):
                            self._route_change_sent_sig[car_id] = route_sig
                        self._latest_routes[car_id] = route_points
                        if route_sig:
                            self._latest_route_versions[car_id] = route_sig
                    else:
                        self._latest_routes.pop(car_id, None)
                        self._latest_route_visibility.pop(car_id, None)
                        self._route_change_sent_sig.pop(car_id, None)
                        self._latest_route_versions.pop(car_id, None)
                        self._route_change_sig.pop(car_id, None)
                        self._last_s_start_sig.pop(car_id, None)
                    progress_idx = 0
                    progress_total = 0
                    progress_ratio = 0.0
                    if car_state:
                        try:
                            progress_idx = int(car_state.get("route_progress_idx", 0) or 0)
                        except Exception:
                            progress_idx = 0
                        try:
                            progress_total = int(car_state.get("route_progress_total", 0) or 0)
                        except Exception:
                            progress_total = 0
                        try:
                            progress_ratio = float(car_state.get("route_progress_ratio", 0.0) or 0.0)
                        except Exception:
                            progress_ratio = 0.0
                    cars_on_map.append({
                        "id": map_id,
                        "carId": car_id,
                        "x": x_norm,
                        "y": y_norm,
                        "yaw": yaw,
                        "color": car_color,
                        "status": map_status,
                    })
                    cars_status.append({
                        "class": cls,
                        "car_id": car_id,
                        "color": car_color,
                        "speed": meta.get("speed", 0.0),
                        "battery": battery_val,
                        "cameraIds": camera_ids,
                        "path_future": [],
                        "category": category_val,
                        "resolution": str(resolution_val) if resolution_val is not None else "",
                        "routeChanged": route_changed,
                        "routeVersion": route_sig,
                        "route_progress_idx": progress_idx,
                        "route_progress_total": progress_total,
                        "route_progress_ratio": progress_ratio,
                    })
                else:
                    obstacle_id = f"ob-{ext_id}"
                    source_cams = meta.get("source_cams") or []
                    camera_ids = self._normalize_camera_ids(source_cams)
                    speed_val = meta.get("speed", 0.0)
                    try:
                        speed_val = float(speed_val)
                    except Exception:
                        speed_val = 0.0
                    history = self._obstacle_stop_history.setdefault(ext_id, deque(maxlen=self.obstacle_stop_window))
                    history.append(speed_val)
                    avg_speed = float(np.mean(history)) if history else speed_val
                    active_obstacle_ids.add(ext_id)
                    if avg_speed > self.obstacle_stop_speed_max:
                        continue  # 움직임이 일정 이상이면 UI에 표시하지 않음
                    obstacles_on_map.append({
                        "id": obstacle_id,
                        "obstacleId": obstacle_id,
                        "x": x_norm,
                        "y": y_norm,
                        "kind": self._obstacle_kind(cls),
                    })
                    obstacles_status.append({
                        "id": obstacle_id,
                        "class": cls,
                        "cameraIds": camera_ids,
                    })

        messages.append({
            "type": "carStatus",
            "ts": ts,
            "data": {
                "mode": "snapshot",
                "carsOnMap": cars_on_map,
                "carsStatus": cars_status,
            },
        })

        if route_change_msgs:
            messages.append({
                "type": "carRouteChange",
                "ts": ts,
                "data": {
                    "changes": route_change_msgs,
                },
            })

        obstacle_msg = self._build_obstacle_delta_message(
            obstacles_on_map,
            obstacles_status,
            ts,
        )
        obstacle_snapshot = {
            "type": "obstacleStatus",
            "ts": ts,
            "data": {
                "mode": "snapshot",
                "obstaclesOnMap": obstacles_on_map,
                "obstaclesStatus": obstacles_status,
            },
        }
        if obstacle_msg:
            messages.append(obstacle_msg)

        if self.ws_hub:
            initial_msgs: List[dict] = []
            if self._ui_cam_status:
                initial_msgs.append(self._ui_cam_status)
            # Always include latest traffic light snapshot for new connections.
            try:
                tl_snapshot = {
                    "type": "trafficLightStatus",
                    "ts": ts,
                    "data": {
                        "mode": "snapshot",
                        "trafficLightsOnMap": list(self._ui_traffic_lights_on_map),
                        "trafficLightsStatus": list(self._ui_traffic_light_status.values()),
                    },
                }
                initial_msgs.append(tl_snapshot)
            except Exception:
                pass
            initial_msgs.append({
                "type": "carStatus",
                "ts": ts,
                "data": {
                    "mode": "snapshot",
                    "carsOnMap": cars_on_map,
                    "carsStatus": cars_status,
                },
            })
            initial_msgs.append(obstacle_snapshot)
            if self._latest_routes:
                try:
                    route_snapshot = {
                        "type": "carRouteChange",
                        "ts": ts,
                        "data": {
                            "changes": [
                                {
                                    "carId": car_id,
                                    "newRoute": route,
                                    "routeVersion": self._latest_route_versions.get(car_id),
                                    "visible": bool(self._latest_route_visibility.get(car_id, True)),
                                }
                                for car_id, route in sorted(self._latest_routes.items())
                            ],
                        },
                    }
                    initial_msgs.append(route_snapshot)
                except Exception:
                    pass
            if self._cluster_stage_snapshots:
                for key in sorted(self._cluster_stage_snapshots.keys()):
                    initial_msgs.append(self._cluster_stage_snapshots[key])
            self.ws_hub.set_initial_messages(initial_msgs)

        messages.append(obstacle_snapshot)

        return messages

    def _register_cam_if_needed(self, cam_name: str):
        if cam_name not in self.buffer:
            self.buffer[cam_name] = deque(maxlen=1)
        self.active_cams.add(cam_name)

    def _should_log(self) -> bool:
        now = time.time()
        if now >= self._next_log_ts:
            self._next_log_ts = now + 1.0
            return True
        return False

    def _log_pipeline(self, raw_stats: Dict[str, int], fused: List[dict], tracks: np.ndarray, timestamp: float,
                      timings: Optional[Dict[str, float]] = None):
        """"
        파이프라인 로그 출력
        [Fusion] ts=1766250365.874 total_raw=4 cams=(cam1:4) clusters=4 tracks=4
        fused_sample={"cx": -8.90107380003167, "cy": 12.338851432457503, "length": 4.4, "width": 2.7, "yaw": -138.16341035956063, "score": 0.967749834060669, "source_cams": ["cam1"], "color": null}
        track_sample={"id": 927, "cls": 1, "cx": -12.218371690364483, "cy": -16.645277481758548, "length": 4.4, "width": 2.7, "yaw": 171.02710871210473}
        timings(ms) {"gather": 0.0092999980552122, "fuse": 0.6614999947487377, "track": 5.105099997308571}
        """
        total_raw = sum(raw_stats.values())
        fused_count = len(fused)
        track_count = int(len(tracks)) if tracks is not None else 0
        cams_str = ", ".join([f"{cam}:{cnt}" for cam, cnt in sorted(raw_stats.items())]) or "-"
        print(
            f"[Fusion] ts={timestamp:.3f} total_raw={total_raw} cams=({cams_str}) "
            f"clusters={fused_count} tracks={track_count}"
        )
        debug_parts = []
        rx_qsize = getattr(self, "_last_rx_qsize", None)
        if rx_qsize is not None:
            debug_parts.append(f"rx_q={rx_qsize}")
        if self._last_input_latency_ms is not None:
            debug_parts.append(f"in_delay_ms={self._last_input_latency_ms:.1f}")
        if self._last_tx_send_ms is not None:
            debug_parts.append(f"tx_ms={self._last_tx_send_ms:.1f}")
        if self._last_tx_item_count is not None:
            debug_parts.append(f"tx_items={self._last_tx_item_count}")
        if self._last_tx_payload_bytes is not None:
            debug_parts.append(f"tx_bytes={self._last_tx_payload_bytes}")
        if self._last_tx_interval_ms is not None:
            debug_parts.append(f"tx_interval_ms={self._last_tx_interval_ms:.1f}")
        if debug_parts:
            print(f"  debug {' '.join(debug_parts)}")
        if fused_count:
            sample_keys = ["cx", "cy", "length", "width", "yaw", "score", "source_cams", "color"]
            sample_fused = {k: fused[0][k] for k in sample_keys if k in fused[0]}
            print(f"  fused_sample={json.dumps(sample_fused, ensure_ascii=False)}")
        if track_count:
            row = tracks[0]
            sample_track = {
                "id": int(row[0]),
                "cls": int(row[1]),
                "cx": float(row[2]),
                "cy": float(row[3]),
                "length": float(row[4]),
                "width": float(row[5]),
                "yaw": float(row[6]),
            }
            print(f"  track_sample={json.dumps(sample_track, ensure_ascii=False)}")
        if timings:
            print(f"  timings(ms) {json.dumps(timings)}")

    def _gather_current(self):
        detections = []
        now = time.time()
        for cam, dq in self.buffer.items():
            if not dq:
                continue
            entry = dq[-1]
            ts = float(entry.get("ts", 0.0) or 0.0)
            if (now - ts) > self.buffer_ttl: # 버퍼 유효시간 확인(너무 오래된 거 버림)
                dq.clear()
                continue
            for det in entry["dets"]:
                cls = int(det.get("cls"))
                if cls==0: 
                    det["length"] = vehicle_fixed_length
                    det["width"] = vehicle_fixed_width
                if cls==1: 
                    det["length"] = rubberCone_fixed_length
                    det["width"] = rubberCone_fixed_width
                if cls==2: 
                    det["length"] = barricade_fixed_length
                    det["width"] = barricade_fixed_width
                det_copy = det.copy()
                det_copy["cam"] = cam
                det_copy["ts"] = ts
                detections.append(det_copy)
        return detections

    def _process_command_queue(self): 
        """
        명령 큐 처리 yaw/색상 명령 처리
        """
        if not self.command_queue:
            return
        while True:
            try:
                item = self.command_queue.get_nowait()
            except queue.Empty:
                break
            response = self._handle_command_item(item)
            resp_q = item.get("response")
            if resp_q:
                try:
                    resp_q.put_nowait(response)
                except queue.Full:
                    pass

    async def _handle_ws_message(self, raw_message: str, _websocket) -> Optional[dict]:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("type") != "adminCommand":
            return None

        request_id = payload.get("requestId")
        cmd = payload.get("cmd")
        if not cmd:
            return {
                "type": "adminResponse",
                "requestId": request_id,
                "status": "error",
                "message": "cmd required",
            }
        if not self.command_queue:
            return {
                "type": "adminResponse",
                "requestId": request_id,
                "cmd": cmd,
                "status": "error",
                "message": "command server disabled",
            }

        response_q: queue.Queue = queue.Queue(maxsize=1)
        trimmed_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"type", "requestId"}
        }
        item = {"cmd": cmd, "payload": trimmed_payload, "response": response_q}
        try:
            self.command_queue.put_nowait(item)
        except queue.Full:
            return {
                "type": "adminResponse",
                "requestId": request_id,
                "cmd": cmd,
                "status": "error",
                "message": "server busy",
            }

        timeout = payload.get("timeout", 2.0)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = 2.0

        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(None, response_q.get, True, timeout)
        except queue.Empty:
            response = {"status": "error", "message": "command timeout"}
        return {
            "type": "adminResponse",
            "requestId": request_id,
            "cmd": cmd,
            "response": response,
        }

    def _resolve_external_track_id(self, external_id: int) -> Optional[int]:
        # 우선 외부 ID -> 내부 ID 매핑 테이블을 신뢰한다.
        internal_id = self._external_to_internal.get(external_id)
        if internal_id is not None:
            return internal_id
        # 폴백: track_id_map 값을 스캔해 외부 ID와 매칭되는 내부 ID를 찾는다.
        for iid, ext in self._track_id_map.items():
            if ext == external_id:
                return iid
        return None


    def _handle_command_item(self, item: dict) -> dict: 
        """
        단일 명령 처리 yaw/색상 명령 처리
        """
        cmd = str(item.get("cmd") or "").strip().lower()
        payload = item.get("payload") or {}
        if cmd == "flip_yaw":
            track_id = payload.get("track_id")
            if track_id is None:
                return {"status": "error", "message": "track_id required"}
            try:
                tid = int(track_id)
            except (TypeError, ValueError):
                return {"status": "error", "message": "track_id must be int"}
            resolved_id = self._resolve_external_track_id(tid)
            if resolved_id is None:
                return {"status": "error", "message": f"track {tid} not found"}
            delta = payload.get("delta", 180.0)
            try:
                delta = float(delta)
            except (TypeError, ValueError):
                delta = 180.0
            flipped = self.tracker.force_flip_yaw(resolved_id, offset_deg=delta)
            if flipped:
                print(f"[Command] flipped track {tid} yaw by {delta:.1f}°")
                return {"status": "ok", "track_id": tid, "delta": delta}
            return {"status": "error", "message": f"track {tid} not found"}
        if cmd == "set_color":
            track_id = payload.get("track_id")
            if track_id is None:
                return {"status": "error", "message": "track_id required"}
            try:
                tid = int(track_id)
            except (TypeError, ValueError):
                return {"status": "error", "message": "track_id must be int"}
            resolved_id = self._resolve_external_track_id(tid)
            if resolved_id is None:
                return {"status": "error", "message": f"track {tid} not found"}
            raw_color = payload.get("color")
            normalized_color = normalize_color_label(raw_color)
            if raw_color is not None and normalized_color is None:
                raw_str = str(raw_color).strip().lower()
                if raw_str and raw_str != "none":
                    return {"status": "error", "message": f"invalid color '{raw_color}'"}
                normalized_color = None
            updated = self.tracker.force_set_color(resolved_id, normalized_color)
            if updated:
                print(f"[Command] set track {tid} color -> {normalized_color}")
                return {"status": "ok", "track_id": tid, "color": normalized_color}
            return {"status": "error", "message": f"track {tid} not found"}
        if cmd == "set_yaw":
            track_id = payload.get("track_id")
            if track_id is None:
                return {"status": "error", "message": "track_id required"}
            try:
                tid = int(track_id)
            except (TypeError, ValueError):
                return {"status": "error", "message": "track_id must be int"}
            resolved_id = self._resolve_external_track_id(tid)
            if resolved_id is None:
                return {"status": "error", "message": f"track {tid} not found"}
            raw_yaw = payload.get("yaw")
            try:
                yaw_val = float(raw_yaw)
            except (TypeError, ValueError):
                return {"status": "error", "message": "yaw must be float"}
            updated = self.tracker.force_set_yaw(resolved_id, yaw_val)
            if updated:
                print(f"[Command] set track {tid} yaw -> {yaw_val:.1f}°")
                return {"status": "ok", "track_id": tid, "yaw": yaw_val}
            return {"status": "error", "message": f"track {tid} not found"}
        if cmd == "set_car_count":
            raw_count = payload.get("car_count", payload.get("count"))
            if raw_count is None:
                return {"status": "error", "message": "car_count required"}
            try:
                car_count = int(raw_count)
            except (TypeError, ValueError):
                return {"status": "error", "message": "car_count must be int"}
            if not (self.car_id_min <= car_count <= self.car_id_ceiling):
                return {
                    "status": "error",
                    "message": f"car_count must be between {self.car_id_min} and {self.car_id_ceiling}",
                }
            if car_count == self.car_id_max:
                return {"status": "ok", "car_count": self.car_id_max}
            prev_count = self.car_id_max
            self.car_id_max = car_count
            used_ext_ids = set(self._track_id_map.values())
            if car_count < prev_count:
                disabled_ids = set(range(car_count + 1, self.car_id_ceiling + 1))
                for internal_id, ext_id in list(self._track_id_map.items()):
                    if ext_id in disabled_ids:
                        new_ext = self._assign_external_id(False, None, None, used_ext_ids)
                        self._track_id_map[internal_id] = new_ext
                        self._external_to_internal.pop(ext_id, None)
                        self._external_to_internal[new_ext] = internal_id
                        used_ext_ids.discard(ext_id)
                        used_ext_ids.add(new_ext)
            self._refresh_id_pools(used_ext_ids)
            print(f"[Command] set car_count -> {self.car_id_max}")
            self._broadcast_snapshot_now("set_car_count")
            return {"status": "ok", "car_count": self.car_id_max}
        if cmd == "swap_ids":
            track_id_a = payload.get("track_id_a", payload.get("id_a"))
            track_id_b = payload.get("track_id_b", payload.get("id_b"))
            if track_id_a is None or track_id_b is None:
                return {"status": "error", "message": "track_id_a and track_id_b required"}
            try:
                tid_a = int(track_id_a)
                tid_b = int(track_id_b)
            except (TypeError, ValueError):
                return {"status": "error", "message": "track_id_a/track_id_b must be int"}
            if tid_a == tid_b:
                return {"status": "error", "message": "track IDs must be different"}
            internal_a = self._resolve_external_track_id(tid_a)
            internal_b = self._resolve_external_track_id(tid_b)
            if internal_a is None and internal_b is None:
                return {"status": "error", "message": "track not found"}
            if internal_a is None or internal_b is None:
                return {"status": "error", "message": "both track IDs must exist to swap"}
                internal_target = internal_b if internal_a is None else internal_a
                desired_ext = tid_a if internal_a is None else tid_b
                if desired_ext <= 0:
                    return {"status": "error", "message": "external id must be positive"}
                current_ext = self._track_id_map.get(internal_target)
                if current_ext is None:
                    return {"status": "error", "message": "track not found"}
                if desired_ext == current_ext:
                    return {"status": "ok", "track_id": desired_ext, "previous": current_ext}
                # 완전히 비워낸 뒤 재할당(특히 예약 ID 1이 다른 캐시에 붙잡혀 있을 때)
                self._force_detach_external_id(desired_ext, keep_internal=internal_target)
                self._external_to_internal.pop(current_ext, None)
                self._track_id_map[internal_target] = desired_ext
                self._external_to_internal[desired_ext] = internal_target
                self._recycle_external_id(current_ext)
                used_ext_ids = set(self._track_id_map.values())
                self._refresh_id_pools(used_ext_ids)
                print(f"[Command] set external ID {current_ext} -> {desired_ext}")
                return {"status": "ok", "track_id": desired_ext, "previous": current_ext}
            ext_a = self._track_id_map.get(internal_a)
            ext_b = self._track_id_map.get(internal_b)
            if ext_a is None or ext_b is None:
                return {"status": "error", "message": "track not found"}
            self._force_detach_external_id(ext_a, keep_internal=internal_a)
            self._force_detach_external_id(ext_b, keep_internal=internal_b)
            self._track_id_map[internal_a] = ext_b
            self._track_id_map[internal_b] = ext_a
            self._external_to_internal[ext_a] = internal_b
            self._external_to_internal[ext_b] = internal_a
            print(f"[Command] swapped external IDs {ext_a} <-> {ext_b}")
            self._broadcast_snapshot_now("swap_ids")
            return {"status": "ok", "swapped": [ext_a, ext_b]}
        if cmd == "list_tracks":
            tracks = self.tracker.list_tracks()
            for item in tracks:
                try:
                    internal_id = int(item.get("id"))
                except Exception:
                    continue
                ext_id = self._track_id_map.get(internal_id)
                if ext_id is not None:
                    item["id"] = ext_id
            return {"status": "ok", "tracks": tracks, "count": len(tracks)}
        return {"status": "error", "message": f"unknown command '{cmd}'"}

    def _fuse_boxes(self, raw_detections: List[dict]) -> List[dict]:
        """
        클러스터링 및 대표값 생성: cls, 위치, 크기, 방향, 색상 기반
        raw_detections: [{"cls":...,"cx","cy","length","width","yaw","score","color_hex","cam"}, ...]
        fused_list: [{"cx", "cy", "length","width","yaw","score","source_cams","color","color_votes","cls","cls_votes"}, ...]
        """
        if not raw_detections:
            return []
        boxes = np.array([[d["cls"], d["cx"], d["cy"], d["length"], d["width"], d["yaw"]] for d in raw_detections], dtype=float)
        cams = [d.get("cam", "?") for d in raw_detections]
        colors = [normalize_color_label(d.get("color")) for d in raw_detections]
        clusters, cluster_debug = cluster_by_aabb_iou(
            boxes,
            config=self.cluster_config,
            color_labels=colors,
            track_hints=self._prev_tracks_for_cluster,
            return_debug=True
        )
        self.last_cluster_debug = cluster_debug.get("counters", {}) if isinstance(cluster_debug, dict) else {}
        track_matches = cluster_debug.get("track_matches", []) if isinstance(cluster_debug, dict) else []
        fused_list = []
        for idxs in clusters:
            weight_bias = self._color_weight_biases(raw_detections, idxs)
            iou_support = self._pairwise_iou_support(boxes, idxs)
            extra_weights = self._compose_extra_weights(raw_detections, idxs, weight_bias, iou_support, track_matches)
            rep = fuse_cluster_weighted(
                boxes, cams, idxs, self.cam_xy,
                extra_weights=extra_weights
            )
            extras = self._aggregate_cluster(raw_detections, idxs, rep)
            fused_list.append({
                "cx": float(rep[0]),
                "cy": float(rep[1]),
                "length": float(rep[2]),
                "width": float(rep[3]),
                "yaw": float(rep[4]),
                **extras,
            })
        return fused_list

    def _color_weight_biases(self, detections: List[dict], idxs: List[int]) -> List[float]:
        normalized = [normalize_color_label(detections[i].get("color")) for i in idxs]
        color_counts = Counter([c for c in normalized if c])
        if not color_counts:
            return [1.0] * len(idxs)
        top_color, top_count = color_counts.most_common(1)[0]
        if top_count < max(self.color_bias_min_votes, 1) or self.color_bias_strength <= 0.0:
            return [1.0] * len(idxs)
        boost = 1.0 + self.color_bias_strength
        penalty = max(1.0 - self.color_bias_strength, 0.1)
        biases = []
        for color in normalized:
            if color == top_color:
                biases.append(boost)
            elif color:
                biases.append(penalty)
            else:
                biases.append(1.0)
        return biases

    def _pairwise_iou_support(self, boxes: np.ndarray, idxs: List[int]) -> Dict[int, float]:
        support = {idx: 0.0 for idx in idxs}
        if len(idxs) <= 1:
            return support
        for a in range(len(idxs)):
            ia = idxs[a]
            for b in range(a + 1, len(idxs)):
                ib = idxs[b]
                iou = aabb_iou_axis_aligned(boxes[ia, -5:], boxes[ib, -5:])
                support[ia] += iou
                support[ib] += iou
        return support

    def _compose_extra_weights(
        self,
        detections: List[dict],
        idxs: List[int],
        base_weights: List[float],
        iou_support: Dict[int, float],
        track_matches: List[Tuple[Optional[int], float]],
    ) -> List[float]:
        weights: List[float] = []
        cfg = self.cluster_config
        for local_idx, det_idx in enumerate(idxs):
            w = base_weights[local_idx] if local_idx < len(base_weights) else 1.0
            score = float(detections[det_idx].get("score", 0.0) or 0.0)
            track_score = 0.0
            if track_matches and det_idx < len(track_matches):
                track_score = float(track_matches[det_idx][1] or 0.0)
            iou_sum = float(iou_support.get(det_idx, 0.0))
            w *= (1.0 + cfg.weight_score_scale * score)
            w *= (1.0 + cfg.weight_iou_scale * min(iou_sum, cfg.weight_iou_cap))
            w *= (1.0 + cfg.weight_track_scale * track_score)
            weights.append(max(w, 1e-4))
        return weights

    def _aggregate_cluster(self, detections: List[dict], idxs: List[int], fused_box: Optional[np.ndarray] = None) -> dict:
        subset = [detections[i] for i in idxs]
        if not subset:
            return {"score": 0.0, "source_cams": []}
        score = np.mean([float(d.get("score", 0.0)) for d in subset])
        cls_counts = Counter([int(d.get("cls", -1)) for d in subset if "cls" in d])
        fused_cls = None
        if cls_counts:
            # 차량 관측이 한 번이라도 있으면 차량으로 확정(오검 상쇄)
            if 0 in cls_counts:
                fused_cls = 0
            else:
                fused_cls = cls_counts.most_common(1)[0][0]
        cams = [d.get("cam", "?") for d in subset]
        normalized_colors = [normalize_color_label(d.get("color")) for d in subset]
        valid_colors = [c for c in normalized_colors if c is not None]
        color_counts = Counter(valid_colors)
        color = color_counts.most_common(1)[0][0] if color_counts else None
        hex_candidates = [normalize_color_hex(d.get("color_hex")) for d in subset if d.get("color_hex")]
        hex_candidates = [h for h in hex_candidates if h]
        color_hex = random.choice(hex_candidates) if hex_candidates else None
        return {
            "score": float(score),
            "source_cams": cams,
            "cls": fused_cls,
            "cls_votes": dict(cls_counts),
            "color": color,
            "color_hex": color_hex,
            "color_votes": dict(color_counts),
        }

    def _update_track_meta(self, matches: List[Tuple[int, int]], fused_list: List[dict], track_attrs: Dict[int, dict]):
        """
        트랙 메타데이터 업데이트
        """
        active_ids = set(track_attrs.keys())
        self.track_meta = {tid: meta for tid, meta in self.track_meta.items() if tid in active_ids}
        for tid, attrs in track_attrs.items():
            meta = self.track_meta.setdefault(tid, {})
            vel = attrs.get("velocity")
            if vel and len(vel) == 2:
                scale = 1.0 / max(self.dt, 1e-6)
                vx = float(vel[0]) * scale
                vy = float(vel[1]) * scale
                meta["velocity"] = [vx, vy]
                meta["speed"] = float(np.hypot(vx, vy))
            elif "speed" in attrs:
                meta["speed"] = float(attrs.get("speed", 0.0))
            color = normalize_color_label(attrs.get("color"))
            if color:
                meta["color"] = color
                hex_color = color_label_to_hex(color)
                if hex_color:
                    meta["color_hex"] = hex_color
            elif attrs.get("color_hex"):
                hex_color = normalize_color_hex(attrs.get("color_hex"))
                if hex_color:
                    meta["color_hex"] = hex_color
            else:
                meta.pop("color", None)
                meta.pop("color_hex", None)
            meta["color_locked"] = bool(attrs.get("color_locked"))
            if "color_confidence" in attrs:
                meta["color_confidence"] = attrs["color_confidence"]
        for tid, det_idx in matches:
            if det_idx < 0 or det_idx >= len(fused_list):
                continue
            det = fused_list[det_idx]
            meta = self.track_meta.setdefault(tid, {})
            meta.update({
                "score": float(det.get("score", 0.0)),
                "source_cams": list(det.get("source_cams", [])),
            })
            if not meta.get("color_locked"):
                color = normalize_color_label(det.get("color"))
                if color:
                    meta["color"] = color
                hex_color = normalize_color_hex(det.get("color_hex"))
                if hex_color:
                    meta["color_hex"] = hex_color
            votes = det.get("color_votes")
            if votes:
                meta["color_votes"] = dict(votes)

    def _broadcast_tracks(
        self,
        tracks: np.ndarray,
        ts: float,
        raw_stats: Optional[Dict[str, int]] = None,
        cls_counts: Optional[Dict[int, int]] = None,
        fused: Optional[List[dict]] = None,
        timings: Optional[Dict[str, float]] = None,
    ):
        """
        제어컴, CARLA, WebSocket 허브로 트랙 전송
        """
        mapped_tracks, mapped_meta = self._map_track_ids(tracks, ts)
        payload = self._build_track_payload(mapped_tracks, mapped_meta, ts)
        self._set_last_track_payload(payload)
        payload_bytes = None
        try:
            payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except Exception:
            pass
        if self.track_tx:
            try:
                t_send = time.perf_counter()
                self.track_tx.send(mapped_tracks, mapped_meta, ts)
                send_ms = (time.perf_counter() - t_send) * 1000.0
                self._last_tx_send_ms = send_ms
                self._last_tx_item_count = int(len(mapped_tracks)) if mapped_tracks is not None else 0
                self._last_tx_payload_bytes = payload_bytes
                if self._last_tx_ts is not None:
                    self._last_tx_interval_ms = max(0.0, (time.time() - self._last_tx_ts) * 1000.0)
                self._last_tx_ts = time.time()
                if timings is not None:
                    timings["tx_send"] = send_ms
                    if self._last_tx_interval_ms is not None:
                        timings["tx_interval"] = self._last_tx_interval_ms
                if send_ms > self._tx_warn_ms:
                    proto = getattr(self.track_tx, "protocol", "?")
                    print(f"[TrackBroadcaster] slow send {send_ms:.1f} ms items={self._last_tx_item_count} bytes={payload_bytes} proto={proto}")
            except Exception as exc:
                print(f"[TrackBroadcaster] send error: {exc}")
        if self.ws_hub or self._debug_http_enabled:
            messages = []
            try:
                messages = self._build_ui_messages(mapped_tracks, mapped_meta, ts)
            except Exception as exc:
                print(f"[UI] build message failed: {exc}")
            self._set_last_ui_snapshot(messages, ts)
            if self.ws_hub:
                for message in messages:
                    self.ws_hub.broadcast(message)
        if self.dashboard:
            self._update_dashboard_snapshot(
                raw_stats,
                cls_counts,
                fused,
                mapped_tracks,
                mapped_meta,
                timings,
                ts,
            )

    def _set_last_track_payload(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        with self._debug_lock:
            self._last_track_payload = payload
        self._refresh_ws_initial_messages()

    def _set_last_ui_snapshot(self, messages: List[dict], ts: float) -> None:
        snap = {
            "ts": ts,
            "messages": list(messages or []),
        }
        with self._debug_lock:
            snap["clusterStages"] = list(self._cluster_stage_snapshots.values())
        self._last_ui_snapshot = snap
        self._refresh_ws_initial_messages()

    def _refresh_ws_initial_messages(self) -> None:
        if not self.ws_hub:
            return
        ts = time.time()
        initial: List[dict] = [
            {"type": "carStatus", "ts": ts, "data": {"mode": "snapshot", "carsOnMap": [], "carsStatus": []}},
        ]
        if self._ui_cam_status:
            initial.append(self._ui_cam_status)
        with self._debug_lock:
            ui_snap = self._last_ui_snapshot if isinstance(self._last_ui_snapshot, dict) else None
            cluster = list(self._cluster_stage_snapshots.values())
            latest_routes = dict(self._latest_routes)
            latest_route_versions = dict(self._latest_route_versions)
            latest_route_visibility = dict(self._latest_route_visibility)
        if ui_snap and ui_snap.get("messages"):
            initial.extend(ui_snap.get("messages"))
        if latest_routes:
            try:
                initial.append({
                    "type": "carRouteChange",
                    "ts": ts,
                    "data": {
                        "changes": [
                            {
                                "carId": car_id,
                                "newRoute": route,
                                "routeVersion": latest_route_versions.get(car_id),
                                "visible": bool(latest_route_visibility.get(car_id, True)),
                            }
                            for car_id, route in sorted(latest_routes.items())
                        ],
                    },
                })
            except Exception:
                pass
        if cluster:
            initial.extend(cluster)
        # 트래픽 라이트 스냅샷은 항상 포함시켜 신규 연결에서도 지도가 보이도록 한다.
        has_traffic = any(msg.get("type") == "trafficLightStatus" for msg in initial)
        if not has_traffic:
            traffic_snapshot = {
                "type": "trafficLightStatus",
                "ts": ts,
                "data": {
                    "mode": "snapshot",
                    "trafficLightsOnMap": list(self._ui_traffic_lights_on_map),
                    "trafficLightsStatus": list(self._ui_traffic_light_status.values()),
                },
            }
            self._last_traffic_light_msg = traffic_snapshot
            initial.append(traffic_snapshot)
        try:
            self.ws_hub.set_initial_messages(initial)
        except Exception:
            pass

    def _initialize_ui_snapshots(self) -> None:
        """
        클라이언트가 아무 트랙이 없어도 로딩을 끝낼 수 있도록
        최소한의 빈 스냅샷을 초기 메시지에 채워둔다.
        """
        ts = time.time()
        init_msgs = [
            {"type": "carStatus", "ts": ts, "data": {"mode": "snapshot", "carsOnMap": [], "carsStatus": []}},
        ]
        if self._ui_cam_status is None:
            self._ui_cam_status = self._build_cam_status_message(ts)
        if self._ui_cam_status:
            init_msgs.append(self._ui_cam_status)
        traffic_snapshot = {
            "type": "trafficLightStatus",
            "ts": ts,
            "data": {
                "mode": "snapshot",
                "trafficLightsOnMap": list(self._ui_traffic_lights_on_map),
                "trafficLightsStatus": list(self._ui_traffic_light_status.values()),
            },
        }
        self._last_traffic_light_msg = traffic_snapshot
        init_msgs.append(traffic_snapshot)
        try:
            self.ws_hub.set_initial_messages(init_msgs)
        except Exception:
            pass

    def _start_ws_heartbeat(self, interval: float = 1.0) -> None:
        """
        주기적으로 빈 스냅샷을 브로드캐스트해 클라이언트 로딩을 깨운다.
        """
        if not self.ws_hub or self._ws_heartbeat_thread:
            return

        def _loop():
            while not self._ws_heartbeat_stop.is_set():
                # 최신 스냅샷을 재전송하여 초기 로딩/재연결을 돕는다.
                with self._debug_lock:
                    track_payload = dict(self._last_track_payload) if isinstance(self._last_track_payload, dict) else None
                    ui_snap = dict(self._last_ui_snapshot) if isinstance(self._last_ui_snapshot, dict) else None
                    cluster_msgs = list(self._cluster_stage_snapshots.values())
                # 트랙 스냅샷은 WS로 보내지 않는다.
                # UI 메시지 스냅샷
                if ui_snap and ui_snap.get("messages"):
                    for msg in ui_snap.get("messages"):
                        try:
                            self.ws_hub.broadcast(msg)
                        except Exception:
                            pass
                # 카메라 상태가 없으면 기본값을 한 번 보내준다.
                if not ui_snap or not ui_snap.get("messages"):
                    if self._ui_cam_status:
                        try:
                            self.ws_hub.broadcast(self._ui_cam_status)
                        except Exception:
                            pass
                # 클러스터 스냅샷
                for msg in cluster_msgs:
                    try:
                        self.ws_hub.broadcast(msg)
                    except Exception:
                        pass
                self._ws_heartbeat_stop.wait(interval)

        self._ws_heartbeat_thread = threading.Thread(target=_loop, daemon=True)
        self._ws_heartbeat_thread.start()

    def _update_dashboard_snapshot(
        self,
        raw_stats: Optional[Dict[str, int]],
        cls_counts: Optional[Dict[int, int]],
        fused: Optional[List[dict]],
        mapped_tracks: Optional[np.ndarray],
        mapped_meta: Optional[Dict[int, dict]],
        timings: Optional[Dict[str, float]],
        ts: float,
    ) -> None:
        if not self.dashboard:
            return
        try:
            track_rows = mapped_tracks.tolist() if mapped_tracks is not None else []
        except Exception:
            track_rows = []
        car_status: Dict[int, dict] = {}
        if self.status_state and mapped_tracks is not None:
            for row in mapped_tracks:
                try:
                    ext_id = int(row[0])
                    cls = int(row[1])
                except Exception:
                    continue
                if cls != 0:
                    continue
                try:
                    st = self.status_state.get_car(ext_id)
                except Exception:
                    st = None
                if not st:
                    continue
                try:
                    path_list = st.get("path") or []
                    future_list = st.get("path_future") or []
                    status_entry = {
                        "battery": st.get("battery"),
                        "has_path": bool(path_list),
                        "has_future": bool(future_list),
                        "path_len": len(path_list),
                        "path_future_len": len(future_list),
                        "route_progress_idx": st.get("route_progress_idx"),
                        "route_progress_total": st.get("route_progress_total"),
                        "route_progress_ratio": st.get("route_progress_ratio"),
                        "s_start": st.get("s_start"),
                        "route_version": st.get("route_version"),
                        "route_change_count": st.get("route_change_count", 0),
                    }
                except Exception:
                    continue
                car_status[ext_id] = status_entry
        snap = {
            "ts": ts,
            "raw_total": sum(raw_stats.values()) if raw_stats else 0,
            "raw_per_cam": dict(raw_stats) if raw_stats else {},
            "raw_per_cls": dict(cls_counts) if cls_counts else {},
            "clusters": list(fused) if fused else [],
            "tracks": track_rows,
            "track_meta": dict(mapped_meta) if mapped_meta else {},
            "timings": dict(timings) if timings else {},
            "tracker_metrics": self.tracker.get_last_metrics() if self.tracker else {},
            "ext_to_int": dict(self._external_to_internal),
            "car_status": car_status,
        }
        self.dashboard.update(snap)

    def _broadcast_snapshot_now(self, reason: Optional[str] = None) -> None:
        """
        현재 트랙 상태를 즉시 재전송해 UI/UI 캐시와 싱크를 맞춘다.
        프레임 입력을 기다리지 않고 list_tracks 기반으로 만든다.
        """
        if not self.tracker:
            return
        try:
            items = self.tracker.list_tracks()
        except Exception as exc:
            print(f"[Command] snapshot broadcast failed: {exc}")
            return
        rows: List[List[float]] = []
        for it in items:
            try:
                cls_val = int(it.get("class", it.get("cls", 1)))
                rows.append([
                    float(it.get("id")),
                    cls_val,
                    float(it.get("cx")),
                    float(it.get("cy")),
                    float(it.get("length")),
                    float(it.get("width")),
                    float(it.get("yaw")),
                ])
            except Exception:
                continue
        tracks_arr = np.array(rows, dtype=float) if rows else np.zeros((0, 7), dtype=float)
        ts_now = time.time()
        self._broadcast_tracks(tracks_arr, ts_now)
        if reason:
            print(f"[Command] broadcast snapshot ({reason})")

    def _get_last_track_payload(self) -> dict:
        with self._debug_lock:
            if self._last_track_payload:
                return self._last_track_payload
        return {"type": "global_tracks", "timestamp": None, "items": []}

    def _build_track_payload(self, tracks: np.ndarray, meta: Dict[int, dict], ts: float) -> dict:
        items: List[dict] = []
        if tracks is not None and len(tracks):
            for row in tracks:
                try:
                    tid = int(row[0])
                    cls = int(row[1])
                    cx, cy, L, W, yaw = map(float, row[2:7])
                except Exception:
                    continue
                extra = meta.get(tid, {}) if meta else {}
                vx = vy = 0.0
                velocity = extra.get("velocity")
                if velocity and len(velocity) >= 2:
                    try:
                        vx = float(velocity[0])
                        vy = float(velocity[1])
                    except Exception:
                        vx = vy = 0.0
                speed_val = extra.get("speed")
                if speed_val is None:
                    speed_val = float(np.hypot(vx, vy))
                items.append({
                    "id": tid,
                    "class": cls,
                    "center": [cx, cy, float(extra.get("cz", 0.0))],
                    "length": L,
                    "width": W,
                    "yaw": yaw,
                    "velocity": [vx, vy],
                    "speed": float(speed_val),
                    "score": float(extra.get("score", 0.0)),
                    "sources": list(extra.get("source_cams", [])),
                    "color": extra.get("color"),
                    "color_hex": extra.get("color_hex"),
                    "color_confidence": float(extra.get("color_confidence", 0.0)),
                })
        return {"type": "global_tracks", "timestamp": ts, "items": items}

    def _get_last_ui_snapshot(self) -> dict:
        with self._debug_lock:
            if self._last_ui_snapshot:
                return self._last_ui_snapshot
            cluster = list(self._cluster_stage_snapshots.values())
        return {"ts": None, "messages": [], "clusterStages": cluster}

    def _get_cluster_snapshots(self) -> dict:
        with self._debug_lock:
            snapshots = list(self._cluster_stage_snapshots.values())
        return {"snapshots": snapshots}

    def _start_debug_http(self) -> None:
        if self._debug_http_thread:
            return
        handler = self._make_debug_http_handler()
        try:
            self._debug_httpd = ThreadingHTTPServer(
                (self._debug_http_host, int(self._debug_http_port)),
                handler,
            )
        except Exception as exc:
            print(f"[DebugHTTP] failed to start: {exc}")
            self._debug_httpd = None
            return
        self._debug_http_thread = threading.Thread(
            target=self._debug_httpd.serve_forever,
            daemon=True,
        )
        self._debug_http_thread.start()
        print(f"[DebugHTTP] listening on http://{self._debug_http_host}:{self._debug_http_port}")

    def _make_debug_http_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, obj: dict, status: int = 200) -> None:
                data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                parts = [p for p in parsed.path.split("/") if p]
                if not parts:
                    return self._send_json({"error": "not_found"}, status=404)
                if parts[0] == "tracks":
                    return self._send_json(outer._get_last_track_payload())
                if parts[0] == "ui":
                    return self._send_json(outer._get_last_ui_snapshot())
                if parts[0] == "cluster":
                    return self._send_json(outer._get_cluster_snapshots())
                return self._send_json({"error": "not_found"}, status=404)

            def log_message(self, _format: str, *args):  # noqa: N802
                return

        return Handler

    def _run_tracker_step(self, fused: List[dict]) -> np.ndarray:
        """
        트래커 실행, 트랙/메타 데이터 업데이트, 트랙 반환
        """
        det_rows = []
        det_colors: List[Optional[str]] = []
        gt_yaws: List[Optional[float]] = []
        for det in fused:
            cls_val = int(det.get("cls", 0)) if det.get("cls") is not None else 0
            det_rows.append([
                cls_val,
                det["cx"],
                det["cy"],
                det["length"],
                det["width"],
                det["yaw"],
            ])
            det_colors.append(det.get("color"))
            gt_val = None
            if self.lane_matcher is not None:
                lane_id, yaw, dist = self.lane_matcher.match(det["cx"], det["cy"])
                if lane_id is not None and yaw is not None:
                    det["gt_lane_id"] = lane_id
                    det["gt_lane_dist"] = float(dist)
                    det["gt_yaw"] = float(yaw)
                    gt_val = float(yaw)
            gt_yaws.append(gt_val)

        dets_for_tracker = np.array(det_rows, dtype=float) if det_rows else np.zeros((0, 6), dtype=float)
        
        tracks = self.tracker.update(dets_for_tracker, det_colors, gt_yaws=gt_yaws if det_rows else None)

        track_attrs = self.tracker.get_track_attributes()
        self._update_track_meta(self.tracker.get_latest_matches(), fused, track_attrs)
        tracks = self._dedupe_close_car_tracks(tracks)
        return tracks

    def _dedupe_close_car_tracks(self, tracks: np.ndarray, dist_gate: float = 1.0, iou_gate: float = 0.2) -> np.ndarray:
        """
        차량 트랙 간 거리가 매우 가까우면 하나로 정리한다.
        dist_gate: 중심 거리 게이트(m)
        iou_gate: AABB IoU 게이트
        """
        if tracks is None or len(tracks) <= 1:
            return tracks
        car_rows = [(idx, row) for idx, row in enumerate(tracks) if int(row[1]) == 0]
        to_drop = set()
        track_obj_by_id = {int(t.id): t for t in getattr(self.tracker, "tracks", [])}
        for i in range(len(car_rows)):
            idx_a, row_a = car_rows[i]
            tid_a = int(row_a[0])
            for j in range(i + 1, len(car_rows)):
                idx_b, row_b = car_rows[j]
                tid_b = int(row_b[0])
                cx_a, cy_a, l_a, w_a, yaw_a = map(float, row_a[2:7])
                cx_b, cy_b, l_b, w_b, yaw_b = map(float, row_b[2:7])
                dist = math.hypot(cx_a - cx_b, cy_a - cy_b)
                if dist > dist_gate:
                    continue
                iou_val = aabb_iou_axis_aligned(
                    np.array([cx_a, cy_a, l_a, w_a, yaw_a]),
                    np.array([cx_b, cy_b, l_b, w_b, yaw_b]),
                )
                if iou_val < iou_gate:
                    continue
                ta = track_obj_by_id.get(tid_a)
                tb = track_obj_by_id.get(tid_b)
                age_a = getattr(ta, "age", 0)
                age_b = getattr(tb, "age", 0)
                keep_tid = tid_a if age_a >= age_b else tid_b
                drop_idx = idx_b if keep_tid == tid_a else idx_a
                to_drop.add(drop_idx)
                drop_obj = tb if keep_tid == tid_a else ta
                if drop_obj:
                    drop_obj.state = TrackState.DELETED
        if not to_drop:
            return tracks
        kept = np.array([row for idx, row in enumerate(tracks) if idx not in to_drop], dtype=float)
        return kept

    def main_loop(self):
        timings: Dict[str, float] = {} # 단계별 처리 시간 기록용(ms)
        last = time.time()
        while True:
            self._process_command_queue()
            try:
                self._last_rx_qsize = self.inference_receiver.q.qsize()
                # 인지 결과 수신
                item = self.inference_receiver.q.get_nowait()
                cam = item["cam"]
                source_ts = float(item.get("ts", 0.0) or 0.0)
                ts = time.time()  # 서버 시계로 수신 시각을 강제 사용
                dets = item["dets"]
                self._register_cam_if_needed(cam)
                self.buffer[cam].clear()
                self.buffer[cam].append({"ts": ts, "dets": dets, "source_ts": source_ts})
                self._last_rx_qsize = self.inference_receiver.q.qsize()
            except queue.Empty:
                pass
            
            # 메인 루프 주기 제어
            now = time.time()
            if now - last < self.dt:
                time.sleep(0.005)
                continue
            last = now

            t0 = time.perf_counter()
            raw_dets = self._gather_current() # 인지 결과 수집 [{"cls":...,"cx","cy","length","width","yaw","score","color_hex","cam","ts"}, ...]
            raw_cam_counts: Dict[str, int] = {}
            raw_cls_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
            for det in raw_dets:
                cam = det.get("cam", "?")
                raw_cam_counts[cam] = raw_cam_counts.get(cam, 0) + 1
                try:
                    cls_val = int(det.get("cls", 1))
                    raw_cls_counts[cls_val] = raw_cls_counts.get(cls_val, 0) + 1
                except Exception:
                    raw_cls_counts[1] = raw_cls_counts.get(1, 0) + 1
            frame_ts = time.time()  # 출력/송신 타임스탬프는 항상 서버 시계 기준
            try:
                self._last_input_latency_ms = max(0.0, (time.time() - frame_ts) * 1000.0)
            except Exception:
                self._last_input_latency_ms = None
            timings["gather"] = (time.perf_counter() - t0) * 1000.0
            self._broadcast_cluster_stage("preCluster", raw_dets, frame_ts)
            t1 = time.perf_counter()
            fused = self._fuse_boxes(raw_dets) # 클러스터링 - 융합 [{"cls":...,"cx",...,"score","source_cams"...}, ...]
            timings["fuse"] = (time.perf_counter() - t1) * 1000.0
            self._broadcast_cluster_stage("postCluster", fused, frame_ts)
            t2 = time.perf_counter()
            tracks = self._run_tracker_step(fused) # 트랙 업데이트 np.ndarray([[id, cls, cx, cy, length, width, yaw], ...])
            timings["track"] = (time.perf_counter() - t2) * 1000.0
            self._broadcast_tracks(tracks, frame_ts, raw_cam_counts, raw_cls_counts, fused, timings)
            self._prev_tracks_for_cluster = tracks.copy() if tracks is not None and len(tracks) else None

            if self.log_pipeline and not self.dashboard and self._should_log(): # 1초에 한번 로그
                self._log_pipeline(raw_cam_counts, fused, tracks, frame_ts, timings)
                if self.last_cluster_debug:
                    print(f"[ClusterDebug] {json.dumps(self.last_cluster_debug, ensure_ascii=False)}")
                tracker_metrics = self.tracker.get_last_metrics()
                if tracker_metrics:
                    print(f"[TrackerMetrics] {json.dumps(tracker_metrics, ensure_ascii=False)}")
                if self.debug_assoc_logging:
                    assoc_debug = self.tracker.get_last_debug_info()
                    if assoc_debug:
                        payload = {
                            "clusters": fused,
                            "pred_tracks": assoc_debug.get("predicted_tracks", []),
                            "matches": assoc_debug.get("matched", []),
                            "unmatched_tracks": assoc_debug.get("unmatched_tracks", []),
                            "unmatched_dets": assoc_debug.get("unmatched_detections", []),
                            "unmatched_reasons": assoc_debug.get("unmatched_reasons", {}),
                            "cost_stats": assoc_debug.get("cost_stats"),
                            "cost_shape": assoc_debug.get("cost_matrix_shape"),
                        }
                        print(f"[AssocDebug] {json.dumps(payload, ensure_ascii=False)}")

    def start(self):
        self.inference_receiver.start() # 인지 결과 리시버 시작
        if self.command_server: # 명령 서버 시작
            try:
                self.command_server.start()
            except Exception as exc:
                print(f"[CommandServer] failed to start: {exc}")
        if self._debug_http_enabled:
            self._start_debug_http()
        if self.dashboard:
            self.dashboard.start()
        threading.Thread(target=self.main_loop, daemon=True).start() # 메인 루프 시작
        print("[Fusion] server started")
