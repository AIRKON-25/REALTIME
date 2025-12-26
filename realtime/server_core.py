import asyncio
import heapq
import json
import os
import queue
import random
import threading
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from comms.command_server import CommandServer
from comms.track_broadcaster import TrackBroadcaster
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
from utils.tracking.tracker import SortTracker, TrackerConfigCar, TrackerConfigObstacle
from utils.tracking._constants import ASSOC_CENTER_NORM, ASSOC_CENTER_WEIGHT, IOU_CLUSTER_THR
from realtime.runtime_constants import COLOR_BIAS_STRENGTH, COLOR_BIAS_MIN_VOTES, vehicle_fixed_length, vehicle_fixed_width

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
        class_merge_policy: ClassMergePolicy = ClassMergePolicy.STRICT, # 차-장애물 머지 허용하려면 ClassMergePolicy.ALLOW_CAR_OBSTACLE
        enable_cluster_split: bool = False, # 클러스터 분할 허용하려면 True
        tracker_config_car: Optional[TrackerConfigCar] = None,
        tracker_config_obstacle: Optional[TrackerConfigObstacle] = None,
        assoc_center_weight: Optional[float] = None,
        assoc_center_norm: Optional[float] = None,
        debug_assoc_logging: bool = False,
        log_pipeline: bool = True,
        log_udp_packets: bool = False,
        lane_map_path: Optional[str] = "utils/make_H/lanes.json",
    ):
        self.fps = fps
        self.dt = 1.0 / max(1e-3, fps) # 루프 주기 초
        self.buffer_ttl = max(self.dt * 2.0, 1.0) # 버퍼 유효 시간 초
        self.cluster_config = cluster_config or ClusterConfig(
            iou_cluster_thr=iou_cluster_thr,
            merge_policy=class_merge_policy,
            allow_split_multi_modal=enable_cluster_split,
        )
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
            except Exception as exc:
                print(f"[WebSocketHub] init failed: {exc}")
                self.ws_hub = None

        self._ui_cameras_on_map = [
            {"id": "camMarker-1", "cameraId": "cam-1", "x": -0.01, "y": 0.50},
            {"id": "camMarker-2", "cameraId": "cam-2", "x": 0.25, "y": -0.01},
            {"id": "camMarker-3", "cameraId": "cam-3", "x": 0.75, "y": -0.01},
            {"id": "camMarker-4", "cameraId": "cam-4", "x": 1.01, "y": 0.50},
            {"id": "camMarker-5", "cameraId": "cam-5", "x": 0.75, "y": 1.01},
            {"id": "camMarker-6", "cameraId": "cam-6", "x": 0.25, "y": 1.01},
        ]
        self._ui_cameras_status = [
            {"id": "cam-1", "name": "Camera 1", "streamUrl": "http://192.168.0.101:8080/stream"},
            {"id": "cam-2", "name": "Camera 2", "streamUrl": "http://192.168.0.103:8080/stream"},
            {"id": "cam-3", "name": "Camera 3", "streamUrl": "http://192.168.0.104:8080/stream"},
            {"id": "cam-4", "name": "Camera 4", "streamUrl": "http://192.168.0.106:8080/stream"},
            {"id": "cam-5", "name": "Camera 5", "streamUrl": "http://192.168.0.102:8080/stream"},
            {"id": "cam-6", "name": "Camera 6", "streamUrl": "http://192.168.0.106:8080/stream"},
        ]
        self._ui_traffic_lights_on_map = [
            {"id": "tl-1", "trafficLightId": 1, "x": 0.50, "y": -0.02, "yaw": 0.0},
            {"id": "tl-2", "trafficLightId": 2, "x": 0.50, "y": 1.02, "yaw": 180.0},
        ]
        self._ui_traffic_light_status = {
            int(item["trafficLightId"]): {"trafficLightId": int(item["trafficLightId"]), "light": "red"}
            for item in self._ui_traffic_lights_on_map
        }
        self._ui_traffic_light_map: Dict[str, dict] = {}
        self._ui_traffic_light_status_cache: Dict[int, dict] = {}
        self._ui_cam_status: Optional[dict] = None
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

        self.track_tx = TrackBroadcaster(tx_host, tx_port, tx_protocol) if tx_host else None
        self.carla_tx = TrackBroadcaster(carla_host, carla_port) if carla_host else None

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

    def _select_car_id(
        self,
        track_info: dict,
        now: Optional[float],
        used_ext_ids: Optional[set] = None,
    ) -> Optional[int]:
        if not self._car_id_pool:
            return None
        if now is None:
            now = time.time()
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
            cost, _ = self._compute_reid_cost(track_info, history, now)
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
        is_car = bool(info) and int(info.get("cls", 1)) == 0
        if info:
            self._external_history[ext_id] = info
        if ext_id <= self.car_id_max and is_car:
            self._released_external[ext_id] = {
                "ts": now,
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
            if now - float(info.get("ts", 0.0)) <= self._reid_keep_secs:
                continue
            self._released_external.pop(ext_id, None)
            self._recycle_external_id(ext_id)

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
        now = float(ts) if ts is not None else time.time()
        self._prune_released_ids(now)
        if tracks is None or len(tracks) == 0:
            for tid in list(self._track_id_map.keys()):
                self._stash_released_id(tid, now)
            self._external_to_internal = {}
            return tracks, {}

        active_ids = {int(row[0]) for row in tracks}
        for tid in list(self._track_id_map.keys()):
            if tid not in active_ids:
                self._stash_released_id(tid, now)

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
            info = {
                "internal_id": internal_id,
                "ts": now,
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
                    cost, dist = self._compute_reid_cost(track_info, released_info, now)
                    if not relax_gate and dist > self._reid_dist_gate_m:
                        continue
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
        mapped_meta = {
            id_map[int(row[0])]: self.track_meta.get(int(row[0]), {})
            for row in tracks
        }
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
        current_map = {str(item["trafficLightId"]): item for item in self._ui_traffic_lights_on_map}
        current_status = {int(tid): status for tid, status in self._ui_traffic_light_status.items()}

        if not self._ui_traffic_light_map and not self._ui_traffic_light_status_cache:
            self._ui_traffic_light_map = current_map
            self._ui_traffic_light_status_cache = current_status
            return {
                "type": "trafficLightStatus",
                "ts": ts,
                "data": {
                    "mode": "snapshot",
                    "trafficLightsOnMap": list(current_map.values()),
                    "trafficLightsStatus": list(current_status.values()),
                },
            }

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

        return {"type": "trafficLightStatus", "ts": ts, "data": data}

    @staticmethod
    def _normalize_car_color(label: Optional[str]) -> str:
        allowed = {"red", "green", "blue", "yellow", "purple", "white"}
        if label in allowed:
            return label
        return "red"

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

    def _build_ui_messages(
        self,
        tracks: np.ndarray,
        track_meta: Dict[int, dict],
        ts: float,
    ) -> List[dict]:
        messages: List[dict] = []

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

        if tracks is not None and len(tracks):
            for row in tracks:
                tid = int(row[0])
                cls = int(row[1])
                cx = float(row[2])
                cy = float(-row[3])
                yaw = float(row[6])

                meta = track_meta.get(tid, {})
                color_label = meta.get("color") or hex_to_color_label(meta.get("color_hex"))
                car_color = self._normalize_car_color(color_label)

                x_norm, y_norm = self._world_to_map_xy(cx, cy)

                if cls == 0:
                    car_id = f"car-{tid}"
                    map_id = f"mcar-{tid}"
                    cars_on_map.append({
                        "id": map_id,
                        "carId": car_id,
                        "x": x_norm,
                        "y": y_norm,
                        "yaw": yaw,
                        "color": car_color,
                        "status": "normal",
                    })
                    cars_status.append({
                        "class": cls,
                        "id": car_id,
                        "color": car_color,
                        "speed": meta.get("speed", 0.0),
                        "battery": 100,
                        "fromLabel": "-",
                        "toLabel": "-",
                        "cameraId": None,
                        "routeChanged": False,
                    })
                else:
                    obstacle_id = f"ob-{tid}"
                    source_cams = meta.get("source_cams") or []
                    cam_val = source_cams[0] if source_cams else None
                    camera_id = None
                    if cam_val is not None:
                        cam_str = str(cam_val)
                        if cam_str.startswith("cam-"):
                            camera_id = cam_str
                        elif cam_str.startswith("cam") and len(cam_str) > 3:
                            camera_id = f"cam-{cam_str[3:]}"
                        else:
                            camera_id = cam_str
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
                        "cameraId": camera_id,
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

        obstacle_msg = self._build_obstacle_delta_message(
            obstacles_on_map,
            obstacles_status,
            ts,
        )
        if obstacle_msg:
            messages.append(obstacle_msg)

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
        internal_id = self._external_to_internal.get(external_id)
        if internal_id is not None:
            return internal_id
        if external_id in self._track_id_map:
            return external_id
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
                internal_target = internal_b if internal_a is None else internal_a
                desired_ext = tid_a if internal_a is None else tid_b
                if desired_ext <= 0:
                    return {"status": "error", "message": "external id must be positive"}
                current_ext = self._track_id_map.get(internal_target)
                if current_ext is None:
                    return {"status": "error", "message": "track not found"}
                if desired_ext == current_ext:
                    return {"status": "ok", "track_id": desired_ext, "previous": current_ext}
                self._released_external.pop(desired_ext, None)
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
            self._released_external.pop(ext_a, None)
            self._released_external.pop(ext_b, None)
            self._track_id_map[internal_a] = ext_b
            self._track_id_map[internal_b] = ext_a
            self._external_to_internal[ext_a] = internal_b
            self._external_to_internal[ext_b] = internal_a
            print(f"[Command] swapped external IDs {ext_a} <-> {ext_b}")
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
        fused_cls = cls_counts.most_common(1)[0][0] if cls_counts else None
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

    def _broadcast_tracks(self, tracks: np.ndarray, ts: float):
        """
        제어컴, CARLA, WebSocket 허브로 트랙 전송
        """
        mapped_tracks, mapped_meta = self._map_track_ids(tracks, ts)
        if self.track_tx:
            self.track_tx.send(mapped_tracks, mapped_meta, ts)
        if self.carla_tx:
            self.carla_tx.send(mapped_tracks, mapped_meta, ts)
        if self.ws_hub:
            messages = self._build_ui_messages(mapped_tracks, mapped_meta, ts)
            for message in messages:
                self.ws_hub.broadcast(message)

    def _run_tracker_step(self, fused: List[dict]) -> np.ndarray:
        """
        트래커 실행, 트랙/메타 데이터 업데이트, 트랙 반환
        """
        det_rows = []
        det_colors: List[Optional[str]] = []
        gt_yaws: List[Optional[float]] = []
        for det in fused:
            det_rows.append([
                det["cls"],
                det["cx"],
                det["cy"],
                (self.tracker_fixed_length if self.tracker_fixed_length is not None else det["length"]),
                (self.tracker_fixed_width if self.tracker_fixed_width is not None else det["width"]),
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
        return tracks

    def main_loop(self):
        timings: Dict[str, float] = {} # 단계별 처리 시간 기록용(ms)
        last = time.time()
        while True:
            self._process_command_queue()
            try:
                # 인지 결과 수신
                item = self.inference_receiver.q.get_nowait()
                cam = item["cam"]
                ts = float(item.get("ts", 0.0) or time.time())
                dets = item["dets"]
                self._register_cam_if_needed(cam)
                self.buffer[cam].clear()
                self.buffer[cam].append({"ts": ts, "dets": dets})
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
            timings["gather"] = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            fused = self._fuse_boxes(raw_dets) # 클러스터링 - 융합 [{"cls":...,"cx",...,"score","source_cams"...}, ...]
            timings["fuse"] = (time.perf_counter() - t1) * 1000.0
            t2 = time.perf_counter()
            tracks = self._run_tracker_step(fused) # 트랙 업데이트 np.ndarray([[id, cls, cx, cy, length, width, yaw], ...])
            timings["track"] = (time.perf_counter() - t2) * 1000.0
            self._broadcast_tracks(tracks, now)
            self._prev_tracks_for_cluster = tracks.copy() if tracks is not None and len(tracks) else None

            if self.log_pipeline and self._should_log(): # 1초에 한번 로그
                stats = {}
                for det in raw_dets:
                    cam = det.get("cam", "?")
                    stats[cam] = stats.get(cam, 0) + 1 # 카메라별 원시 검출 개수 집계
                self._log_pipeline(stats, fused, tracks, now, timings)
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
        threading.Thread(target=self.main_loop, daemon=True).start() # 메인 루프 시작
        print("[Fusion] server started")
