import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from comms.status_receiver import StatusReceiver
from realtime.runtime_constants import PATH_PROGRESS_DIST_M


@dataclass
class CarStatus:
    car_id: int
    battery: Optional[float]
    path: List[List[float]] = field(default_factory=list)  # 항상 존재, 없으면 빈 리스트
    path_past: List[List[float]] = field(default_factory=list)  # 지나온 구간
    path_future: List[List[float]] = field(default_factory=list)  # 앞으로 갈 구간
    category: str = ""
    resolution: Optional[float] = None
    s_start: List[Any] = field(default_factory=list)
    s_end: List[Any] = field(default_factory=list)
    end: Optional[str] = None
    ts: float = field(default_factory=time.time)


@dataclass
class TrafficStatus:
    trafficLight_id: int
    light: str
    left_green: Optional[bool]
    ts: float = field(default_factory=time.time)


class StatusState:
    """수신한 차량/경로/신호등/목적지 상태를 메모리에 보관하고 조회."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cars: Dict[int, CarStatus] = {}
        self.traffic: Dict[int, TrafficStatus] = {}

    @staticmethod
    def _normalize_path(path) -> List[List[float]]:
        norm: List[List[float]] = []
        if not path:
            return norm
        for pt in path:
            try:
                x, y = float(pt[0]), float(pt[1])
                norm.append([x, y])
            except Exception:
                continue
        return norm

    @staticmethod
    def _extract_path_points(obj) -> List[List[float]]:
        # 허용: [[x,y], ...] 리스트, dict에 poses/points 형태, dict에 path 키 등
        if not obj:
            return []
        if isinstance(obj, list):
            # list of list or list of dict
            if obj and isinstance(obj[0], dict):
                pts = []
                for it in obj:
                    # dict with x/y
                    if "x" in it and "y" in it:
                        try:
                            pts.append([float(it["x"]), float(it["y"])])
                            continue
                        except Exception:
                            pass
                    # dict with pose.position.{x,y}
                    pose = it.get("pose") if isinstance(it, dict) else None
                    if pose and "position" in pose:
                        pos = pose["position"]
                        try:
                            pts.append([float(pos.get("x")), float(pos.get("y"))])
                            continue
                        except Exception:
                            pass
                return StatusState._normalize_path(pts)
            return StatusState._normalize_path(obj)
        if isinstance(obj, dict):
            if "poses" in obj and isinstance(obj["poses"], list):
                return StatusState._extract_path_points(obj["poses"])
            if "path" in obj:
                return StatusState._extract_path_points(obj["path"])
        return []

    @staticmethod
    def _nearest_path_index(path: List[List[float]], pos: Tuple[float, float]) -> Tuple[Optional[int], float]:
        if not path:
            return None, float("inf")
        px, py = float(pos[0]), float(pos[1])
        best_idx: Optional[int] = None
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
        return best_idx, best_dist

    def update_car(self, car_id: int, battery: Optional[float], ts: Optional[float] = None) -> None:
        now = ts or time.time()
        with self._lock:
            existing = self.cars.get(int(car_id))
            path = list(existing.path) if existing else []
            path_past = list(existing.path_past) if existing else []
            path_future = list(existing.path_future) if existing else []
            category = existing.category if existing else ""
            resolution = existing.resolution if existing else None
            s_start = list(existing.s_start) if existing else []
            s_end = list(existing.s_end) if existing else []
            end = existing.end if existing else None
            self.cars[int(car_id)] = CarStatus(
                int(car_id),
                battery,
                path=path,
                path_past=path_past,
                path_future=path_future,
                category=category,
                resolution=resolution,
                s_start=s_start,
                s_end=s_end,
                end=end,
                ts=now,
            )

    def update_route(
        self,
        car_id: int,
        path,
        ts: Optional[float] = None,
        category: str = "",
        resolution: Optional[float] = None,
        s_start: Optional[List[Any]] = None,
        s_end: Optional[List[Any]] = None,
    ) -> None:
        now = ts or time.time()
        path_list = self._normalize_path(path)
        opt_start = list(s_start) if s_start else []
        opt_end = list(s_end) if s_end else []
        with self._lock:
            existing = self.cars.get(int(car_id))
            battery = existing.battery if existing else None
            end = existing.end if existing else None
            self.cars[int(car_id)] = CarStatus(
                int(car_id),
                battery,
                path=path_list,
                path_past=[],
                path_future=list(path_list),
                category=category or (existing.category if existing else ""),
                resolution=resolution if resolution is not None else (existing.resolution if existing else None),
                s_start=opt_start if opt_start else (list(existing.s_start) if existing else []),
                s_end=opt_end if opt_end else (list(existing.s_end) if existing else []),
                end=end,
                ts=now,
            )

    def update_traffic(self, tl_id: int, light: str, left_green: Optional[bool], ts: Optional[float] = None) -> None:
        now = ts or time.time()
        with self._lock:
            self.traffic[int(tl_id)] = TrafficStatus(int(tl_id), light, left_green, ts=now)

    def update_dest(self, car_id: int, dest_name: str, ts: Optional[float] = None) -> None:
        now = ts or time.time()
        with self._lock:
            existing = self.cars.get(int(car_id))
            battery = existing.battery if existing else None
            path = list(existing.path) if existing else []
            path_past = list(existing.path_past) if existing else []
            path_future = list(existing.path_future) if existing else []
            category = existing.category if existing else ""
            resolution = existing.resolution if existing else None
            s_start = list(existing.s_start) if existing else []
            s_end = list(existing.s_end) if existing else []
            self.cars[int(car_id)] = CarStatus(
                int(car_id),
                battery,
                path=path,
                path_past=path_past,
                path_future=path_future,
                category=category,
                resolution=resolution,
                s_start=s_start,
                s_end=s_end,
                end=dest_name,
                ts=now,
            )

    def _asdict(self, obj: Any) -> Any:
        return asdict(obj) if obj is not None else None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cars": {k: asdict(v) for k, v in self.cars.items()},
                "traffic": {k: asdict(v) for k, v in self.traffic.items()},
            }

    def get_car(self, car_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._asdict(self.cars.get(int(car_id)))

    def get_route(self, car_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            car = self.cars.get(int(car_id))
            if not car:
                return None
            return {
                "car_id": car.car_id,
                "path": car.path,
                "path_past": car.path_past,
                "path_future": car.path_future,
                "category": car.category,
                "resolution": car.resolution,
                "s_start": car.s_start,
                "s_end": car.s_end,
                "ts": car.ts,
            }

    def get_traffic(self, tl_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._asdict(self.traffic.get(int(tl_id)))

    def get_dest(self, car_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            car = self.cars.get(int(car_id))
            if not car or car.end is None:
                return None
            return {"car_id": car.car_id, "end": car.end, "ts": car.ts}

    def advance_path_progress(
        self,
        car_id: int,
        position: Tuple[float, float],
        gate_m: float = PATH_PROGRESS_DIST_M,
    ) -> Optional[Dict[str, Any]]:
        """
        현재 위치가 경로의 앞 구간에 충분히 근접하면 path_past/path_future를 갱신.
        """
        gate = max(0.0, float(gate_m))
        with self._lock:
            car = self.cars.get(int(car_id))
            if not car or not car.path_future:
                return None
            idx, dist = self._nearest_path_index(car.path_future, position)
            if idx is None or dist > gate:
                return None
            consumed = car.path_future[: idx + 1]
            car.path_past.extend(consumed)
            car.path_future = car.path_future[idx + 1 :]
            car.ts = time.time()
            return self._asdict(car)


class StatusServer:
    """
    udp_status_broadcaster.py가 보내는 상태 패킷을 받아 StatusState에 저장.
    필요 시 간단한 HTTP GET API로 조회도 제공.
    """

    def __init__(
        self,
        udp_host: str = "0.0.0.0",
        udp_port: int = 60070,
        log_packets: bool = False,
        http_host: Optional[str] = None,
        http_port: Optional[int] = None,
    ) -> None:
        self.state = StatusState()
        self.receiver = StatusReceiver(
            host=udp_host,
            port=udp_port,
            log_packets=log_packets,
            on_message=self._on_message,
        )
        self.http_host = http_host
        self.http_port = http_port
        self.httpd: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.receiver.start()
        if self.http_host and self.http_port:
            self._start_http()

    def stop(self) -> None:
        self.receiver.stop()
        if self.httpd:
            try:
                self.httpd.shutdown()
            except Exception:
                pass
        if self._http_thread:
            self._http_thread.join(timeout=0.5)

    # 내부 처리
    def _on_message(self, payload: dict, addr: Tuple[str, int]) -> None:
        msg_type = payload.get("type")
        now = time.time()
        if msg_type == "carStatus":
            try:
                cid = int(payload.get("car_id"))
                battery = payload.get("battery")
                battery_val = float(battery) if battery is not None else None
                self.state.update_car(cid, battery_val, ts=now)
            except Exception:
                return
        elif msg_type == "route":
            # 신구 포맷 모두 처리
            pl = payload.get("payload")
            resolution = payload.get("resolution")
            if isinstance(pl, dict):
                # 구 포맷: pl["planning"]에 [{"car_id":..,"path":[[x,y],...]}]
                planning = pl.get("planning") or []
                for item in planning:
                    try:
                        cid = int(item.get("car_id"))
                    except Exception:
                        continue
                    path = item.get("path") or []
                    path_list = self.state._normalize_path(path)
                    self.state.update_route(cid, path_list, ts=now)
            elif isinstance(pl, list):
                # 신규 포맷: payload=[{vid, category, optional{s_start,s_end}, planning}, ...]
                for item in pl:
                    try:
                        cid = int(item.get("vid") or item.get("car_id"))
                    except Exception:
                        continue
                    category = item.get("category") or ""
                    optional = item.get("optional") or {}
                    s_start = optional.get("s_start") or []
                    s_end = optional.get("s_end") or []
                    path_raw = item.get("planning")
                    path_list = self.state._extract_path_points(path_raw)
                    self.state.update_route(
                        cid,
                        path_list,
                        ts=now,
                        category=category,
                        resolution=resolution,
                        s_start=s_start,
                        s_end=s_end,
                    )
        elif msg_type == "trafficLight":
            try:
                tl_id = int(payload.get("trafficLight_id"))
                light = str(payload.get("light"))
                left_green = payload.get("left_green")
                if left_green is not None:
                    left_green = bool(left_green)
                self.state.update_traffic(tl_id, light, left_green, ts=now)
            except Exception:
                return
        elif msg_type == "end":
            try:
                cid = int(payload.get("car_id"))
                dest_name = str(payload.get("end"))
                self.state.update_dest(cid, dest_name, ts=now)
            except Exception:
                return

    def _start_http(self) -> None:
        handler = self._make_handler()
        self.httpd = ThreadingHTTPServer((self.http_host, int(self.http_port)), handler)
        self._http_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._http_thread.start()
        print(f"[StatusServer] HTTP listening on {self.http_host}:{self.http_port}")

    def _make_handler(self):
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, obj: Any, status: int = 200) -> None:
                data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
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
                if parts[0] == "snapshot":
                    return self._send_json(state.snapshot())
                if len(parts) == 2 and parts[0] == "car":
                    try:
                        cid = int(parts[1])
                    except ValueError:
                        return self._send_json({"error": "bad_car_id"}, status=400)
                    res = state.get_car(cid)
                    return self._send_json(res or {})
                if len(parts) == 2 and parts[0] == "route":
                    try:
                        cid = int(parts[1])
                    except ValueError:
                        return self._send_json({"error": "bad_car_id"}, status=400)
                    res = state.get_route(cid)
                    return self._send_json(res or {})
                if len(parts) == 2 and parts[0] == "traffic":
                    try:
                        tid = int(parts[1])
                    except ValueError:
                        return self._send_json({"error": "bad_traffic_id"}, status=400)
                    res = state.get_traffic(tid)
                    return self._send_json(res or {})
                if len(parts) == 2 and parts[0] == "dest":
                    try:
                        cid = int(parts[1])
                    except ValueError:
                        return self._send_json({"error": "bad_car_id"}, status=400)
                    res = state.get_dest(cid)
                    return self._send_json(res or {})
                return self._send_json({"error": "not_found"}, status=404)

            def log_message(self, _format: str, *args: Any) -> None:
                # 기본 http.server 로그는 지운다
                return

        return Handler
