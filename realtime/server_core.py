import json
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
from data_io.camera_assets import load_camera_markers, safe_float
from utils.colors import (
    color_label_to_hex,
    hex_to_color_label,
    normalize_color_hex,
    normalize_color_label,
)
from utils.tracking.merge_dist_wbf import (
    cluster_by_aabb_iou,
    fuse_cluster_weighted,
)
from utils.tracking.tracker import SortTracker


class RealtimeFusionServer:
    def __init__(
        self,
        cam_ports: Dict[str, int],
        cam_positions_path: Optional[str] = None,
        local_ply_dir: Optional[str] = None,
        local_lut_dir: Optional[str] = None,
        fps: float = 10.0,
        iou_cluster_thr: float = 0.25,
        single_port: int = 50050,
        tx_host: Optional[str] = None,
        tx_port: int = 60050,
        tx_protocol: str = "udp",
        carla_host: Optional[str] = None,
        carla_port: int = 61000,
        global_ply: str = "real_global_ply.ply",
        vehicle_glb: str = "pointcloud/car.glb",
        tracker_fixed_length: Optional[float] = None,
        tracker_fixed_width: Optional[float] = None,
        command_host: Optional[str] = None,
        command_port: Optional[int] = None,
        flip_ply_y: bool = True,
        ws_host: Optional[str] = "0.0.0.0",
        ws_port: int = 9001,
    ):
        self.fps = fps
        self.dt = 1.0 / max(1e-3, fps)
        self.buffer_ttl = max(self.dt * 2.0, 1.0)
        self.iou_thr = iou_cluster_thr
        self.track_meta: Dict[int, dict] = {}
        self.active_cams = set()
        self.color_bias_strength = 0.3
        self.color_bias_min_votes = 2
        delta = min(max(self.color_bias_strength * 0.25, 0.0), 0.00)
        self.color_cluster_bonus = delta
        self.color_cluster_penalty = delta

        self._log_interval = 1.0
        self._next_log_ts = 0.0
        self.ws_hub: Optional[WebSocketHub] = None
        if ws_host:
            try:
                self.ws_hub = WebSocketHub(ws_host, ws_port, path="/monitor")
                self.ws_hub.start()
            except Exception as exc:
                print(f"[WebSocketHub] init failed: {exc}")
                self.ws_hub = None

        self.receiver = UDPReceiverSingle(single_port)

        self.camera_markers = load_camera_markers(cam_positions_path, local_ply_dir, local_lut_dir)
        self.cam_xy: Dict[str, Tuple[float, float]] = {}
        for marker in self.camera_markers:
            pos = marker.get("position") or {}
            x = safe_float(pos.get("x"))
            y = safe_float(pos.get("y"))
            if x is None or y is None:
                continue
            name = marker.get("name") or marker.get("key")
            if not name:
                continue
            self.cam_xy[str(name)] = (float(x), float(y))
        if not self.cam_xy:
            print("[Fusion] WARN: no camera positions loaded; distance weighting falls back to origin.")

        self.buffer: Dict[str, deque] = {cam: deque(maxlen=1) for cam in cam_ports.keys()}

        self.track_tx = TrackBroadcaster(tx_host, tx_port, tx_protocol) if tx_host else None
        self.carla_tx = TrackBroadcaster(carla_host, carla_port) if carla_host else None

        smooth_window = max(3, min(8, int(round(0.2 / max(self.dt, 1e-3)))))
        self.tracker = SortTracker(
            max_age=20,
            min_hits=20,
            iou_threshold=0.15,
            smooth_window=smooth_window,
        )
        self._log_interval = 1.0
        self._next_log_ts = 0.0

        self.command_queue: Optional[queue.Queue] = None
        self.command_server: Optional[CommandServer] = None
        if command_host and command_port:
            self.command_queue = queue.Queue()
            self.command_server = CommandServer(command_host, command_port, self.command_queue)

        self.tracker_fixed_length = float(tracker_fixed_length) if tracker_fixed_length is not None else None
        self.tracker_fixed_width = float(tracker_fixed_width) if tracker_fixed_width is not None else None

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

    def _build_ui_snapshot(self, tracks: np.ndarray, ts: float) -> Optional[dict]:
        if tracks is None or len(tracks) == 0:
            cars_on_map = []
            cars_status = []
        else:
            cars_on_map = []
            cars_status = []
            for row in tracks:
                tid = int(row[0])
                cls = int(row[1])
                cx = float(row[2])
                cy = float(-row[3])
                length = float(row[4])
                width = float(row[5])
                yaw = float(row[6])

                meta = self.track_meta.get(tid, {})
                color_hex = meta.get("color_hex")
                color_label = hex_to_color_label(color_hex)
                ui_color = color_hex or color_label_to_hex(color_label) or "#22c55e"

                car_id = f"car-{tid}"
                map_id = f"mcar-{tid}"
                x_norm, y_norm = self._world_to_map_xy(cx, cy)

                cars_on_map.append({
                    "id": map_id,
                    "carId": car_id,
                    "x": x_norm,
                    "y": y_norm,
                    "yaw": yaw,
                    "color": color_label,
                    "status": "normal",
                })

                cars_status.append({
                    "class": cls,
                    "id": car_id,
                    "color": ui_color,
                    "speed": meta.get("speed", 0.0),
                    "battery": 100,
                    "fromLabel": "-",
                    "toLabel": "-",
                    "cameraId": None,
                    "routeChanged": False,
                })

        snapshot = {
            "type": "snapshot",
            "payload": {
                "carsOnMap": cars_on_map,
                "carsStatus": cars_status,
                "camerasOnMap": [{"id": "camMarker-1", "cameraId": "cam-1", "x": 0.5, "y": -0.03},
                                 {"id": "camMarker-2", "cameraId": "cam-2", "x": -0.05, "y": 0.5}],
                "camerasStatus": [],
                "incident": None,
                "routeChanges": [],
            },
        }
        return snapshot

    def _register_cam_if_needed(self, cam_name: str):
        if cam_name not in self.buffer:
            self.buffer[cam_name] = deque(maxlen=1)
        self.active_cams.add(cam_name)

    def _should_log(self) -> bool:
        now = time.time()
        if now >= self._next_log_ts:
            self._next_log_ts = now + self._log_interval
            return True
        return False

    def _log_pipeline(self, raw_stats: Dict[str, int], fused: List[dict], tracks: np.ndarray, timestamp: float,
                      timings: Optional[Dict[str, float]] = None):
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
            if (now - ts) > self.buffer_ttl:
                dq.clear()
                continue
            for det in entry["dets"]:
                det_copy = det.copy()
                det_copy["cam"] = cam
                det_copy["ts"] = ts
                detections.append(det_copy)
        return detections

    def _process_command_queue(self):
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

    def _handle_command_item(self, item: dict) -> dict:
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
            delta = payload.get("delta", 180.0)
            try:
                delta = float(delta)
            except (TypeError, ValueError):
                delta = 180.0
            flipped = self.tracker.force_flip_yaw(tid, offset_deg=delta)
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
            raw_color = payload.get("color")
            normalized_color = normalize_color_label(raw_color)
            if raw_color is not None and normalized_color is None:
                raw_str = str(raw_color).strip().lower()
                if raw_str and raw_str != "none":
                    return {"status": "error", "message": f"invalid color '{raw_color}'"}
                normalized_color = None
            updated = self.tracker.force_set_color(tid, normalized_color)
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
            raw_yaw = payload.get("yaw")
            try:
                yaw_val = float(raw_yaw)
            except (TypeError, ValueError):
                return {"status": "error", "message": "yaw must be float"}
            updated = self.tracker.force_set_yaw(tid, yaw_val)
            if updated:
                print(f"[Command] set track {tid} yaw -> {yaw_val:.1f}°")
                return {"status": "ok", "track_id": tid, "yaw": yaw_val}
            return {"status": "error", "message": f"track {tid} not found"}
        if cmd == "list_tracks":
            tracks = self.tracker.list_tracks()
            return {"status": "ok", "tracks": tracks, "count": len(tracks)}
        return {"status": "error", "message": f"unknown command '{cmd}'"}

    def _fuse_boxes(self, raw_detections: List[dict]) -> List[dict]:
        if not raw_detections:
            return []
        boxes = np.array([[d["cls"], d["cx"], d["cy"], d["length"], d["width"], d["yaw"]] for d in raw_detections], dtype=float)
        cams = [d.get("cam", "?") for d in raw_detections]
        colors = [normalize_color_label(d.get("color")) for d in raw_detections]
        clusters = cluster_by_aabb_iou(
            boxes,
            iou_cluster_thr=self.iou_thr,
            color_labels=colors,
            color_bonus=self.color_cluster_bonus,
            color_penalty=self.color_cluster_penalty,
        )
        fused_list = []
        for idxs in clusters:
            weight_bias = self._color_weight_biases(raw_detections, idxs)
            rep = fuse_cluster_weighted(
                boxes, cams, idxs, self.cam_xy,
                d0=5.0, p=2.0, extra_weights=weight_bias
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

    def _aggregate_cluster(self, detections: List[dict], idxs: List[int], fused_box: Optional[np.ndarray] = None) -> dict:
        subset = [detections[i] for i in idxs]
        if not subset:
            return {"cz": 0.0, "pitch": 0.0, "roll": 0.0, "score": 0.0, "source_cams": []}
        score = np.mean([float(d.get("score", 0.0)) for d in subset])
        pitch = np.mean([float(d.get("pitch", 0.0)) for d in subset])
        roll = np.mean([float(d.get("roll", 0.0)) for d in subset])
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
        if fused_box is not None and len(fused_box) >= 4:
            cz = np.mean([float(d.get("cz", 0.0)) for d in subset])
        else:
            cz = np.mean([float(d.get("cz", 0.0)) for d in subset])
        return {
            "cz": float(cz),
            "pitch": float(pitch),
            "roll": float(roll),
            "score": float(score),
            "source_cams": cams,
            "cls": fused_cls,
            "cls_votes": dict(cls_counts),
            "color": color,
            "color_hex": color_hex,
            "color_votes": dict(color_counts),
        }

    def _update_track_meta(self, matches: List[Tuple[int, int]], fused_list: List[dict], track_attrs: Dict[int, dict]):
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
        if self.track_tx:
            self.track_tx.send(tracks, self.track_meta, ts)
        if self.carla_tx:
            self.carla_tx.send(tracks, self.track_meta, ts)
        if self.ws_hub:
            snapshot = self._build_ui_snapshot(tracks, ts)
            if snapshot is not None:
                self.ws_hub.broadcast(snapshot)

    def main_loop(self):
        timings: Dict[str, float] = {}
        last = time.time()
        while True:
            self._process_command_queue()
            try:
                item = self.receiver.q.get_nowait()
                cam = item["cam"]
                ts = float(item.get("ts", 0.0) or time.time())
                dets = item["dets"]
                self._register_cam_if_needed(cam)
                self.buffer[cam].clear()
                self.buffer[cam].append({"ts": ts, "dets": dets})
            except queue.Empty:
                pass

            now = time.time()
            if now - last < self.dt:
                time.sleep(0.005)
                continue
            last = now

            t0 = time.perf_counter()
            raw_dets = self._gather_current()
            timings["gather"] = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            fused = self._fuse_boxes(raw_dets)
            timings["fuse"] = (time.perf_counter() - t1) * 1000.0

            det_rows = []
            det_colors: List[Optional[str]] = []
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
            dets_for_tracker = np.array(det_rows, dtype=float) if det_rows else np.zeros((0, 6), dtype=float)
            t2 = time.perf_counter()
            tracks = self.tracker.update(dets_for_tracker, det_colors)
            timings["track"] = (time.perf_counter() - t2) * 1000.0
            track_attrs = self.tracker.get_track_attributes()
            self._update_track_meta(self.tracker.get_latest_matches(), fused, track_attrs)
            self._broadcast_tracks(tracks, now)

            if self._should_log():
                stats = {}
                for det in raw_dets:
                    cam = det.get("cam", "?")
                    stats[cam] = stats.get(cam, 0) + 1
                self._log_pipeline(stats, fused, tracks, now, timings)

    def start(self):
        self.receiver.start()
        if self.command_server:
            try:
                self.command_server.start()
            except Exception as exc:
                print(f"[CommandServer] failed to start: {exc}")
        threading.Thread(target=self.main_loop, daemon=True).start()
        print("[Fusion] server started")


def parse_cam_ports(text: str) -> Dict[str, int]:
    out = {}
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, port = tok.split(":")
        out[name.strip()] = int(port)
    return out
