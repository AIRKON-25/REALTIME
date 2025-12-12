import argparse
import json
import queue
import random
import re
import socket
import threading
import time
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import asyncio
import websockets  # pip install websockets
import numpy as np
import open3d as o3d

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None

import os, gzip, json, csv, hashlib, threading, queue, time  # noqa: F401
from datetime import datetime, timezone  # noqa: F401

from utils.merge_dist_wbf import (
    cluster_by_aabb_iou, fuse_cluster_weighted
)
from utils.tracker import SortTracker

# ----- 색상 관련 유틸 -----
COLOR_LABELS = ("red", "pink", "green", "white", "yellow", "purple")
VALID_COLORS = {color: color for color in COLOR_LABELS}
COLOR_HEX_MAP = {
    "red": "#f52629",
    "pink": "#f53e96",
    "green": "#48ad0d",
    "white": "#f0f0f0",
    "yellow": "#ffdd00",
    "purple": "#781de7",
}


def normalize_color_hex(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not text.startswith("#"):
        text = f"#{text}"
    if re.match(r"^#[0-9a-fA-F]{6}$", text):
        return text.lower()
    return None


def normalize_color_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    color = str(value).strip().lower()
    if not color or color == "none":
        return None
    return VALID_COLORS.get(color)


def color_label_to_hex(color: Optional[str]) -> Optional[str]:
    if not color:
        return None
    return COLOR_HEX_MAP.get(color)


def _normalize_base_url(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    base = str(value).strip()
    if not base:
        return None
    return base.rstrip("/")


# ---일단 저장용(현재는 사용 안 함, 그대로 둠)---
def _md5sum(path, bufsize=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _slugify_label(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", str(text)).strip("-").lower()
    return slug or "cam"


def _dedup_slug(base: str, used: set) -> str:
    slug = base
    idx = 2
    while slug in used:
        slug = f"{base}-{idx}"
        idx += 1
    used.add(slug)
    return slug


def _filter_candidates_by_cam_id(candidates, cam_id: Optional[int]):
    if cam_id is None:
        return candidates
    pattern = re.compile(rf"^cam_?{cam_id}(?!\d)", re.IGNORECASE)
    filtered = [c for c in candidates if pattern.search(c.name)]
    return filtered or candidates


def _guess_local_ply(root: Path, cam_id: Optional[int], name: Optional[str]) -> Optional[Path]:
    if cam_id is None:
        return None
    candidates = []
    patterns = [
        f"cam_{cam_id}_*.ply",
        f"cam{cam_id}_*.ply",
    ]
    if name:
        name_slug = _slugify_label(name).replace("-", "_")
        patterns.append(f"{name_slug}*.ply")
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    if not candidates:
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    candidates = _filter_candidates_by_cam_id(candidates, cam_id)
    candidates.sort()
    return candidates[0].resolve()


def _guess_local_lut(root: Path, cam_id: Optional[int], name: Optional[str]) -> Optional[Path]:
    if cam_id is None:
        return None
    candidates = []
    patterns = [
        f"cam_{cam_id}_*.npz",
        f"cam{cam_id}_*.npz",
    ]
    if name:
        name_slug = _slugify_label(name).replace("-", "_")
        patterns.append(f"{name_slug}*.npz")
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    if not candidates:
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    candidates = _filter_candidates_by_cam_id(candidates, cam_id)
    candidates.sort()
    return candidates[0].resolve()


def _lut_mask_from_obj(obj: dict) -> Optional[np.ndarray]:
    for key in ("ground_valid_mask", "valid_mask", "floor_mask"):
        if key in obj:
            mask = np.asarray(obj[key]).astype(bool)
            if "X" in obj and mask.shape == np.asarray(obj["X"]).shape:
                return mask
    if all(k in obj for k in ("X", "Y", "Z")):
        X = np.asarray(obj["X"])
        Y = np.asarray(obj["Y"])
        Z = np.asarray(obj["Z"])
        return np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    return None


def _load_visible_xyz_from_lut(path: Path) -> Optional[Tuple[np.ndarray, bool]]:
    try:
        with np.load(str(path)) as data:
            if not {"X", "Y", "Z"}.issubset(set(data.files)):
                print(f"[Fusion] LUT missing XYZ -> {path}")
                return None
            mask = _lut_mask_from_obj(data)
            if mask is None:
                print(f"[Fusion] LUT mask missing -> {path}")
                return None
            X = np.asarray(data["X"], dtype=np.float32)
            Y = np.asarray(data["Y"], dtype=np.float32)
            Z = np.asarray(data["Z"], dtype=np.float32)
            xyz = np.stack([X[mask], Y[mask], Z[mask]], axis=1).astype(np.float32)
            colors = None
            has_rgb = False
            if {"R", "G", "B"}.issubset(set(data.files)):
                R = np.asarray(data["R"], dtype=np.float32)
                G = np.asarray(data["G"], dtype=np.float32)
                B = np.asarray(data["B"], dtype=np.float32)
                try:
                    colors = np.stack([R[mask], G[mask], B[mask]], axis=1).astype(np.float32)
                    max_val = float(np.nanmax(colors)) if colors.size else 0.0
                    if max_val > 1.01:
                        colors /= 255.0
                    has_rgb = True
                except Exception as exc:  # pragma: no cover
                    print(f"[Fusion] LUT RGB load failed for {path}: {exc}")
                    colors = None
                    has_rgb = False
    except Exception as exc:
        print(f"[Fusion] failed to load LUT {path}: {exc}")
        return None
    if xyz.size == 0:
        return None
    if colors is not None and colors.shape[0] == xyz.shape[0]:
        pts = np.concatenate([xyz, colors], axis=1)
        return pts, True
    return xyz, False


def load_camera_markers(path: Optional[str], local_ply_root: Optional[str] = None,
                        local_lut_root: Optional[str] = None) -> List[dict]:
    markers: List[dict] = []
    if not path:
        return markers
    json_path = Path(path)
    if not json_path.exists():
        print(f"[Fusion] camera position file not found: {json_path}")
        return markers
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Fusion] failed to read camera position file {json_path}: {exc}")
        return markers

    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        cams = raw.get("cameras")
        if isinstance(cams, list):
            entries = cams
        else:
            entries = [raw]
    else:
        print(f"[Fusion] camera position file {json_path} has unexpected format")
        return markers

    base_dir = json_path.parent
    ply_root = Path(local_ply_root).resolve() if local_ply_root else None
    lut_root = Path(local_lut_root).resolve() if local_lut_root else None
    used_slugs = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("label")
        camera_id = entry.get("camera_id", entry.get("id"))
        if camera_id is not None:
            try:
                camera_id = int(camera_id)
            except (TypeError, ValueError):
                camera_id = None
        if camera_id is None and name:
            nums = re.findall(r"\d+", str(name))
            if nums:
                try:
                    camera_id = int(nums[-1])
                except ValueError:
                    camera_id = None

        pos_src = entry.get("pos") if isinstance(entry.get("pos"), dict) else {}
        x = _safe_float(pos_src.get("x", entry.get("x")))
        y = _safe_float(pos_src.get("y", entry.get("y")))
        if x is None or y is None:
            continue
        z = _safe_float(pos_src.get("z", entry.get("z")), 0.0)

        rot_src = entry.get("rot") if isinstance(entry.get("rot"), dict) else {}
        rotation = {
            "pitch": _safe_float(rot_src.get("pitch", entry.get("pitch")), 0.0),
            "yaw": _safe_float(rot_src.get("yaw", entry.get("yaw")), 0.0),
            "roll": _safe_float(rot_src.get("roll", entry.get("roll")), 0.0),
        }

        local_ref = entry.get("local_ply") or entry.get("visible_ply")
        ply_path = None
        if isinstance(local_ref, str) and local_ref:
            candidate = Path(local_ref)
            ply_path = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        if (ply_path is None or not ply_path.exists()) and ply_root and ply_root.exists():
            guessed = _guess_local_ply(ply_root, camera_id, name)
            if guessed and guessed.exists():
                ply_path = guessed
        if ply_path and not ply_path.exists():
            print(f"[Fusion] WARN: local ply for {name or camera_id} missing: {ply_path}")
            ply_path = None

        lut_ref = entry.get("local_lut") or entry.get("lut") or entry.get("lut_npz")
        lut_path = None
        if isinstance(lut_ref, str) and lut_ref:
            candidate = Path(lut_ref)
            lut_path = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        if (lut_path is None or not lut_path.exists()) and lut_root and lut_root.exists():
            guessed_lut = _guess_local_lut(lut_root, camera_id, name)
            if guessed_lut and guessed_lut.exists():
                lut_path = guessed_lut
        if lut_path and not lut_path.exists():
            print(f"[Fusion] WARN: local LUT for {name or camera_id} missing: {lut_path}")
            lut_path = None

        display_name = str(name or (f"cam{camera_id}" if camera_id is not None else f"marker{idx+1}"))
        slug = _dedup_slug(_slugify_label(display_name), used_slugs)
        overlay_base_url = entry.get("overlay_base_url") or entry.get("overlay_url") or entry.get("overlay_host")
        if overlay_base_url is not None:
            overlay_base_url = str(overlay_base_url).strip()
            if overlay_base_url:
                overlay_base_url = overlay_base_url.rstrip("/")
            else:
                overlay_base_url = None
        markers.append({
            "key": slug,
            "name": display_name,
            "camera_id": camera_id,
            "position": {"x": float(x), "y": float(y), "z": float(z)},
            "rotation": rotation,
            "local_ply": str(ply_path) if ply_path else None,
            "local_lut": str(lut_path) if lut_path else None,
            "overlay_base_url": overlay_base_url,
        })

    if not markers:
        print(f"[Fusion] camera position file {json_path} contained no usable entries")
    return markers


class GroundHeightLookup:
    """
    간단한 지면 높이 질의용: global ply의 XY→Z 근접 값을 가져온다.
    """
    def __init__(self, ply_path: Optional[str], *, flip_y: bool = False, max_points: int = 500_000):
        self.enabled = False
        self.xy: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.tree = None

        if not ply_path:
            return
        path = Path(ply_path)
        if not path.exists():
            print(f"[GroundZ] ply not found: {path}")
            return
        try:
            cloud = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(cloud.points, dtype=np.float32)
        except Exception as exc:
            print(f"[GroundZ] failed to load ply {path}: {exc}")
            return
        if pts.size == 0:
            print(f"[GroundZ] empty ply: {path}")
            return
        if flip_y:
            pts[:, 1] *= -1.0
        if max_points and len(pts) > max_points:
            stride = max(1, int(len(pts) / max_points))
            pts = pts[::stride]
            print(f"[GroundZ] subsampled ply to {len(pts)} points (stride={stride})")
        self.xy = pts[:, :2].astype(np.float32, copy=False)
        self.z = pts[:, 2].astype(np.float32, copy=False)
        if cKDTree is not None:
            try:
                self.tree = cKDTree(self.xy)
            except Exception as exc:
                print(f"[GroundZ] cKDTree build failed: {exc}")
                self.tree = None
        self.enabled = True
        print(f"[GroundZ] loaded {len(self.z)} points from {path.name} (flip_y={flip_y}, kdtree={'yes' if self.tree else 'no'})")

    def query(self, x: float, y: float, *, default: float = 0.0, k: int = 5) -> float:
        if not self.enabled or self.xy is None or self.z is None:
            return float(default)
        try:
            if self.tree is not None:
                k_use = min(max(1, int(k)), len(self.z))
                dists, idxs = self.tree.query([float(x), float(y)], k=k_use)
                idx_arr = np.atleast_1d(idxs)
                z_vals = self.z[idx_arr]
                if k_use > 1:
                    dist_arr = np.atleast_1d(dists)
                    weights = 1.0 / np.maximum(dist_arr, 1e-3)
                    return float(np.average(z_vals, weights=weights))
                return float(z_vals[0])
            diff = self.xy - np.array([float(x), float(y)], dtype=np.float32)
            d2 = np.sum(diff * diff, axis=1)
            idx = int(np.argmin(d2))
            return float(self.z[idx])
        except Exception:
            return float(default)


class TrackBroadcaster:
    """
    UDP/TCP 브로드캐스터.
    track 패킷 포맷:
    {
        "type": "global_tracks",
        "timestamp": unix_ts,
        "items": [
            {"id": tid, "class": cls, "center": [cx, cy, cz], "length": L,
             "width": W, "yaw": yaw_deg, "pitch": pitch, "roll": roll,
             "score": score, "sources": ["cam1", ...]}
        ]
    }
    """
    def __init__(self, host: str, port: int, protocol: str = "udp"):
        self.host = host
        self.port = int(port)
        self.protocol = protocol.lower()
        if self.protocol not in ("udp", "tcp"):
            raise ValueError("protocol must be 'udp' or 'tcp'")
        if self.protocol == "udp":
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.addr = (self.host, self.port)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.addr = None

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def send(self, tracks: np.ndarray, extras: Dict[int, dict], ts: float):
        payload = {
            "type": "global_tracks",
            "timestamp": ts,
            "items": []
        }
        if tracks is not None and len(tracks):
            for row in tracks:
                tid = int(row[0])
                cls = int(row[1])
                cx, cy, L, W, yaw = map(float, row[2:7])
                extra = extras.get(tid, {})
                payload["items"].append({
                    "id": tid,
                    "class": cls,
                    "center": [cx, cy, float(extra.get("cz", 0.0))],
                    "length": L,
                    "width": W,
                    "yaw": yaw,
                    "pitch": float(extra.get("pitch", 0.0)),
                    "roll": float(extra.get("roll", 0.0)),
                    "score": float(extra.get("score", 0.0)),
                    "sources": list(extra.get("source_cams", [])),
                    "color": extra.get("color"),
                    "color_hex": extra.get("color_hex"),
                    "color_confidence": float(extra.get("color_confidence", 0.0)),
                })
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            if self.protocol == "udp":
                self.sock.sendto(data, self.addr)
            else:
                self.sock.sendall(data + b"\n")
        except Exception as e:
            print(f"[TrackBroadcaster] send error: {e}")
class WebSocketHub:
    """
    프론트엔드(React)와 실시간으로 JSON을 주고받는 간단한 WebSocket 서버.
    ws://<host>:<port>/monitor 로 접속하면 된다.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 9001, path: str = "/monitor"):
        self.host = host
        self.port = int(port)
        self.path = path
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: "set[websockets.WebSocketServerProtocol]" = set()
        self._thread: Optional[threading.Thread] = None

    async def _handler(self, websocket):
        # websockets 15.x에서는 websocket.path 가 None 또는 '' 인 경우가 많음
        # 따라서 경로 체크는 제거하는 것이 안전하다.
        self._clients.add(websocket)
        print(f"[WebSocketHub] client connected ({len(self._clients)} total)")

        try:
            async for _ in websocket:
                pass
        except Exception as exc:
            print(f"[WebSocketHub] client error: {exc}")
        finally:
            self._clients.discard(websocket)
            print(f"[WebSocketHub] client disconnected ({len(self._clients)} remaining)")

    async def _run_server(self):
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"[WebSocketHub] listening on ws://{self.host}:{self.port}{self.path}")
            await asyncio.Future()  # run forever

    def start(self):
        if self._thread:
            return

        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_server())

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def broadcast(self, message: dict):
        """
        모든 연결된 클라이언트에 JSON 메시지를 전송한다.
        message는 dict로 주면 내부에서 json.dumps 한다.
        """
        if not self._loop or not self._clients:
            return
        data = json.dumps(message, ensure_ascii=False)

        async def _send_all():
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

        asyncio.run_coroutine_threadsafe(_send_all(), self._loop)


class CommandServer:
    """
    간단한 TCP 명령 서버. 각 연결은 JSON 한 줄을 보내고 응답을 받는다.
    """
    def __init__(self, host: str, port: int, command_queue: "queue.Queue[dict]", response_timeout: float = 2.0):
        self.host = host
        self.port = int(port)
        self.q = command_queue
        self.response_timeout = response_timeout
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._thread.start()
        print(f"[CommandServer] listening on {self.host}:{self.port}")

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _serve_forever(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self.host, self.port))
                srv.listen()
                self._sock = srv
                while self._running:
                    try:
                        conn, addr = srv.accept()
                    except OSError:
                        break
                    threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
        except Exception as exc:
            print(f"[CommandServer] server error: {exc}")

    def _handle_client(self, conn: socket.socket, addr):
        with conn:
            try:
                data = self._recv_all(conn)
                if not data:
                    return
                response = self._process_payload(data)
            except Exception as exc:
                response = {"status": "error", "message": str(exc)}
            try:
                payload = (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
                conn.sendall(payload)
            except Exception:
                pass

    def _recv_all(self, conn: socket.socket) -> str:
        buf = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in chunk:
                break
        return buf.decode("utf-8").strip()

    def _process_payload(self, data: str) -> dict:
        if not data:
            return {"status": "error", "message": "empty payload"}
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return {"status": "error", "message": "invalid json"}
        if not isinstance(payload, dict):
            return {"status": "error", "message": "payload must be object"}
        cmd = payload.get("cmd")
        if not cmd:
            return {"status": "error", "message": "cmd required"}
        response_q: queue.Queue = queue.Queue(maxsize=1)
        item = {"cmd": cmd, "payload": payload, "response": response_q}
        try:
            self.q.put_nowait(item)
        except queue.Full:
            return {"status": "error", "message": "server busy"}
        try:
            return response_q.get(timeout=self.response_timeout)
        except queue.Empty:
            return {"status": "error", "message": "command timeout"}


class UDPReceiverSingle:
    def __init__(self, port: int, host: str = "0.0.0.0", max_bytes: int = 65507):
        self.host = host
        self.port = int(port)
        self.max_bytes = max_bytes
        self.sock = None
        self.th = None
        self.running = False
        self.q = queue.Queue(maxsize=4096)

    def _log_packet(self, cam: str, dets: List[dict], meta: Optional[dict] = None):
        """
        수신 데이터 확인용 로그. meta는 JSON payload 전체(dict)일 수도 있고 None일 수도 있다.
        """
        try:
            cnt = len(dets) if dets else 0
            ts = meta.get("timestamp") if isinstance(meta, dict) else None
            capture_ts = meta.get("capture_ts") if isinstance(meta, dict) else None
            camera_id = meta.get("camera_id") if isinstance(meta, dict) else None
            print(
                "[UDPReceiverSingle] recv",
                f"cam={cam}",
                f"camera_id={camera_id}",
                f"timestamp={ts}",
                f"capture_ts={capture_ts}",
                f"detections={cnt}",
            )
            if dets:
                sample = json.dumps(dets[0], ensure_ascii=False)
                print(f"  sample_det={sample}")
        except Exception as exc:
            print(f"[UDPReceiverSingle] log error: {exc}")

    def start(self):
        if self.running:
            return
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.th = threading.Thread(target=self._rx_loop, daemon=True)
        self.th.start()
        print(f"[UDPReceiverSingle] listening on {self.host}:{self.port}")

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        if self.th:
            self.th.join(timeout=0.5)

    def _rx_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.max_bytes)
                ts = time.time()
                cam, dets = self._parse_payload(data)
                if dets is None:
                    continue
                self.q.put_nowait({"cam": cam, "ts": ts, "dets": dets})
            except Exception:
                continue

    def _parse_payload(self, data: bytes):
        try:
            msg = json.loads(data.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == "bev_labels":
                cam_id = int(msg.get("camera_id", 0) or 0)
                cam = f"cam{cam_id}" if cam_id else "cam?"
                dets = []
                for it in msg.get("items", []):
                    cx, cy = it["center"]
                    color = normalize_color_label(it.get("color"))
                    dets.append({
                        "cls": int(it.get("class", 0)),
                        "cx": float(cx),
                        "cy": float(cy),
                        "length": float(it.get("length", 0.0)),
                        "width": float(it.get("width", 0.0)),
                        "yaw": float(it.get("yaw", 0.0)),
                        "score": float(it.get("score", 0.0)),
                        "cz": float(it.get("cz", 0.0)),
                        "pitch": float(it.get("pitch", 0.0)),
                        "roll": float(it.get("roll", 0.0)),
                        "color": color,
                        "color_hex": it.get("color_hex"),
                    })
                self._log_packet(cam, dets if dets else [], meta=msg)
                return cam, dets if dets else []
        except Exception:
            pass


# -----------------------------
# 파이프라인: 수집 → 통합/융합 → 추적 → 브로드캐스트(JSON)
# -----------------------------
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
        tx_host: Optional[str] = None, tx_port: int = 60050, tx_protocol: str = "udp",
        carla_host: Optional[str] = None, carla_port: int = 61000,
        global_ply: str = "real_global_ply.ply",
        vehicle_glb: str = "pointcloud/car.glb",  # 더 이상 사용하지 않지만 인터페이스 유지
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
        
        # 이건
        self._log_interval = 1.0
        self._next_log_ts = 0.0
        # WebSocket 브로드캐스터 (React UI로 차량 정보 전송)
        self.ws_hub: Optional[WebSocketHub] = None
        if ws_host:
            try:
                self.ws_hub = WebSocketHub(ws_host, ws_port, path="/monitor")
                self.ws_hub.start()
            except Exception as exc:
                print(f"[WebSocketHub] init failed: {exc}")
                self.ws_hub = None

        # 단일 소켓 리시버 (엣지→서버 UDP)
        self.receiver = UDPReceiverSingle(single_port)

        # 카메라 위치(가중치/거리 계산용)
        self.camera_markers = load_camera_markers(cam_positions_path, local_ply_dir, local_lut_dir)
        self.cam_xy: Dict[str, Tuple[float, float]] = {}
        for marker in self.camera_markers:
            pos = marker.get("position") or {}
            x = _safe_float(pos.get("x"))
            y = _safe_float(pos.get("y"))
            if x is None or y is None:
                continue
            name = marker.get("name") or marker.get("key")
            if not name:
                continue
            self.cam_xy[str(name)] = (float(x), float(y))
        if not self.cam_xy:
            print("[Fusion] WARN: no camera positions loaded; distance weighting falls back to origin.")

        # 프레임 버퍼(최근 T초 동안 카메라별 최신)
        self.buffer: Dict[str, deque] = {cam: deque(maxlen=1) for cam in cam_ports.keys()}

        # 지면 높이 조회
        self.ground_height = GroundHeightLookup(global_ply, flip_y=flip_ply_y)

        # 브로드캐스트 (UDP/TCP)
        self.track_tx = TrackBroadcaster(tx_host, tx_port, tx_protocol) if tx_host else None
        self.carla_tx = TrackBroadcaster(carla_host, carla_port) if carla_host else None

        # 추적기
        smooth_window = max(3, min(8, int(round(0.2 / max(self.dt, 1e-3)))))
        self.tracker = SortTracker(
            max_age=20,
            min_hits=20,
            iou_threshold=0.15,
            smooth_window=smooth_window,
        )
        self._log_interval = 1.0
        self._next_log_ts = 0.0

        # 명령 서버 (yaw/색상 수정 등)
        self.command_queue: Optional[queue.Queue] = None
        self.command_server: Optional[CommandServer] = None
        if command_host and command_port:
            self.command_queue = queue.Queue()
            self.command_server = CommandServer(command_host, command_port, self.command_queue)

        self.tracker_fixed_length = float(tracker_fixed_length) if tracker_fixed_length is not None else None
        self.tracker_fixed_width = float(tracker_fixed_width) if tracker_fixed_width is not None else None

    def _world_to_map_xy(self, cx: float, cy: float) -> Tuple[float, float]:
        """
        세계 좌표(cx, cy)를 UI에서 쓰는 0~1 비율 좌표로 변환.
        아래 범위는 맵에 맞게 나중에 조정하면 된다.
        """
        # TODO: 실제 맵 스케일에 맞게 조정
        x_min, x_max = -50.0, 50.0
        y_min, y_max = -50.0, 50.0

        def _norm(v, vmin, vmax):
            if vmax <= vmin:
                return 0.5
            t = (v - vmin) / (vmax - vmin)
            return float(max(0.0, min(1.0, t)))

        x_norm = _norm(cx, x_min, x_max)
        y_norm = _norm(cy, y_min, y_max)

        return x_norm, y_norm

    def _build_ui_snapshot(self, tracks: np.ndarray, ts: float) -> Optional[dict]:
        """
        현재 트랙 상태를 React 프론트의 ServerSnapshot 형태로 변환.
        - carsOnMap: 맵 위 차량 아이콘
        - carsStatus: 오른쪽 Car Status 패널
        나머지는 일단 비워 둔다.
        """
        if tracks is None or len(tracks) == 0:
            # 차량 없을 때는 snapshot 자체를 보내지 않거나, 빈 상태로 보낼 수 있다.
            # 여기서는 빈 snapshot을 보내자.
            cars_on_map = []
            cars_status = []
        else:
            cars_on_map = []
            cars_status = []
            for row in tracks:
                tid = int(row[0])
                cls = int(row[1])
                cx = float(row[2])
                cy = float(row[3])
                length = float(row[4])
                width = float(row[5])
                yaw = float(row[6])

                meta = self.track_meta.get(tid, {})
                color_label = meta.get("color")
                color_hex = meta.get("color_hex")
                ui_color = color_hex or color_label_to_hex(color_label) or "#22c55e"

                # UI용 ID들
                car_id = f"car-{tid}"
                map_id = f"mcar-{tid}"

                # 세계 좌표 → 0~1 맵 좌표
                x_norm, y_norm = self._world_to_map_xy(cx, cy)

                cars_on_map.append({
                    "id": map_id,
                    "carId": car_id,
                    "x": x_norm,
                    "y": y_norm,
                    "yaw": yaw,
                    "color": ui_color,
                    "status": "normal",  # 경로 변경 로직은 나중에
                })

                cars_status.append({
                    "id": car_id,
                    "color": ui_color,
                    "speed": meta.get("speed", 0.0),      # 추적기에서 속도 정보가 있으면 나중에 연결
                    "battery": 100,                        # 일단 임의 값
                    "fromLabel": "-",                      # TODO: 나중에 실제 출발지
                    "toLabel": "-",                        # TODO: 나중에 실제 도착지
                    "cameraId": None,
                    "routeChanged": False,
                })

        snapshot = {
            "type": "snapshot",
            "payload": {
                "carsOnMap": cars_on_map,
                "carsStatus": cars_status,
                "camerasOnMap": [],
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

    def _ground_height_at(self, x: float, y: float, default: float = 0.0) -> float:
        if hasattr(self, "ground_height") and self.ground_height and self.ground_height.enabled:
            return self.ground_height.query(x, y, default=default)
        return float(default)

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
                "class": int(row[1]),
                "cx": float(row[2]),
                "cy": float(row[3]),
                "length": float(row[4]),
                "width": float(row[5]),
                "yaw": float(row[6]),
            }
            print(f"  track_sample={json.dumps(sample_track, ensure_ascii=False)}")
        if timings:
            timing_str = " ".join(f"{k}={v:.2f}ms" for k, v in timings.items())
            print(f"  timings {timing_str}")

    def _send_command(self, payload: dict, timeout: float = 2.0) -> dict:
        if not self.command_queue:
            raise RuntimeError("command queue not initialized")
        cmd = payload.get("cmd")
        if not cmd:
            raise RuntimeError("cmd required")
        resp_q: queue.Queue = queue.Queue(maxsize=1)
        item = {"cmd": cmd, "payload": payload, "response": resp_q}
        try:
            self.command_queue.put_nowait(item)
        except queue.Full:
            raise RuntimeError("command queue busy")
        try:
            return resp_q.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError("command timeout") from exc

    def start(self):
        self.receiver.start()
        if self.command_server:
            self.command_server.start()
        self._main_loop()

    def _main_loop(self):
        last = time.time()
        while True:
            timings: Dict[str, float] = {}
            self._process_command_queue()
            try:
                while True:
                    item = self.receiver.q.get_nowait()
                    cam = item["cam"]
                    ts = item["ts"]
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

            # ---- ① 카메라별 최신 → ② 융합 ----
            t0 = time.perf_counter()
            raw_dets = self._gather_current()
            timings["gather"] = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            fused = self._fuse_boxes(raw_dets)
            timings["fuse"] = (time.perf_counter() - t1) * 1000.0

            # ---- ③ 추적(SORT) ----
            det_rows = []
            det_colors: List[Optional[str]] = []
            for det in fused:
                det_rows.append([
                    0,
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
        boxes = np.array([[d["cx"], d["cy"], d["length"], d["width"], d["yaw"]] for d in raw_detections], dtype=float)
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
        cams = [d.get("cam", "?") for d in subset]
        normalized_colors = [normalize_color_label(d.get("color")) for d in subset]
        valid_colors = [c for c in normalized_colors if c is not None]
        color_counts = Counter(valid_colors)
        color = color_counts.most_common(1)[0][0] if color_counts else None
        hex_candidates = [normalize_color_hex(d.get("color_hex")) for d in subset if d.get("color_hex")]
        hex_candidates = [h for h in hex_candidates if h]
        color_hex = random.choice(hex_candidates) if hex_candidates else None
        if fused_box is not None and len(fused_box) >= 4:
            cx_rep = float(fused_box[0])
            cy_rep = float(fused_box[1])
            cz_default = np.mean([float(d.get("cz", 0.0)) for d in subset])
            cz = self._ground_height_at(cx_rep, cy_rep, default=cz_default)
        else:
            cz = np.mean([float(d.get("cz", 0.0)) for d in subset])
        return {
            "cz": float(cz),
            "pitch": float(pitch),
            "roll": float(roll),
            "score": float(score),
            "source_cams": cams,
            "color": color,
            "color_hex": color_hex,
            "color_votes": dict(color_counts),
        }

    def _update_track_meta(self, matches: List[Tuple[int, int]], fused_list: List[dict], track_attrs: Dict[int, dict]):
        active_ids = set(track_attrs.keys())
        self.track_meta = {tid: meta for tid, meta in self.track_meta.items() if tid in active_ids}
        for tid, attrs in track_attrs.items():
            meta = self.track_meta.setdefault(tid, {})
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
                "cz": float(det.get("cz", 0.0)),
                "pitch": float(det.get("pitch", 0.0)),
                "roll": float(det.get("roll", 0.0)),
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
        # ✅ React UI용 WebSocket 브로드캐스트
        if self.ws_hub:
            snapshot = self._build_ui_snapshot(tracks, ts)
            if snapshot is not None:
                self.ws_hub.broadcast(snapshot)


def parse_cam_ports(text: str) -> Dict[str, int]:
    """
    예: "cam1:50050,cam2:50051"
    """
    out = {}
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, port = tok.split(":")
        out[name.strip()] = int(port)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-ports", default="cam1:50050,cam2:50051")
    ap.add_argument("--cam-positions-json", default="camera_position.json")
    ap.add_argument("--local-ply-dir", default="outputs",
                    help="카메라별 로컬 PLY를 탐색할 디렉토리(패턴: cam_<id>_*.ply)")
    ap.add_argument("--local-lut-dir", default="outputs",
                    help="카메라별 LUT(npz)를 탐색할 디렉토리(패턴: cam_<id>_*.npz)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--iou-thr", type=float, default=0.01)
    ap.add_argument("--roll-secs", type=int, default=60)
    ap.add_argument("--roll-max-rows", type=int, default=1000)
    ap.add_argument("--udp-port", type=int, default=50050)
    ap.add_argument("--tx-host", default=None)
    ap.add_argument("--tx-port", type=int, default=60050)
    ap.add_argument("--tx-protocol", choices=["udp", "tcp"], default="udp")
    ap.add_argument("--carla-host", default=None)
    ap.add_argument("--carla-port", type=int, default=61000)
    ap.add_argument("--global-ply", default="pointcloud/real_coshow_map_1127.ply")
    ap.add_argument("--vehicle-glb", default="pointcloud/car.glb")
    # 아래 web/viewer 관련 옵션들은 더 이상 사용하지 않지만, CLI 호환을 위해 남겨둠
    ap.add_argument("--web-host", default="0.0.0.0")
    ap.add_argument("--web-port", type=int, default=18000)
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--overlay-base-url", type=str, default=None)
    ap.add_argument("--tracker-fixed-length", type=float, default=None)
    ap.add_argument("--tracker-fixed-width", type=float, default=None)
    ap.add_argument("--size-mode", choices=["bbox", "fixed", "mesh"], default="mesh")
    ap.add_argument("--fixed-length", type=float, default=4.5)
    ap.add_argument("--fixed-width", type=float, default=1.8)
    ap.add_argument("--height-scale", type=float, default=0.5,
                    help="bbox/fixed 모드일 때 차량 높이 = width * height_scale")
    ap.add_argument("--mesh-scale", type=float, default=1.0,
                    help="size-mode=mesh 일 때 GLB에 곱할 유니폼 스케일")
    ap.add_argument("--mesh-height", type=float, default=0.0,
                    help="size-mode=mesh 일 때 지면 높이 계산용 높이(0이면 mesh-scale 사용)")
    ap.add_argument("--z-offset", type=float, default=0.0,
                    help="모든 박스에 추가할 z 오프셋")
    ap.add_argument("--invert-bev-y", dest="invert_bev_y", action="store_true")
    ap.add_argument("--no-invert-bev-y", dest="invert_bev_y", action="store_false")
    ap.set_defaults(invert_bev_y=True)
    ap.add_argument("--normalize-vehicle", dest="normalize_vehicle", action="store_true")
    ap.add_argument("--no-normalize-vehicle", dest="normalize_vehicle", action="store_false")
    ap.set_defaults(normalize_vehicle=True)
    ap.add_argument("--vehicle-y-up", dest="vehicle_y_up", action="store_true")
    ap.add_argument("--vehicle-z-up", dest="vehicle_y_up", action="store_false")
    ap.set_defaults(vehicle_y_up=True)
    ap.add_argument("--flip-ply-y", dest="flip_ply_y", action="store_true",
                    help="global ply의 Y축을 반전하여 로드")
    ap.add_argument("--no-flip-ply-y", dest="flip_ply_y", action="store_false",
                    help="global ply Y축 반전하지 않음")
    ap.set_defaults(flip_ply_y=True)
    ap.add_argument("--flip-marker-x", dest="flip_marker_x", action="store_true")
    ap.add_argument("--no-flip-marker-x", dest="flip_marker_x", action="store_false")
    ap.set_defaults(flip_marker_x=False)
    ap.add_argument("--flip-marker-y", dest="flip_marker_y", action="store_true")
    ap.add_argument("--no-flip-marker-y", dest="flip_marker_y", action="store_false")
    ap.set_defaults(flip_marker_y=True)
    ap.add_argument("--cmd-host", default="0.0.0.0",
                    help="yaw 명령 서버 바인드 호스트 (미지정 시 비활성화)")
    ap.add_argument("--cmd-port", type=int, default=18100,
                    help="yaw 명령 서버 포트")

    args = ap.parse_args()

    cam_ports = parse_cam_ports(args.cam_ports)

    server = RealtimeFusionServer(
        cam_ports=cam_ports,
        cam_positions_path=args.cam_positions_json,
        local_ply_dir=args.local_ply_dir,
        local_lut_dir=args.local_lut_dir,
        fps=args.fps,
        iou_cluster_thr=args.iou_thr,
        single_port=args.udp_port,
        tx_host=args.tx_host,
        tx_port=args.tx_port,
        tx_protocol=args.tx_protocol,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        global_ply=args.global_ply,
        vehicle_glb=args.vehicle_glb,
        tracker_fixed_length=args.tracker_fixed_length,
        tracker_fixed_width=args.tracker_fixed_width,
        command_host=args.cmd_host,
        command_port=args.cmd_port,
        flip_ply_y=args.flip_ply_y,
        ws_host=None if args.no_web else args.web_host,
        ws_port=args.web_port,
    )
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.receiver.stop()
        if server.track_tx:
            server.track_tx.close()
        if server.carla_tx:
            server.carla_tx.close()


if __name__ == "__main__":
    main()

'''
python server.py \
  --udp-port 50050 \
  --tx-host 127.0.0.1 --tx-port 60050
'''
