import json
import socket
from typing import Dict, Optional

import numpy as np


class TrackBroadcaster:
    """
    인지서버 -> carla/제어서버
    매 프레임마다 tracks+extras 데이터를 json으로 해서 지정된 호스트/포트로 전송
    UDP/TCP 브로드캐스터.
    track 패킷 포맷:
    {
        "type": "global_tracks",
        "timestamp": unix_ts,
        "items": [
            {"id": tid, "class": cls, "center": [cx, cy, cz], "length": L,
             "width": W, "yaw": yaw_deg, "velocity": [vx, vy], "speed": speed,
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
            self.addr: Optional[tuple[str, int]] = (self.host, self.port)
        else:  # TCP
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.addr = None

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def send(self, tracks: np.ndarray, extras: Dict[int, dict], ts: float) -> None:
        payload = {
            "type": "global_tracks",
            "timestamp": ts,
            "items": [],
        }
        if tracks is not None and len(tracks):
            for row in tracks:
                tid = int(row[0])
                cls = int(row[1])
                cx, cy, L, W, yaw = map(float, row[2:7])
                extra = extras.get(tid, {})
                vx, vy = 0.0, 0.0
                velocity = extra.get("velocity")
                if velocity and len(velocity) >= 2:
                    try:
                        vx = float(velocity[0])
                        vy = float(velocity[1])
                    except Exception:
                        vx, vy = 0.0, 0.0
                speed_val = extra.get("speed")
                if speed_val is None:
                    speed_val = float(np.hypot(vx, vy))
                payload["items"].append({
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
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            if self.protocol == "udp":
                if self.addr:
                    self.sock.sendto(data, self.addr)
            else:
                self.sock.sendall(data + b"\n")
        except Exception as exc:
            print(f"[TrackBroadcaster] send error: {exc}")
