import json
import math
from typing import Dict, List, Optional, Tuple


class LaneMatcher:
    """
    lanes.json을 로드해 차량 위치(cx, cy)에 가장 가까운 차선과 yaw를 찾아준다.
    lanes.json 포맷 예시:
    [
      { "id": "lane-1", "polyline": [[x,y], ...], "assign_gate_m": 2.0 },
      ...
    ]
    """

    def __init__(self, lanes: List[dict], default_gate: float = 2.0):
        self._segments: List[Dict[str, object]] = []
        self.default_gate = default_gate
        for lane in lanes:
            lid = str(lane.get("id", "lane"))
            poly = lane.get("polyline") or []
            if len(poly) < 2:
                continue
            gate = float(lane.get("assign_gate_m", default_gate))
            for i in range(len(poly) - 1):
                x0, y0 = poly[i][0], poly[i][1]
                x1, y1 = poly[i + 1][0], poly[i + 1][1]
                dx, dy = x1 - x0, y1 - y0
                seg_len2 = dx * dx + dy * dy
                if seg_len2 <= 0.0:
                    continue
                yaw = math.degrees(math.atan2(dy, dx))
                self._segments.append({
                    "id": lid,
                    "p0": (x0, y0),
                    "p1": (x1, y1),
                    "dx": dx,
                    "dy": dy,
                    "len2": seg_len2,
                    "yaw": yaw,
                    "gate": gate,
                })

    @classmethod
    def from_json(cls, path: str, default_gate: float = 2.0) -> "LaneMatcher":
        with open(path, "r", encoding="utf-8") as f:
            lanes = json.load(f)
        return cls(lanes, default_gate=default_gate)

    def match(self, cx: float, cy: float) -> Tuple[Optional[str], Optional[float], float]:
        """
        주어진 위치에서 가장 가까운 차선과 yaw를 반환.
        반환: (lane_id, yaw_deg, dist_m). assign_gate를 넘으면 (None, None, inf).
        """
        best = (None, None, float("inf"))
        for seg in self._segments:
            px, py = seg["p0"]
            dx, dy = seg["dx"], seg["dy"]
            len2 = seg["len2"]
            t = ((cx - px) * dx + (cy - py) * dy) / len2
            t = max(0.0, min(1.0, t))
            proj_x = px + t * dx
            proj_y = py + t * dy
            dist = math.hypot(cx - proj_x, cy - proj_y)
            if dist < best[2]:
                best = (seg["id"], seg["yaw"], dist)
        lane_id, yaw, dist = best
        if dist == float("inf"):
            return None, None, dist
        # gate는 매칭된 세그먼트의 gate 사용
        gate = next((seg["gate"] for seg in self._segments if seg["id"] == lane_id and math.isclose(seg["yaw"], yaw)), self.default_gate)
        if dist > gate:
            return None, None, dist
        return lane_id, yaw, dist
