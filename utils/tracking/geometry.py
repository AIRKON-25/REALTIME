import math
from typing import List
import numpy as np

def wrap_deg(angle: float) -> float:
    """각도를 -180도에서 180도 범위로 래핑"""
    a = (angle + 180.0) % 360.0
    if a < 0:
        a += 360.0
    return a - 180.0


def nearest_equivalent_deg(meas: float, ref: float, period: float = 360.0) -> float:
    """
    근처에 있는 동등한 각도를 반환
    예: meas=170, ref=-170 -> 반환값은 -190 (ref에 더 가까움)
    180도 주기 각도에 유용함
    """
    d = meas - ref
    d = (d + period / 2.0) % period - period / 2.0
    return ref + d


def carla_to_aabb(detection: np.ndarray) -> np.ndarray:
    """
    CARLA 형식의 검출을 축 정렬 경계 상자(AABB)로 변환
    """
    # detection: [class, x_c, y_c, l, w, yaw_deg]
    x_c, y_c, l, w, yaw_deg = detection[1:6]
    yaw = math.radians(yaw_deg)

    dx, dy = l / 2.0, w / 2.0
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rotated_corners = corners @ R.T + np.array([x_c, y_c])

    x_min = np.min(rotated_corners[:, 0])
    x_max = np.max(rotated_corners[:, 0])
    y_min = np.min(rotated_corners[:, 1])
    y_max = np.max(rotated_corners[:, 1])

    aabb_width = x_max - x_min
    aabb_height = y_max - y_min

    return np.array([x_min, y_min, aabb_width, aabb_height])


def iou_bbox(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    두 축 정렬 경계 상자 간의 IoU 계산
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2]) * max(0.0, boxA[3])
    areaB = max(0.0, boxB[2]) * max(0.0, boxB[3])

    denom = areaA + areaB - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def iou_batch(detections_carla: np.ndarray, tracks: List["Track"]) -> np.ndarray:
    """
    검출과 예측된 트랙 간의 IoU 기반 비용 행렬 계산
    1 - IoU 값을 사용하여 비용을 나타냄
    """
    cost_matrix = np.zeros((len(detections_carla), len(tracks)), dtype=np.float32)
    for i, det_carla in enumerate(detections_carla):
        det_aabb = carla_to_aabb(det_carla)
        for j, track in enumerate(tracks):
            pred_xc, pred_yc = track.kf_pos.x[:2].flatten()
            temp_obb = np.array([0, pred_xc, pred_yc, track.car_length, track.car_width, track.car_yaw])
            pred_aabb = carla_to_aabb(temp_obb)
            cost_matrix[i, j] = 1.0 - iou_bbox(det_aabb, pred_aabb)
    return cost_matrix


def aabb_iou_axis_aligned(b1, b2) -> float:
    """
    두 축 정렬 경계 상자 간의 IoU 계산
    """
    g1 = b1[-5:] if len(b1) >= 5 else b1
    g2 = b2[-5:] if len(b2) >= 5 else b2
    x1, y1, L1, W1, _ = g1
    x2, y2, L2, W2, _ = g2
    A = [x1 - L1 / 2, y1 - W1 / 2, x1 + L1 / 2, y1 + W1 / 2]
    B = [x2 - L2 / 2, y2 - W2 / 2, x2 + L2 / 2, y2 + W2 / 2]
    ix1, iy1 = max(A[0], B[0]), max(A[1], B[1])
    ix2, iy2 = min(A[2], B[2]), min(A[3], B[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (L1 * W1) + (L2 * W2) - inter
    return inter / union if union > 1e-12 else 0.0
