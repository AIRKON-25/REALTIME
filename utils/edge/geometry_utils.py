import numpy as np


def parallelogram_from_triangle(p0, p1, p2):
    """Restores four vertices of a parallelogram using three triangle points."""
    p3 = 2 * p0 - p1
    p4 = 2 * p0 - p2
    return np.stack([p1, p2, p3, p4], axis=0)


def aabb_of_poly4(poly4):
    """Returns the axis-aligned bbox (x0, y0, w, h) for a 4-point polygon."""
    xs = poly4[:, 0]
    ys = poly4[:, 1]
    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max(), ys.max()
    return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)


def iou_aabb_xywh(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-9
    return inter / union

def polygon_area(poly4: np.ndarray) -> float:
    a = poly4[1] - poly4[0]
    b = poly4[3] - poly4[0]
    return float(abs(np.cross(a, b)))

def tiny_filter_on_dets(dets_img, min_area=20.0, min_edge=3.0):
    """
    아주 작은 평행사변형(너무 작은 면적/짧은 변)을 제거.
    dets_img: decode_predictions가 반환한 단일 이미지의 detection 리스트.
    각 item은 최소한 'tri' (3,2) 좌표를 포함한다고 가정. 없으면 필터 스킵.
    """
    filtered = []
    for d in dets_img:
        tri = None
        if isinstance(d, dict):
            if 'tri' in d:
                tri = np.asarray(d['tri'], dtype=np.float32)
            elif 'points' in d:
                tri = np.asarray(d['points'], dtype=np.float32)
        if tri is None or tri.shape != (3, 2):
            filtered.append(d)  # 좌표 없으면 필터 불가 → 그대로 통과
            continue
        poly4 = parallelogram_from_triangle(tri[0], tri[1], tri[2])  # (4,2)
        area = polygon_area(poly4)
        edges = np.linalg.norm(np.roll(poly4, -1, axis=0) - poly4, axis=1)
        if area >= min_area and edges.min() >= min_edge:
            filtered.append(d)
    return filtered
