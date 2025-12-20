import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.tracking.geometry import wrap_deg


def distance_weight(d: float, d0: float = 5.0, p: float = 2.0) -> float:
    """
    거리 가중치: w = (d0 / (d0 + d)) ** p  (0<w<=1)
    - d0: 기준 거리(작을수록 가까운 것에 강조), p: 감쇠율 지수
    """
    return float((d0 / (d0 + max(0.0, d))) ** p)


def _normalize_with_cap(ws: np.ndarray, max_frac: float = 0.7) -> np.ndarray:
    w = ws.astype(float).copy()
    s = w.sum()
    if s <= 0:
        w[:] = 1.0 / len(w)
        return w
    w /= s
    if w.max() <= max_frac:
        return w
    over = w > max_frac
    excess = (w[over] - max_frac).sum()
    w[over] = max_frac
    remain = (~over)
    if remain.any():
        w[remain] += excess * (w[remain] / w[remain].sum())
    w = np.clip(w, 1e-6, 1.0)
    return w / w.sum()


def _cauchy_weights(resid: np.ndarray, scale: float, c: float = 2.385) -> np.ndarray:
    s = max(scale, 1e-6)
    return 1.0 / (1.0 + (resid / s) ** 2)


def _weighted_central_xy(boxes: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    cx = float(np.average(boxes[:, 0], weights=w))
    cy = float(np.average(boxes[:, 1], weights=w))
    return cx, cy


def _weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(v, weights=w))


def _weighted_circular_mean_deg(angles_deg: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    ang = np.deg2rad(angles_deg)
    S = float(np.average(np.sin(ang), weights=w))
    C = float(np.average(np.cos(ang), weights=w))
    mean = math.degrees(math.atan2(S, C))
    R = math.hypot(S, C)  # 0~1
    return mean, R


def _weighted_circular_median_deg(angles_deg: np.ndarray, w: np.ndarray) -> float:
    a = np.mod(angles_deg + 360.0, 360.0)
    idx = np.argsort(a)
    a_sorted = a[idx]
    w_sorted = w[idx] / max(w.sum(), 1e-9)
    csum = np.cumsum(w_sorted)
    j = np.searchsorted(csum, 0.5)
    med = a_sorted[min(j, len(a_sorted) - 1)]
    return wrap_deg(med)


def _align_angles_to_reference(angles_deg: np.ndarray, ref_deg: float, flip_threshold: float = 90.0) -> np.ndarray:
    out = []
    for a in angles_deg:
        d = wrap_deg(a - ref_deg)
        if abs(d) > flip_threshold:
            out.append(wrap_deg(a + 180.0))
        else:
            out.append(wrap_deg(a))
    return np.array(out, dtype=float)


def fuse_cluster_weighted(
    boxes: np.ndarray,
    cams: List[str],
    idxs: List[int],
    cam_ground_xy: Dict[str, Tuple[float, float]],
    d0: float = 5.0,
    p: float = 2.0,
    extra_weights: Optional[List[float]] = None,
    w_floor: float = 1e-4,
    max_cam_frac: float = 0.65,
    yaw_flip_thr: float = 90.0,
    yaw_R_fallback: float = 0.25,
    loc_scale_k: float = 1.4826,
    size_scale_rel: float = 0.20,
) -> np.ndarray:
    cluster_boxes = np.array([boxes[i] for i in idxs], dtype=float)
    cluster_geom = cluster_boxes[:, -5:] if cluster_boxes.ndim == 2 and cluster_boxes.shape[1] >= 5 else cluster_boxes
    cluster_cams = [cams[i] for i in idxs]
    M = len(cluster_geom)
    if M == 1:
        return cluster_geom[0].astype(float)

    weight_bias = np.ones(M, dtype=float)
    if extra_weights is not None:
        try:
            arr = np.array(extra_weights, dtype=float)
            if arr.size == M:
                arr = np.clip(arr, 0.0, None)
                if np.all(arr <= 0):
                    arr = np.ones(M, dtype=float)
                weight_bias = arr
        except Exception:
            pass

    w_dist = []
    for idx, (b, cam) in enumerate(zip(cluster_geom, cluster_cams)):
        cx, cy = b[0], b[1]
        cam_xy = cam_ground_xy.get(cam, (0.0, 0.0))
        d = math.hypot(cx - cam_xy[0], cy - cam_xy[1])
        bias = weight_bias[idx] if idx < len(weight_bias) else 1.0
        w = distance_weight(d, d0=d0, p=p) * max(bias, 1e-6)
        w_dist.append(max(w, w_floor))
    w_dist = np.array(w_dist, dtype=float)
    w_base = _normalize_with_cap(w_dist, max_frac=max_cam_frac)

    cx0, cy0 = _weighted_central_xy(cluster_geom, w_base)
    r_loc = np.hypot(cluster_geom[:, 0] - cx0, cluster_geom[:, 1] - cy0)
    med_r = float(np.average(np.sort(r_loc), weights=np.sort(w_base))) if M > 1 else float(r_loc.mean())
    scale_loc = max(med_r * loc_scale_k, 1e-3)
    w_loc = _cauchy_weights(r_loc, scale_loc)

    Ls = cluster_geom[:, 2]
    Ws = cluster_geom[:, 3]
    L0 = _weighted_mean(Ls, w_base)
    W0 = _weighted_mean(Ws, w_base)
    rel_L = np.abs(Ls - L0) / max(abs(L0), 1e-6)
    rel_W = np.abs(Ws - W0) / max(abs(W0), 1e-6)
    r_size = 0.5 * (rel_L + rel_W)
    scale_size = max(size_scale_rel, 1e-3)
    w_size = _cauchy_weights(r_size, scale_size)

    w_comb = _normalize_with_cap(w_base * w_loc * w_size, max_frac=max_cam_frac)

    yaws = cluster_geom[:, 4].astype(float)
    ref_yaw, _ = _weighted_circular_mean_deg(yaws, w_comb)
    yaws_aligned = _align_angles_to_reference(yaws, ref_yaw, flip_threshold=yaw_flip_thr)
    mean_yaw, R = _weighted_circular_mean_deg(yaws_aligned, w_comb)
    if R < yaw_R_fallback:
        mean_yaw = _weighted_circular_median_deg(yaws_aligned, w_comb)

    mean_cx = float(np.average(cluster_geom[:, 0], weights=w_comb))
    mean_cy = float(np.average(cluster_geom[:, 1], weights=w_comb))
    mean_L = float(np.average(Ls, weights=w_comb))
    mean_W = float(np.average(Ws, weights=w_comb))

    return np.array([mean_cx, mean_cy, mean_L, mean_W, wrap_deg(mean_yaw)], dtype=float)
