import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from utils.colors import normalize_color_label
from utils.tracking.geometry import aabb_iou_axis_aligned, nearest_equivalent_deg, wrap_deg
from utils.tracking._constants import (
    COLOR_BONUS,
    COLOR_PENALTY,
    IOU_CLUSTER_THR,
    IOU_FALLBACK_CENTER_RATIO,
    IOU_FALLBACK_SIZE_RATIO_MAX,
    YAW_PERIOD_DEG,
)


class ClassMergePolicy(Enum):
    STRICT = "strict"
    ALLOW_CAR_OBSTACLE = "allow_car_obstacle"


@dataclass
class ClusterConfig:
    iou_cluster_thr: float = IOU_CLUSTER_THR
    color_bonus: float = COLOR_BONUS
    color_penalty: float = COLOR_PENALTY
    center_gate_car: float = 1.0
    center_gate_obstacle: float = 1.3
    iou_gate: Optional[float] = None
    iou_fallback_center_ratio: float = IOU_FALLBACK_CENTER_RATIO
    iou_fallback_size_ratio_max: float = IOU_FALLBACK_SIZE_RATIO_MAX
    aspect_ratio_gate: float = 1.8
    area_ratio_gate: float = 3.0
    yaw_diff_gate: float = 75.0
    merge_policy: ClassMergePolicy = ClassMergePolicy.STRICT
    allow_split_multi_modal: bool = False
    split_center_distance: float = 1.25
    split_min_cluster_size: int = 3
    split_balance_ratio: float = 0.25
    misclass_min_iou: float = 0.65
    misclass_max_shape_ratio: float = 1.5
    misclass_max_yaw_diff: float = 25.0
    misclass_track_iou_thr: float = 0.25
    misclass_require_color_match: bool = False
    weight_score_scale: float = 0.5
    weight_iou_scale: float = 1.0
    weight_iou_cap: float = 2.0
    weight_track_scale: float = 0.6


CAR_CLASSES = {0}
OBSTACLE_CLASSES = {1, 2}


def _normalized_center_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    dx = abs(float(b1[0]) - float(b2[0]))
    dy = abs(float(b1[1]) - float(b2[1]))
    w_avg = max((abs(float(b1[2])) + abs(float(b2[2]))) * 0.5, 1e-6)
    h_avg = max((abs(float(b1[3])) + abs(float(b2[3]))) * 0.5, 1e-6)
    return math.hypot(dx / w_avg, dy / h_avg)


def _shape_mismatch(b1: np.ndarray, b2: np.ndarray, aspect_gate: float, area_gate: float) -> bool:
    a1 = abs(float(b1[2])) / max(abs(float(b1[3])), 1e-6)
    a2 = abs(float(b2[2])) / max(abs(float(b2[3])), 1e-6)
    aspect_ratio = max(a1, a2) / max(min(a1, a2), 1e-6)
    area1 = abs(float(b1[2]) * float(b1[3]))
    area2 = abs(float(b2[2]) * float(b2[3]))
    area_ratio = max(area1, area2) / max(min(area1, area2), 1e-6)
    return aspect_ratio > aspect_gate or area_ratio > area_gate


def _size_ratio_mismatch(b1: np.ndarray, b2: np.ndarray, ratio_max: float) -> bool:
    if ratio_max <= 0.0:
        return False
    l1 = abs(float(b1[2]))
    w1 = abs(float(b1[3]))
    l2 = abs(float(b2[2]))
    w2 = abs(float(b2[3]))
    eps = 1e-6
    ratio_l = max(l1, l2) / max(min(l1, l2), eps)
    ratio_w = max(w1, w2) / max(min(w1, w2), eps)
    return ratio_l > ratio_max or ratio_w > ratio_max


def _yaw_diff_deg(yaw_a: float, yaw_b: float, period_deg: float) -> float:
    aligned = nearest_equivalent_deg(yaw_b, yaw_a, period=period_deg)
    return abs(wrap_deg(float(yaw_a) - float(aligned)))


def _best_track_matches(
    boxes_geom: np.ndarray,
    track_hints: Optional[np.ndarray],
) -> List[Tuple[Optional[int], float]]:
    if track_hints is None or len(track_hints) == 0:
        return [(None, 0.0)] * len(boxes_geom)
    matches: List[Tuple[Optional[int], float]] = []
    for geom in boxes_geom:
        best_id: Optional[int] = None
        best_score = 0.0
        for track in track_hints:
            t_box = track[-5:]
            iou = aabb_iou_axis_aligned(geom, t_box)
            if iou > best_score:
                best_score = iou
                try:
                    best_id = int(track[0])
                except Exception:
                    best_id = None
        matches.append((best_id, best_score))
    return matches


def _class_merge_allowed(
    cls_a: Optional[float],
    cls_b: Optional[float],
    b1: np.ndarray,
    b2: np.ndarray,
    iou_val: float,
    colors: Optional[List[Optional[str]]],
    idx_a: int,
    idx_b: int,
    track_matches: List[Tuple[Optional[int], float]],
    cfg: ClusterConfig,
) -> bool:
    if cls_a is None or cls_b is None or cls_a == cls_b:
        return True
    if cfg.merge_policy != ClassMergePolicy.ALLOW_CAR_OBSTACLE:
        return False
    if {cls_a, cls_b} - (CAR_CLASSES | OBSTACLE_CLASSES):
        return False
    if iou_val < cfg.misclass_min_iou:
        return False
    if _shape_mismatch(b1, b2, cfg.misclass_max_shape_ratio, cfg.misclass_max_shape_ratio):
        return False
    if abs(wrap_deg(float(b1[4]) - float(b2[4]))) > cfg.misclass_max_yaw_diff:
        return False
    if cfg.misclass_require_color_match and colors:
        c1 = colors[idx_a]
        c2 = colors[idx_b]
        if c1 and c2 and c1 != c2:
            return False
    if cfg.misclass_track_iou_thr > 0.0:
        t1 = track_matches[idx_a] if idx_a < len(track_matches) else (None, 0.0)
        t2 = track_matches[idx_b] if idx_b < len(track_matches) else (None, 0.0)
        if t1[0] is None or t2[0] is None or t1[0] != t2[0]:
            return False
        if min(t1[1], t2[1]) < cfg.misclass_track_iou_thr:
            return False
    return True


def _maybe_split_cluster(idxs: List[int], boxes_geom: np.ndarray, cfg: ClusterConfig) -> List[List[int]]:
    if not cfg.allow_split_multi_modal or len(idxs) < cfg.split_min_cluster_size:
        return [idxs]
    centers = np.array([[boxes_geom[i][0], boxes_geom[i][1]] for i in idxs], dtype=float)
    sizes = np.array([[boxes_geom[i][2], boxes_geom[i][3]] for i in idxs], dtype=float)
    w_scale = max(float(np.mean(sizes[:, 0])), 1e-3)
    h_scale = max(float(np.mean(sizes[:, 1])), 1e-3)
    norm_centers = np.column_stack(((centers[:, 0] / w_scale), (centers[:, 1] / h_scale)))
    dist_mat = np.linalg.norm(norm_centers[:, None, :] - norm_centers[None, :, :], axis=2)
    seed_a, seed_b = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
    if seed_a == seed_b:
        return [idxs]
    centroids = [norm_centers[seed_a], norm_centers[seed_b]]
    assignments = []
    for p in norm_centers:
        d0 = np.linalg.norm(p - centroids[0])
        d1 = np.linalg.norm(p - centroids[1])
        assignments.append(0 if d0 <= d1 else 1)
    assignments = np.array(assignments)
    split_clusters: List[List[int]] = []
    for label in (0, 1):
        members = [idxs[i] for i, a in enumerate(assignments) if a == label]
        if members:
            split_clusters.append(members)
    if len(split_clusters) != 2:
        return [idxs]
    c0 = norm_centers[assignments == 0].mean(axis=0)
    c1 = norm_centers[assignments == 1].mean(axis=0)
    separation = float(np.linalg.norm(c0 - c1))
    balance = min(len(split_clusters[0]), len(split_clusters[1])) / float(len(idxs))
    if separation < cfg.split_center_distance or balance < cfg.split_balance_ratio:
        return [idxs]
    return split_clusters


def cluster_by_aabb_iou(
    boxes: np.ndarray,
    config: Optional[ClusterConfig] = None,
    iou_cluster_thr: float = IOU_CLUSTER_THR,
    color_labels: Optional[List[Optional[str]]] = None,
    color_bonus: float = COLOR_BONUS,
    color_penalty: float = COLOR_PENALTY,
    track_hints: Optional[np.ndarray] = None,
    return_debug: bool = False,
) -> Union[List[List[int]], Tuple[List[List[int]], Dict[str, object]]]:
    """
    Axis-aligned bounding box 기반 IoU 클러스터링 with configurable gates and optional debug.
    :param boxes: (N, >=5) 형태의 numpy 배열. 최소 5개의 값 (x, y, l, w, yaw) 형식을 가정.
    :param config: ClusterConfig 설정. None이면 iou_cluster_thr 기반 기본 설정.
    :param iou_cluster_thr: 예전 호환성을 위한 IoU 기본값.
    :param color_labels: (N,) 길이의 리스트로, 각 박스의 색상 후보 (정규화 전).
    :param color_bonus: 색상이 같을 때 IoU 임계치에 적용할 보너스(낮춤).
    :param color_penalty: 색상이 다를 때 IoU 임계치에 적용할 패널티(올림).
    :param track_hints: 이전 프레임 트랙 상태 [[id, cls, cx, cy, l, w, yaw], ...].
    :param return_debug: True면 디버그 카운터/트랙 매칭 정보를 함께 반환.
    :return: 클러스터 인덱스 리스트 또는 (클러스터, 디버그 정보) 튜플.
    """
    if boxes.size == 0:
        return [] if not return_debug else ([], {"counters": {}, "track_matches": []})
    cfg = config or ClusterConfig(
        iou_cluster_thr=iou_cluster_thr,
        color_bonus=color_bonus,
        color_penalty=color_penalty,
    )
    if cfg.iou_gate is None:
        cfg.iou_gate = cfg.iou_cluster_thr
    boxes_geom = boxes[:, -5:] if isinstance(boxes, np.ndarray) and boxes.ndim == 2 and boxes.shape[1] >= 5 else boxes
    cls_col = boxes[:, 0] if isinstance(boxes, np.ndarray) and boxes.ndim == 2 and boxes.shape[1] > 5 else None
    N = len(boxes_geom)
    clusters: List[List[int]] = []
    use_color = (
        color_labels is not None
        and len(color_labels) == N
        and (cfg.color_bonus > 0.0 or cfg.color_penalty > 0.0)
    )
    normalized_colors: Optional[List[Optional[str]]] = None
    if use_color:
        normalized_colors = [normalize_color_label(c) for c in color_labels]
    track_matches = _best_track_matches(boxes_geom, track_hints)
    neighbors: Dict[int, List[int]] = defaultdict(list)
    debug_counts: Counter = Counter()
    for i in range(N):
        for j in range(i + 1, N):
            b1 = boxes_geom[i]
            b2 = boxes_geom[j]
            cls_a = cls_col[i] if cls_col is not None else None
            cls_b = cls_col[j] if cls_col is not None else None
            center_thr = cfg.center_gate_obstacle
            if (cls_a in CAR_CLASSES) or (cls_b in CAR_CLASSES):
                center_thr = cfg.center_gate_car
            center_dist = _normalized_center_distance(b1, b2)
            if center_dist > center_thr:
                debug_counts["center_gate"] += 1
                continue
            thr = cfg.iou_gate
            if use_color and normalized_colors:
                c1 = normalized_colors[i]
                c2 = normalized_colors[j]
                if c1 and c2:
                    if c1 == c2:
                        thr = max(cfg.iou_gate - cfg.color_bonus, 0.0)
                    else:
                        thr = min(cfg.iou_gate + cfg.color_penalty, 1.0)
            iou_val = aabb_iou_axis_aligned(b1, b2)
            if not _class_merge_allowed(cls_a, cls_b, b1, b2, iou_val, normalized_colors, i, j, track_matches, cfg):
                debug_counts["class_policy"] += 1
                continue
            if iou_val < thr:
                fallback_center = 0.0
                if cfg.iou_fallback_center_ratio > 0.0:
                    fallback_center = center_thr * cfg.iou_fallback_center_ratio
                if (
                    cfg.iou_fallback_center_ratio > 0.0
                    and center_dist <= fallback_center
                    and not _size_ratio_mismatch(b1, b2, cfg.iou_fallback_size_ratio_max)
                ):
                    debug_counts["iou_fallback"] += 1
                else:
                    debug_counts["iou_gate"] += 1
                    continue
            if _shape_mismatch(b1, b2, cfg.aspect_ratio_gate, cfg.area_ratio_gate):
                debug_counts["shape_gate"] += 1
                continue
            is_obstacle_pair = (cls_a in OBSTACLE_CLASSES) and (cls_b in OBSTACLE_CLASSES)
            if not is_obstacle_pair:
                yaw_period = 360.0
                if (cls_a in CAR_CLASSES) or (cls_b in CAR_CLASSES):
                    yaw_period = YAW_PERIOD_DEG
                if _yaw_diff_deg(float(b1[4]), float(b2[4]), yaw_period) > cfg.yaw_diff_gate:
                    debug_counts["yaw_gate"] += 1
                    continue
            neighbors[i].append(j)
            neighbors[j].append(i)
    used = np.zeros(N, dtype=bool)
    for i in range(N):
        if used[i]:
            continue
        q = [i]
        used[i] = True
        cluster = [i]
        while q:
            k = q.pop()
            for j in neighbors.get(k, []):
                if used[j]:
                    continue
                used[j] = True
                q.append(j)
                cluster.append(j)
        split = _maybe_split_cluster(cluster, boxes_geom, cfg)
        if len(split) > 1:
            debug_counts["split_applied"] += 1
        clusters.extend(split)
    if not return_debug:
        return clusters
    debug_info: Dict[str, object] = {
        "counters": dict(debug_counts),
        "track_matches": track_matches,
    }
    return clusters, debug_info
