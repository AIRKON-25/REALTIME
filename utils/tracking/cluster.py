from typing import List, Optional

import numpy as np

from utils.colors import normalize_color_label
from utils.tracking.geometry import aabb_iou_axis_aligned


def cluster_by_aabb_iou(
    boxes: np.ndarray,
    iou_cluster_thr: float = 0.15,
    color_labels: Optional[List[Optional[str]]] = None,
    color_bonus: float = 0.0,
    color_penalty: float = 0.0,
) -> List[List[int]]:
    if boxes.size == 0:
        return []
    boxes_geom = boxes[:, -5:] if isinstance(boxes, np.ndarray) and boxes.ndim == 2 and boxes.shape[1] >= 5 else boxes
    cls_col = boxes[:, 0] if isinstance(boxes, np.ndarray) and boxes.ndim == 2 and boxes.shape[1] > 5 else None
    N = len(boxes_geom)
    used = np.zeros(N, dtype=bool)
    clusters: List[List[int]] = []
    use_color = (
        color_labels is not None
        and len(color_labels) == N
        and (color_bonus > 0.0 or color_penalty > 0.0)
    )
    normalized_colors: Optional[List[Optional[str]]] = None
    if use_color:
        normalized_colors = [normalize_color_label(c) for c in color_labels]
    for i in range(N):
        if used[i]:
            continue
        q = [i]
        used[i] = True
        cluster = [i]
        while q:
            k = q.pop()
            for j in range(N):
                if used[j]:
                    continue
                thr = iou_cluster_thr
                if use_color:
                    c1 = normalized_colors[k] if normalized_colors else None
                    c2 = normalized_colors[j] if normalized_colors else None
                    if c1 and c2:
                        if c1 == c2:
                            thr = max(iou_cluster_thr - color_bonus, 0.0)
                        else:
                            thr = min(iou_cluster_thr + color_penalty, 1.0)
                if cls_col is not None and cls_col[k] != cls_col[j]:
                    continue
                if aabb_iou_axis_aligned(boxes_geom[k], boxes_geom[j]) >= thr:
                    used[j] = True
                    q.append(j)
                    cluster.append(j)
        clusters.append(cluster)
    return clusters
