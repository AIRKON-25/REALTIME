from typing import List, Optional

import numpy as np

from utils.colors import normalize_color_label
from utils.tracking.geometry import aabb_iou_axis_aligned
from utils.tracking._constants import IOU_CLUSTER_THR, COLOR_BONUS, COLOR_PENALTY

def cluster_by_aabb_iou(
    boxes: np.ndarray,
    iou_cluster_thr: float = IOU_CLUSTER_THR,
    color_labels: Optional[List[Optional[str]]] = None,
    color_bonus: float = COLOR_BONUS,
    color_penalty: float = COLOR_PENALTY,
) -> List[List[int]]:
    """
    Axis-aligned bounding box 기반 IoU 클러스터링 수행
    :param boxes: (N, >=5) 형태의 numpy 배열. 마지막 5개 열이 (x, y, z, w, l) 형식의 AABB 정보를 담고 있어야 함
    :param iou_cluster_thr: 두 박스 간 IoU가 이 임계값 이상일 때 동일 클러스터로 간주
    :param color_labels: (N,) 길이의 리스트로, 각 박스에 대응하는 색상 레이블(문자열) 정보
    :param color_bonus: 동일 색상일 때 IoU 임계값에서 이 값을 차감
    :param color_penalty: 다른 색상일 때 IoU 임계값에 이 값을 가산
    :return: 클러스터링된 박스 인덱스의 리스트. 각 내부 리스트는 하나의 클러스터에 속하는 박스 인덱스를 담음
    """
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
