import os
import math
import glob
import numpy as np
from typing import List, Dict, Tuple, Optional

# ---------------- I/O ----------------
def load_cam_labels(pred_dir: str, frame_key: str) -> List[Tuple[str, np.ndarray]]:
    """
    cam*_frame_{frame_key}.txt -> [(cam_name, arr(N,5)), ...]
    각 행: [cx, cy, L, W, yaw_deg]
    """
    paths = sorted(glob.glob(os.path.join(pred_dir, f"cam*_frame_{frame_key}.txt")))
    result = []
    for path in paths:
        cam_name = os.path.basename(path).split("_frame_")[0]
        rows = []
        with open(path, "r") as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) != 6:
                    continue
                _, cx, cy, L, W, yaw = vals
                rows.append([float(cx), float(cy), float(L), float(W), float(yaw)])
        if rows:
            result.append((cam_name, np.array(rows, dtype=float)))
    return result

# ---------------- 기하/IoU ----------------
def aabb_iou_axis_aligned(b1, b2) -> float: # AABB IoU 계산함 
    x1,y1,L1,W1,_ = b1
    x2,y2,L2,W2,_ = b2
    A = [x1 - L1/2, y1 - W1/2, x1 + L1/2, y1 + W1/2]
    B = [x2 - L2/2, y2 - W2/2, x2 + L2/2, y2 + W2/2]
    ix1, iy1 = max(A[0], B[0]), max(A[1], B[1])
    ix2, iy2 = min(A[2], B[2]), min(A[3], B[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (L1*W1) + (L2*W2) - inter
    return inter/union if union > 1e-12 else 0.0

def obb_to_corners(cx, cy, L, W, yaw_deg): # OBB → 꼭짓점 변환함 
    yaw = math.radians(yaw_deg)
    dx, dy = L/2.0, W/2.0
    corners = np.array([[ dx,  dy],
                        [ dx, -dy],
                        [-dx, -dy],
                        [-dx,  dy]], dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],[s,  c]], dtype=float)
    return corners @ R.T + np.array([cx, cy], dtype=float)

def _sanitize_color_label(val: Optional[str]) -> Optional[str]:
    """
    Treat None/empty/"none" as missing so they don't affect color-based bonuses/penalties.
    """
    if val is None:
        return None
    c = str(val).strip().lower()
    if not c or c == "none":
        return None
    return c

# ---------------- 클러스터링 ----------------
def cluster_by_aabb_iou(
    boxes: np.ndarray,
    iou_cluster_thr: float = 0.15,
    color_labels: Optional[List[Optional[str]]] = None,
    color_bonus: float = 0.0,
    color_penalty: float = 0.0,
) -> List[List[int]]:
    # AABB IoU 계산해서 thr 이상이면 클러스터에 넣기
    if boxes.size == 0:
        return []
    N = len(boxes)
    used = np.zeros(N, dtype=bool)
    clusters: List[List[int]] = []
    normalized_colors: Optional[List[Optional[str]]] = None
    use_color = (
        color_labels is not None
        and len(color_labels) == N
        and (color_bonus > 0.0 or color_penalty > 0.0)
    )
    if use_color:
        normalized_colors = [_sanitize_color_label(c) for c in color_labels]
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
                if aabb_iou_axis_aligned(boxes[k], boxes[j]) >= thr:
                    used[j] = True
                    q.append(j)
                    cluster.append(j)
        clusters.append(cluster)
    return clusters

# 가중평균대표용 유틸s
def wrap_deg(a: float) -> float: # 각도 a를 [-180,180) 범위로 래핑 ~ 혹시 차량 앞뒤 뒤집히면 중간값 계산할때 각도 튀니까 
    return ((a + 180.0) % 360.0) - 180.0

def distance_weight(d: float, d0: float = 5.0, p: float = 2.0) -> float:
    """
    거리 가중치: w = (d0 / (d0 + d)) ** p  (0<w<=1)
    - d0: 특성 거리(작을수록 가까움 강조), p: 스케일링 지수
    """
    return float((d0 / (d0 + max(0.0, d))) ** p)

def _normalize_with_cap(ws: np.ndarray, max_frac: float = 0.7) -> np.ndarray:
    # 합 1로 정규화 후, 1개가 과도하게 크면 캡 씌우고 재정규화
    w = ws.astype(float).copy()
    s = w.sum()
    if s <= 0:
        w[:] = 1.0 / len(w)
        return w
    w /= s
    if w.max() <= max_frac:
        return w
    # 캡 적용
    over = w > max_frac
    excess = (w[over] - max_frac).sum()
    w[over] = max_frac
    remain = (~over)
    if remain.any():
        w[remain] += excess * (w[remain] / w[remain].sum())
    # 수치 안정
    w = np.clip(w, 1e-6, 1.0)
    return w / w.sum()

def _cauchy_weights(resid: np.ndarray, scale: float, c: float = 2.385) -> np.ndarray:
    # Tukey/Hampel도 가능. 여기선 Cauchy: 1 / (1 + (r/scale)^2)
    s = max(scale, 1e-6)
    return 1.0 / (1.0 + (resid / s) ** 2)

def _weighted_central_xy(boxes: np.ndarray, w: np.ndarray) -> Tuple[float,float]:
    # 중심점의 가중평균 ㄱ낸거
    cx = float(np.average(boxes[:,0], weights=w))
    cy = float(np.average(boxes[:,1], weights=w))
    return cx, cy

def _weighted_mean(v: np.ndarray, w: np.ndarray) -> float: # 가중평균 
    return float(np.average(v, weights=w))

def _weighted_circular_mean_deg(angles_deg: np.ndarray, w: np.ndarray) -> Tuple[float, float]: # 원형 가중평균
    # 반환: (mean_deg, resultant_length R in [0,1])
    ang = np.deg2rad(angles_deg)
    S = float(np.average(np.sin(ang), weights=w))
    C = float(np.average(np.cos(ang), weights=w))
    mean = math.degrees(math.atan2(S, C))
    R = math.hypot(S, C)  # 0~1
    return mean, R

def _weighted_circular_median_deg(angles_deg: np.ndarray, w: np.ndarray) -> float:
    # 간단한 근사: 0~360 정렬 후 누적가중 0.5 지점
    a = np.mod(angles_deg + 360.0, 360.0)
    idx = np.argsort(a)
    a_sorted = a[idx]; w_sorted = w[idx] / max(w.sum(), 1e-9)
    csum = np.cumsum(w_sorted)
    j = np.searchsorted(csum, 0.5)
    med = a_sorted[min(j, len(a_sorted)-1)]
    return wrap_deg(med)

def _align_angles_to_reference(angles_deg: np.ndarray, ref_deg: float, flip_threshold: float = 90.0) -> np.ndarray:
    # 각 angle을 ref 기준으로 차이가 90°를 넘으면 180° flip하여 정/역방향 뒤집힘 보정
    out = []
    for a in angles_deg:
        d = wrap_deg(a - ref_deg)
        if abs(d) > flip_threshold:
            out.append(wrap_deg(a + 180.0))
        else:
            out.append(wrap_deg(a))
    return np.array(out, dtype=float)

def fuse_cluster_weighted( # 가중 평균 대표 생성
    boxes: np.ndarray,
    cams: List[str],
    idxs: List[int],
    cam_ground_xy: Dict[str, Tuple[float, float]],
    d0: float = 5.0,
    p: float = 2.0,
    extra_weights: Optional[List[float]] = None,
    w_floor: float = 1e-4,
    max_cam_frac: float = 0.65,      # 가중치 한 캠에 못 쏠리게 
    yaw_flip_thr: float = 90.0,      # ref와 90° 초과면 180° 뒤집어 정렬 ~ 각도 못ㅋ튀게
    yaw_R_fallback: float = 0.25,    # 원형 평균 결과벡터 작으면 중앙값 사용 ~ 아웃라이어 각도 빼기용
    loc_scale_k: float = 1.4826,     # MAD→σ 변환계수
    size_scale_rel: float = 0.20     # 크기 잔차 스케일(상대 20%)
) -> np.ndarray:
    """
    강건 가중 대표 생성:
      1) 거리 가중 → 정규화(+캡)
      2) 중심/크기 잔차로 재가중(이상치 다운웨이트)
      3) yaw는 기준각에 맞춰 180° flip 정렬 → 원형평균(R 작으면 중앙값)
    """
    cluster_boxes = np.array([boxes[i] for i in idxs], dtype=float) 
    cluster_cams  = [cams[i]  for i in idxs]
    M = len(cluster_boxes)
    if M == 1:
        return cluster_boxes[0].astype(float)

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

    # 캠 - 객체 거리 가중치---
    w_dist = []
    for idx, (b, cam) in enumerate(zip(cluster_boxes, cluster_cams)):
        cx, cy = b[0], b[1]
        cam_xy = cam_ground_xy.get(cam, (0.0, 0.0))
        d = math.hypot(cx - cam_xy[0], cy - cam_xy[1])
        bias = weight_bias[idx] if idx < len(weight_bias) else 1.0
        w = distance_weight(d, d0=d0, p=p) * max(bias, 1e-6)
        w_dist.append(max(w, w_floor))
    w_dist = np.array(w_dist, dtype=float)
    w_base = _normalize_with_cap(w_dist, max_frac=max_cam_frac) 

    # 위에만 썼더니 가중치 덜먹어서 huber커널로 가중치 더 줌---
    # 중심(절대편차)
    cx0, cy0 = _weighted_central_xy(cluster_boxes, w_base)
    r_loc = np.hypot(cluster_boxes[:,0] - cx0, cluster_boxes[:,1] - cy0)
    # scale: 가중 중앙절대편차(MAD) 근사
    med_r = float(np.average(np.sort(r_loc), weights=np.sort(w_base))) if M>1 else float(r_loc.mean())
    scale_loc = max(med_r * loc_scale_k, 1e-3)
    w_loc = _cauchy_weights(r_loc, scale_loc)

    # 크기(상대오차)
    Ls = cluster_boxes[:,2]; Ws = cluster_boxes[:,3]
    L0 = _weighted_mean(Ls, w_base); W0 = _weighted_mean(Ws, w_base)
    rel_L = np.abs(Ls - L0) / max(abs(L0), 1e-6)
    rel_W = np.abs(Ws - W0) / max(abs(W0), 1e-6)
    r_size = 0.5 * (rel_L + rel_W)
    scale_size = max(size_scale_rel, 1e-3)  # 예: 20% 상대편차 정도까지 관대
    w_size = _cauchy_weights(r_size, scale_size)

    # 종합 가중(정규화/캡 포함)
    w_comb = _normalize_with_cap(w_base * w_loc * w_size, max_frac=max_cam_frac)

    # 각도처리 yaw 강건 평균---
    yaws = cluster_boxes[:,4].astype(float)

    # 3-1) 초기 기준각: 거리+재가중 가중 원형평균
    ref_yaw, R0 = _weighted_circular_mean_deg(yaws, w_comb)

    # 3-2) ref 기준으로 180° flip 정렬
    yaws_aligned = _align_angles_to_reference(yaws, ref_yaw, flip_threshold=yaw_flip_thr)

    # 3-3) 정렬된 각도로 원형 평균
    mean_yaw, R = _weighted_circular_mean_deg(yaws_aligned, w_comb)
    if R < yaw_R_fallback:
        # 한두개만 튀면 걍 중앙값 폴백
        mean_yaw = _weighted_circular_median_deg(yaws_aligned, w_comb)

    # ---------- 4) 위치/크기 최종 가중 평균 ----------
    mean_cx = float(np.average(cluster_boxes[:,0], weights=w_comb))
    mean_cy = float(np.average(cluster_boxes[:,1], weights=w_comb))
    mean_L  = float(np.average(Ls, weights=w_comb))
    mean_W  = float(np.average(Ws, weights=w_comb))

    return np.array([mean_cx, mean_cy, mean_L, mean_W, wrap_deg(mean_yaw)], dtype=float)

# ---------------- 메인 ----------------
def merge_frame_with_distance_weight(
    pred_dir: str,
    frame_key: str,
    out_dir: str,
    camera_setups: List[dict],
    iou_cluster_thr: float = 0.15,   # AABB IoU 클러스터 임계
    d0: float = 5.0,
    p: float = 2.0,
):
    os.makedirs(out_dir, exist_ok=True)

    cam_ground_xy = {item["name"]: (float(item["pos"]["x"]), float(item["pos"]["y"])) for item in camera_setups}

    cam_arrays = load_cam_labels(pred_dir, frame_key)
    if not cam_arrays:
        print(f"[warn] no inputs for frame {frame_key}")
        return None

    # 한 배열로 합치고, 출처 cam 보관
    boxes_list, cams_list = [], []
    for cam, arr in cam_arrays:
        for row in arr:
            boxes_list.append(row)
            cams_list.append(cam)
    boxes_all = np.array(boxes_list, dtype=float)
    cams_all  = list(cams_list)

    if boxes_all.size == 0:
        out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt")
        open(out_path, "w").close()
        print(f"✅ saved: {out_path} (0 objects)")
        return out_path

    # 가로*세로=1.6 미만의 작은 객체거나 가로와세로모두 2보다 작은 객체 임의 제거
    keep_mask = (boxes_all[:,2] * boxes_all[:,3] >= 2.8) & ((boxes_all[:,2] >= 2.0) | (boxes_all[:,3] >= 2.0)) # 2.8보다 크고 가로나 세로중 하나는 2이상
    boxes_supp = boxes_all[keep_mask]
    # cams_all  = [cams_all[i] for i in range(len(cams_all)) if keep_mask[i]]
    # 여기까지 하드코딩이므로 나중에 제거...
    # areas = boxes_all[:,2] * boxes_all[:,3]
    # keep_mask = areas >= 2.8
    # boxes_all = boxes_all[keep_mask]
    # cams_all  = [cams_all[i] for i in range(len(cams_all)) if keep_mask[i]]

    # 남은 박스들을 원본 인덱스로 재매칭
    keep_idx = []
    used = np.zeros(len(boxes_all), dtype=bool)
    for b in boxes_supp:
    # for b in boxes_all:
        found = None
        for i, bb in enumerate(boxes_all):
            if used[i]:
                continue
            if np.allclose(b, bb, rtol=0, atol=1e-7):
                found = i
                break
        if found is None:
            diffs = np.linalg.norm(boxes_all - b, axis=1)
            i = int(np.argmin(diffs))
            if used[i]:
                continue
            found = i
        used[found] = True
        keep_idx.append(found)

    boxes = boxes_all[keep_idx]
    cams  = [cams_all[i] for i in keep_idx]

    if boxes.size == 0:
        out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt")
        open(out_path, "w").close()
        print(f"✅ saved: {out_path} (0 objects)")
        return out_path

    print("현재", frame_key, "모든 박스 개수", len(boxes))
    clusters = cluster_by_aabb_iou(boxes, iou_cluster_thr=iou_cluster_thr)

    print("클러스터 개수:", len(clusters))

    merged_list = []
    for idxs in clusters:
        print(" - 클러스터 크기:", len(idxs))
        rep = fuse_cluster_weighted(
            boxes, cams, idxs, cam_ground_xy,
            d0=d0, p=p
        )
        merged_list.append(rep)

    merged = np.array(merged_list, dtype=float)

    # 저장
    out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt")
    with open(out_path, "w") as f:
        for cx, cy, L, W, yaw in merged:
            f.write(f"0 {cx:.4f} {cy:.4f} {L:.4f} {W:.4f} {yaw:.2f}\n")
    print(f"✅ saved: {out_path} ({len(merged)} objects)")
    return out_path


if __name__ == "__main__":
    pred_dir = "/inference_results_onnx/bev_labels"
    out_dir  = "/merge_dist_wbf"
    os.makedirs(out_dir, exist_ok=True)

    # 폴더내 싹 돌면서 프레임키 춫ㄹ
    files = sorted(glob.glob(os.path.join(pred_dir, "cam*_frame_*.txt")))
    frame_keys = set()
    for f in files:
        name = os.path.basename(f)
        parts = name.split("_frame_")
        if len(parts) >= 2:
            key_part = parts[1]
            key = os.path.splitext(key_part)[0]
            frame_keys.add(key)

    frame_keys = sorted(frame_keys)
    print(f"Found {len(frame_keys)} frame keys to process")
    for frame_key in frame_keys: # 프레임 마다
        merge_frame_with_distance_weight(
            pred_dir, frame_key, out_dir, CAMERA_SETUPS,
            iou_cluster_thr=0.25, # 0.25 사용!
            d0=5.0,
            p=2.0
        )
