import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.tracking.cluster import cluster_by_aabb_iou
from utils.tracking.fusion import fuse_cluster_weighted


# ---------------- I/O ----------------
def load_cam_labels(pred_dir: str, frame_key: str) -> List[Tuple[str, np.ndarray]]:
    """
    cam*_frame_{frame_key}.txt -> [(cam_name, arr(N,5)), ...]
    ?? [cx, cy, L, W, yaw_deg]
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


# ---------------- ?? ----------------
def merge_frame_with_distance_weight(
    pred_dir: str,
    frame_key: str,
    out_dir: str,
    camera_setups: List[dict],
    iou_cluster_thr: float = 0.15,
    d0: float = 5.0,
    p: float = 2.0,
):
    os.makedirs(out_dir, exist_ok=True)

    cam_ground_xy = {item["name"]: (float(item["pos"]["x"]), float(item["pos"]["y"])) for item in camera_setups}

    cam_arrays = load_cam_labels(pred_dir, frame_key)
    if not cam_arrays:
        print(f"[warn] no inputs for frame {frame_key}")
        return None

    boxes_list, cams_list = [], []
    for cam, arr in cam_arrays:
        for row in arr:
            boxes_list.append(row)
            cams_list.append(cam)
    boxes_all = np.array(boxes_list, dtype=float)
    cams_all = list(cams_list)

    if boxes_all.size == 0:
        out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt")
        open(out_path, "w").close()
        print(f"? saved: {out_path} (0 objects)")
        return out_path

    keep_mask = (boxes_all[:, 2] * boxes_all[:, 3] >= 2.8) & ((boxes_all[:, 2] >= 2.0) | (boxes_all[:, 3] >= 2.0))
    boxes_supp = boxes_all[keep_mask]

    keep_idx = []
    used = np.zeros(len(boxes_all), dtype=bool)
    for b in boxes_supp:
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
    cams = [cams_all[i] for i in keep_idx]

    if boxes.size == 0:
        out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt")
        open(out_path, "w").close()
        print(f"? saved: {out_path} (0 objects)")
        return out_path

    print("??", frame_key, "?? ?? ??", len(boxes))
    clusters = cluster_by_aabb_iou(boxes, iou_cluster_thr=iou_cluster_thr)

    print("???? ??:", len(clusters))

    merged_list = []
    for idxs in clusters:
        print(" - ???? ??:", len(idxs))
        rep = fuse_cluster_weighted(
            boxes, cams, idxs, cam_ground_xy,
            d0=d0, p=p
        )
        merged_list.append(rep)

    merged = np.array(merged_list, dtype=float)

    out_path = os.path.join(out_dir, f"merged_frame_{frame_key}.txt")
    with open(out_path, "w") as f:
        for cx, cy, L, W, yaw in merged:
            f.write(f"0 {cx:.4f} {cy:.4f} {L:.4f} {W:.4f} {yaw:.2f}")
    print(f"? saved: {out_path} ({len(merged)} objects)")
    return out_path
