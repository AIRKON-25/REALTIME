# inference_lstm_trt_npz.py
# YOLO11 2.5D temporal inference with TensorRT engine (ConvLSTM/GRU hidden-state carry + seq reset)
# + BEV by per-pixel LUT (npz) with bilinear sampling (X,Y,Z) & robust masks

import os
import cv2
import math
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, Future

import torch
from tqdm import tqdm

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (creates CUDA context)

# ====== 기존 유틸 ======
from utils.edge.geometry_utils import parallelogram_from_triangle, tiny_filter_on_dets
from utils.edge.evaluation_utils import (
    decode_predictions,
    evaluate_single_image,
    compute_detection_metrics,
    orientation_error_deg,
)

# ====== Matplotlib (optional) ======
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.lines import Line2D
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MplPolygon = None
    Line2D = None
    _MATPLOTLIB_AVAILABLE = False


# =========================
# I/O helpers (라벨/BEV/시각화)
# =========================
def load_gt_triangles(label_path: str) -> np.ndarray:
    if not os.path.isfile(label_path):
        return np.zeros((0, 3, 2), dtype=np.float32)
    tris = []
    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 7:
                continue
            _, p0x, p0y, p1x, p1y, p2x, p2y = p
            tris.append([[float(p0x), float(p0y)],
                         [float(p1x), float(p1y)],
                         [float(p2x), float(p2y)]])
    if len(tris) == 0:
        return np.zeros((0, 3, 2), dtype=np.float32)
    return np.asarray(tris, dtype=np.float32)


def poly_from_tri(tri: np.ndarray) -> np.ndarray:
    p0, p1, p2 = tri[0], tri[1], tri[2]
    return parallelogram_from_triangle(p0, p1, p2).astype(np.float32)


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def iou_polygon(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    try:
        pa = poly_a.astype(np.float32)
        pb = poly_b.astype(np.float32)
        inter_area, _ = cv2.intersectConvexConvex(pa, pb)
        if inter_area <= 0:
            return 0.0
        ua = polygon_area(pa)
        ub = polygon_area(pb)
        union = ua + ub - inter_area
        return float(inter_area / max(union, 1e-9))
    except Exception:
        xa1, ya1 = poly_a[:, 0].min(), poly_a[:, 1].min()
        xa2, ya2 = poly_a[:, 0].max(), poly_a[:, 1].max()
        xb1, yb1 = poly_b[:, 0].min(), poly_b[:, 1].min()
        xb2, yb2 = poly_b[:, 0].max(), poly_b[:, 1].max()
        inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
        inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
        inter = inter_w * inter_h
        ua = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        ub = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = ua + ub - inter
        return float(inter / max(union, 1e-9))


def draw_pred_only(image_bgr, dets, save_path_img, save_path_txt, W, H, W0, H0):
    draw_img = save_path_img is not None
    write_txt = save_path_txt is not None

    if draw_img:
        os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
        img = image_bgr.copy()
    else:
        img = None

    if write_txt:
        os.makedirs(os.path.dirname(save_path_txt), exist_ok=True)
        lines = []
    else:
        lines = None

    sx, sy = float(W0) / float(W), float(H0) / float(H)
    tri_orig_list: List[np.ndarray] = []
    for d in dets:
        tri = np.asarray(d["tri"], dtype=np.float32)
        score = float(d["score"])
        if draw_img:
            poly4 = parallelogram_from_triangle(tri[0], tri[1], tri[2]).astype(np.int32)
            cv2.polylines(img, [poly4], isClosed=True, color=(0, 255, 0), thickness=2)
            cx, cy = int(tri[0][0]), int(tri[0][1])
            cv2.putText(img, f"{score:.2f}", (cx, max(0, cy - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        tri_orig = tri.copy()
        tri_orig[:, 0] *= sx
        tri_orig[:, 1] *= sy
        tri_orig_list.append(tri_orig.copy())
        if write_txt:
            p0o, p1o, p2o = tri_orig[0], tri_orig[1], tri_orig[2]
            lines.append(
                f"0 {p0o[0]:.2f} {p0o[1]:.2f} {p1o[0]:.2f} {p1o[1]:.2f} {p2o[0]:.2f} {p2o[1]:.2f} {score:.4f}"
            )

    if draw_img:
        cv2.imwrite(save_path_img, img)
    if write_txt and lines is not None:
        with open(save_path_txt, "w") as f:
            f.write("\n".join(lines))
    return tri_orig_list


def draw_pred_with_gt(image_bgr_resized, dets, gt_tris_resized, save_path_img_mix, iou_thr=0.5):
    if save_path_img_mix is None:
        return

    os.makedirs(os.path.dirname(save_path_img_mix), exist_ok=True)
    img = image_bgr_resized.copy()
    for g in gt_tris_resized:
        poly_g = poly_from_tri(g).astype(np.int32)
        cv2.polylines(img, [poly_g], True, (0, 255, 0), 2)
        for k in range(3):
            x = int(round(float(g[k, 0]))); y = int(round(float(g[k, 1])))
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
    for d in dets:
        tri = np.asarray(d["tri"], dtype=np.float32)
        score = float(d["score"])
        poly4 = poly_from_tri(tri).astype(np.int32)
        cv2.polylines(img, [poly4], True, (0, 0, 255), 2)
        p0 = tri[0].astype(int)
        cv2.putText(img, f"{score:.2f}", (int(p0[0]), max(0, int(p0[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if gt_tris_resized.shape[0] > 0 and len(dets) > 0:
        det_idx_sorted = np.argsort([-float(d["score"]) for d in dets])
        matched = np.zeros((gt_tris_resized.shape[0],), dtype=bool)
        for di in det_idx_sorted:
            d = dets[di]
            tri_d = np.asarray(d["tri"], dtype=np.float32)
            poly_d = poly_from_tri(tri_d)
            best_j, best_iou = -1, 0.0
            for j, gtri in enumerate(gt_tris_resized):
                if matched[j]:
                    continue
                poly_g = poly_from_tri(gtri)
                iou = iou_polygon(poly_d, poly_g)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                matched[best_j] = True
                p0 = tri_d[0].astype(int)
                cv2.putText(img, f"IoU {best_iou:.2f}",
                            (int(p0[0]), int(p0[1]) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(save_path_img_mix, img)


def _load_and_preprocess_image(path: str, target_hw: Tuple[int, int]):
    img_bgr0 = cv2.imread(path)
    if img_bgr0 is None:
        return None

    target_h, target_w = target_hw
    img_bgr = cv2.resize(img_bgr0, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, 0).copy()  # ensure contiguous

    H0, W0 = img_bgr0.shape[:2]
    scale_resize_x = target_w / float(W0)
    scale_resize_y = target_h / float(H0)
    scale_to_orig_x = float(W0) / float(target_w)
    scale_to_orig_y = float(H0) / float(target_h)

    return {
        "img_bgr": img_bgr,
        "img_np": img_np,
        "H0": H0,
        "W0": W0,
        "scale_resize_x": scale_resize_x,
        "scale_resize_y": scale_resize_y,
        "scale_to_orig_x": scale_to_orig_x,
        "scale_to_orig_y": scale_to_orig_y,
    }


def normalize_angle_deg(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


# =========================
# BEV 시각화 (2D 이미지)
# =========================
def _prepare_bev_canvas(polygons: List[np.ndarray], padding: float = 1.0, target: float = 800.0):
    if not polygons:
        return None
    pts = np.concatenate(polygons, axis=0)
    if pts.size == 0:
        return None
    min_x = float(np.nanmin(pts[:, 0]) - padding)
    max_x = float(np.nanmax(pts[:, 0]) + padding)
    min_y = float(np.nanmin(pts[:, 1]) - padding)
    max_y = float(np.nanmax(pts[:, 1]) + padding)
    range_x = max(max_x - min_x, 1e-3)
    range_y = max(max_y - min_y, 1e-3)
    scale = target / max(range_x, range_y)
    width = int(max(range_x * scale, 300))
    height = int(max(range_y * scale, 300))
    return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
            "scale": scale, "width": width, "height": height}


def _to_canvas(points: np.ndarray, params: dict) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xs = (pts[:, 0] - params["min_x"]) * params["scale"]
    ys = (params["max_y"] - pts[:, 1]) * params["scale"]
    return np.stack([xs, ys], axis=1).astype(np.int32)


def draw_bev_visualization(preds_bev: List[dict], gt_tris_bev: Optional[np.ndarray],
                           save_path_img: str, title: str):
    pred_polys = [det["poly"] for det in preds_bev]
    gt_polys = [poly_from_tri(tri) for tri in gt_tris_bev] if gt_tris_bev is not None else []
    polygons = pred_polys + gt_polys
    params = _prepare_bev_canvas(polygons)
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)

    if params is None:
        canvas = np.full((360, 360, 3), 240, dtype=np.uint8)
        cv2.putText(canvas, "No BEV data", (60, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(save_path_img, canvas)
        return

    if _MATPLOTLIB_AVAILABLE:
        try:
            fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=220)
            ax.set_facecolor("#f7f7f7")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title, fontsize=11, pad=10)
            ax.set_xlabel("X (m)", fontsize=10)
            ax.set_ylabel("Y (m)", fontsize=10)
            ax.set_xlim(params["min_x"], params["max_x"])
            ax.set_ylim(params["max_y"], params["min_y"])

            if params["min_x"] <= 0.0 <= params["max_x"]:
                ax.axvline(0.0, color="#999999", linewidth=0.9,
                           linestyle="--", alpha=0.8)
            if params["min_y"] <= 0.0 <= params["max_y"]:
                ax.axhline(0.0, color="#999999", linewidth=0.9,
                           linestyle="--", alpha=0.8)

            legend_handles = []

            if gt_polys:
                for poly in gt_polys:
                    if not np.all(np.isfinite(poly)):
                        continue
                    patch = MplPolygon(poly, closed=True, fill=False,
                                       edgecolor="#27ae60", linewidth=1.8)
                    ax.add_patch(patch)
                    ax.scatter(poly[:, 0], poly[:, 1],
                               s=8, color="#27ae60", alpha=0.9)
                legend_handles.append(
                    Line2D([0], [0], color="#27ae60", lw=2, label="GT")
                )

            if preds_bev:
                dy = max(params["max_y"] - params["min_y"], 1e-3)
                text_offset = 0.015 * dy
                for det in preds_bev:
                    poly = det["poly"]
                    if not np.all(np.isfinite(poly)):
                        continue
                    patch = MplPolygon(poly, closed=True, fill=False,
                                       edgecolor="#e74c3c", linewidth=1.8)
                    ax.add_patch(patch)
                    center = det["center"]
                    ax.scatter(center[0], center[1],
                               s=20, color="#e74c3c", alpha=0.9)
                    f1, f2 = det.get("front_edge", (None, None))
                    if (f1 is not None and f2 is not None
                            and np.all(np.isfinite(f1)) and np.all(np.isfinite(f2))):
                        ax.plot([f1[0], f2[0]],
                                [f1[1], f2[1]],
                                linewidth=2.2, color="#1f77b4")
                    label = f"{det['score']:.2f} / {det['yaw']:.1f}°"
                    ax.text(center[0], center[1] + text_offset, label,
                            fontsize=7.5, color="#e74c3c",
                            ha="center", va="bottom",
                            bbox=dict(facecolor="#ffffff", alpha=0.6,
                                      edgecolor="none", pad=1.5))
                legend_handles.append(
                    Line2D([0], [0], color="#e74c3c", lw=2, label="Pred")
                )
                legend_handles.append(
                    Line2D([0], [0], color="#1f77b4", lw=3, label="Front edge")
                )

            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper right",
                          frameon=True, framealpha=0.75, fontsize=8)

            fig.tight_layout(pad=0.6)
            fig.savefig(save_path_img, dpi=220)
            plt.close(fig)
            return
        except Exception:
            plt.close("all")

    # OpenCV fallback
    canvas = np.full((params["height"], params["width"], 3), 255, dtype=np.uint8)
    axis_color = (120, 120, 120)
    axis_thickness = 1

    if params["min_x"] <= 0.0 <= params["max_x"]:
        axis_x = np.array([[0.0, params["min_y"]],
                           [0.0, params["max_y"]]], dtype=np.float64)
        axis_x_px = _to_canvas(axis_x, params)
        cv2.line(canvas, tuple(axis_x_px[0]), tuple(axis_x_px[1]),
                 axis_color, axis_thickness, cv2.LINE_AA)

    if params["min_y"] <= 0.0 <= params["max_y"]:
        axis_y = np.array([[params["min_x"], 0.0],
                           [params["max_x"], 0.0]], dtype=np.float64)
        axis_y_px = _to_canvas(axis_y, params)
        cv2.line(canvas, tuple(axis_y_px[0]), tuple(axis_y_px[1]),
                 axis_color, axis_thickness, cv2.LINE_AA)

    for det in preds_bev:
        poly = det["poly"]
        if not np.all(np.isfinite(poly)):
            continue
        poly_px = _to_canvas(poly, params)
        cv2.polylines(canvas, [poly_px], True, (0, 0, 255), 2)
        center_px = _to_canvas(
            np.asarray(det["center"]).reshape(1, 2), params
        )[0]
        cv2.circle(canvas, tuple(center_px), 4, (0, 0, 255), -1)

    cv2.putText(canvas, title, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    coord_text = (
        f"x:[{params['min_x']:.2f},{params['max_x']:.2f}]  "
        f"y:[{params['min_y']:.2f},{params['max_y']:.2f}]"
    )
    cv2.putText(canvas, coord_text, (10, params["height"] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.imwrite(save_path_img, canvas)


def compute_bev_properties(tri_xy, tri_z=None,
                           pitch_clamp_deg: float = 15.0,
                           use_roll: bool = False,
                           roll_threshold_deg: float = 2.0,
                           roll_clamp_deg: float = 8.0):
    """
    tri_xy: (3,2) with p0 (rear), p1/p2 (front-left/right)
    tri_z:  (3,) optional Z at the same three points
    returns:
      center(x,y), length, width, yaw(deg), front_edge(p1,p2),
      cz (float), pitch_deg, roll_deg
    """
    p0, p1, p2 = np.asarray(tri_xy, dtype=np.float64)
    if not np.all(np.isfinite(tri_xy)):
        return None

    # yaw from XY
    front_center = (p1 + p2) / 2.0
    front_vec = front_center - p0
    yaw = math.degrees(math.atan2(front_vec[1], front_vec[0]))
    yaw = (yaw + 180) % 360 - 180

    poly = parallelogram_from_triangle(p0, p1, p2)
    edges = [np.linalg.norm(poly[(i + 1) % 4] - poly[i]) for i in range(4)]
    length = max(edges)
    width = min(edges)
    center = poly.mean(axis=0)
    front_edge = (p1, p2)

    cz = 0.0
    pitch_deg = 0.0
    roll_deg = 0.0
    if tri_z is not None and np.all(np.isfinite(tri_z)):
        z0, z1, z2 = float(tri_z[0]), float(tri_z[1]), float(tri_z[2])
        cz = (z0 + z1 + z2) / 3.0

        # pitch: front vs rear along length
        z_front = 0.5 * (z1 + z2)
        dz_len = z_front - z0
        if length > 1e-6:
            pitch_rad = math.atan2(dz_len, length)
            pitch_deg = math.degrees(pitch_rad)
            pitch_deg = float(np.clip(pitch_deg, -pitch_clamp_deg, pitch_clamp_deg))

        # roll: left-right difference at front edge (optional)
        if use_roll:
            dz_lr = z2 - z1  # right - left
            if width > 1e-6:
                roll_rad = math.atan2(dz_lr, width)
                roll_deg = math.degrees(roll_rad)
                if abs(roll_deg) < roll_threshold_deg:
                    roll_deg = 0.0
                roll_deg = float(np.clip(roll_deg, -roll_clamp_deg, roll_clamp_deg))

    return (float(center[0]), float(center[1])), float(length), float(width), float(yaw), front_edge, float(cz), float(pitch_deg), float(roll_deg)


def compute_bev_properties_3d(
    tri_xy: np.ndarray,
    tri_z: np.ndarray,
    pitch_clamp_deg: float = 15.0,
    use_roll: bool = False,
    roll_threshold_deg: float = 2.0,
    roll_clamp_deg: float = 8.0,
    xy_scale: float = 1.0,
    z_scale: float = 1.0
):
    """
    3D plane-based length/width using plane geometry.
    tri_xy: (3,2) with p0(rear), p1/p2(front-left/right)
    tri_z : (3,)
    """
    if tri_xy is None or tri_z is None:
        return None
    tri_xy = np.asarray(tri_xy, dtype=np.float64)
    tri_z = np.asarray(tri_z, dtype=np.float64).reshape(-1)
    if (tri_xy.shape != (3, 2) or tri_z.shape != (3,)
            or not (np.all(np.isfinite(tri_xy)) and np.all(np.isfinite(tri_z)))):
        return None

    sxy = float(xy_scale) if xy_scale is not None else 1.0
    sz = float(z_scale) if z_scale is not None else 1.0
    if sxy <= 0:
        sxy = 1.0
    if sz <= 0:
        sz = 1.0

    tri_xy_unscaled = tri_xy / sxy
    tri_z_scaled = tri_z * sz

    P0 = np.array([tri_xy_unscaled[0, 0], tri_xy_unscaled[0, 1], tri_z_scaled[0]], dtype=np.float64)
    P1 = np.array([tri_xy_unscaled[1, 0], tri_xy_unscaled[1, 1], tri_z_scaled[1]], dtype=np.float64)
    P2 = np.array([tri_xy_unscaled[2, 0], tri_xy_unscaled[2, 1], tri_z_scaled[2]], dtype=np.float64)
    Pf = 0.5 * (P1 + P2)

    n = np.cross(P1 - P0, P2 - P0)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return None
    n /= n_norm

    L_hat = Pf - P0
    Ln = np.linalg.norm(L_hat)
    if Ln < 1e-9:
        return None
    L_hat /= Ln
    W_hat = np.cross(n, L_hat)
    Wn = np.linalg.norm(W_hat)
    if Wn < 1e-9:
        return None
    W_hat /= Wn

    length_m = abs(np.dot(Pf - P0, L_hat))
    width_m = abs(np.dot(P2 - P1, W_hat))

    P3 = P1 + (P2 - P0)
    center_3d = 0.25 * (P0 + P1 + P2 + P3)
    cz = float(center_3d[2])

    yaw = math.degrees(math.atan2(L_hat[1], L_hat[0]))
    yaw = (yaw + 180) % 360 - 180

    v = Pf - P0
    horiz_len = np.linalg.norm([v[0], v[1]])
    pitch_rad = math.atan2(v[2], max(horiz_len, 1e-9))
    pitch_deg = float(np.clip(math.degrees(pitch_rad), -pitch_clamp_deg, pitch_clamp_deg))

    roll_deg = 0.0
    if use_roll and width_m > 1e-6:
        dz_lr = (P2[2] - P1[2])
        roll_rad = math.atan2(dz_lr, width_m)
        roll_deg = math.degrees(roll_rad)
        if abs(roll_deg) < roll_threshold_deg:
            roll_deg = 0.0
        roll_deg = float(np.clip(roll_deg, -roll_clamp_deg, roll_clamp_deg))

    p0 = tri_xy[0]; p1 = tri_xy[1]; p2 = tri_xy[2]
    poly_xy = parallelogram_from_triangle(p0, p1, p2).astype(np.float32)
    front_edge = (tri_xy[1], tri_xy[2])
    center_xy = poly_xy.mean(axis=0)

    return (float(center_xy[0]), float(center_xy[1])), float(length_m), float(width_m), float(yaw), front_edge, float(cz), float(pitch_deg), float(roll_deg)


def write_bev_labels(save_path: str, bev_dets: List[dict], write_3d: bool = True):
    """
    write_3d=True → 'class cx cy cz length width yaw pitch roll'
    write_3d=False → 'class cx cy length width yaw'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = []
    for det in bev_dets:
        cx, cy = det["center"]
        length = det["length"]
        width = det["width"]
        yaw = det["yaw"]
        if write_3d:
            cz = det.get("cz", 0.0)
            pitch = det.get("pitch", 0.0)
            roll = det.get("roll", 0.0)
            lines.append(
                f"0 {cx:.4f} {cy:.4f} {cz:.4f} {length:.4f} {width:.4f} {yaw:.2f} {pitch:.2f} {roll:.2f}"
            )
        else:
            lines.append(
                f"0 {cx:.4f} {cy:.4f} {length:.4f} {width:.4f} {yaw:.2f}"
            )
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def evaluate_single_image_bev(preds_bev: List[dict], gt_tris_bev: np.ndarray, iou_thr=0.5):
    gt_arr = np.asarray(gt_tris_bev)
    num_gt = gt_arr.shape[0] if gt_arr.ndim == 3 else 0
    preds_sorted = sorted(preds_bev, key=lambda d: d["score"], reverse=True)
    if num_gt == 0:
        records = [(det["score"], 0, 0.0, None) for det in preds_sorted]
        return records, 0
    gt_polys = [poly_from_tri(tri) for tri in gt_arr]
    matched = np.zeros((num_gt,), dtype=bool)
    records = []
    for det in preds_sorted:
        poly_d = det["poly"]
        if not np.all(np.isfinite(poly_d)):
            records.append((det["score"], 0, 0.0, None))
            continue
        best_iou, best_idx = 0.0, -1
        for idx, poly_g in enumerate(gt_polys):
            if matched[idx] or not np.all(np.isfinite(poly_g)):
                continue
            iou = iou_polygon(poly_d, poly_g)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0 and best_iou >= iou_thr:
            matched[best_idx] = True
            orient_err = orientation_error_deg(det["tri"], gt_arr[best_idx])
            records.append((det["score"], 1, best_iou, orient_err))
        else:
            records.append((det["score"], 0, best_iou, None))
    return records, int(matched.sum())


# =========================
# LUT 기반 보간 (핵심 수정 부)
# =========================
def _lut_pick_valid_mask(lut):
    """
    LUT에서 사용할 valid mask를 우선순위로 선택:
    1) ground_valid_mask
    2) valid_mask
    3) floor_mask
    4) fallback: isfinite(X)&isfinite(Y)
    """
    X = np.asarray(lut["X"])
    Y = np.asarray(lut["Y"])
    H, W = X.shape

    for key in ("floor_mask", "ground_valid_mask", "valid_mask"):
        if key in lut:
            V = np.asarray(lut[key]).astype(bool)
            break
    else:
        V = np.isfinite(X) & np.isfinite(Y)

    if V.ndim == 1 and V.size == H * W:
        V = V.reshape(H, W)
    elif V.shape != (H, W):
        V = np.resize(V.astype(bool), (H, W))

    return V


def _bilinear_lut_xy(lut, u, v, min_valid_corners: int = 3, boundary_eps: float = 1e-3):
    """
    lut: dict-like with 'X','Y', and a valid mask
    u,v: 1D float arrays (pixel coords, 0..W-1 / 0..H-1)
    returns: Xw, Yw, valid (same shape as u/v)

    - Boundary clamp to avoid out-of-range indices
    - Allow interpolation if at least min_valid_corners corners are valid
    """
    X = np.asarray(lut["X"])
    Y = np.asarray(lut["Y"])
    V = _lut_pick_valid_mask(lut)

    H, W = X.shape
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()

    eps = float(boundary_eps)
    if not np.isfinite(eps) or eps <= 0:
        eps = 1e-3
    u = np.clip(u, 0.0, W - 1 - eps)
    v = np.clip(v, 0.0, H - 1 - eps)

    Xw = np.full(u.shape, np.nan, dtype=np.float32)
    Yw = np.full(v.shape, np.nan, dtype=np.float32)
    valid = np.zeros(u.shape, dtype=bool)

    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1

    ok = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)
    if not np.any(ok):
        return Xw, Yw, valid

    u0_ok = u0[ok]; v0_ok = v0[ok]
    u1_ok = u1[ok]; v1_ok = v1[ok]
    du = u[ok] - u0_ok
    dv = v[ok] - v0_ok

    X00 = X[v0_ok, u0_ok]; X10 = X[v0_ok, u1_ok]
    X01 = X[v1_ok, u0_ok]; X11 = X[v1_ok, u1_ok]
    Y00 = Y[v0_ok, u0_ok]; Y10 = Y[v0_ok, u1_ok]
    Y01 = Y[v1_ok, u0_ok]; Y11 = Y[v1_ok, u1_ok]
    V00 = V[v0_ok, u0_ok]; V10 = V[v0_ok, u1_ok]
    V01 = V[v1_ok, u0_ok]; V11 = V[v1_ok, u1_ok]

    min_c = int(min_valid_corners)
    min_c = 0 if min_c < 0 else (4 if min_c > 4 else min_c)
    nvalid = V00.astype(int) + V10.astype(int) + V01.astype(int) + V11.astype(int)
    allow = nvalid >= min_c

    if np.any(allow):
        w00 = (1.0 - du) * (1.0 - dv)
        w10 = du * (1.0 - dv)
        w01 = (1.0 - du) * dv
        w11 = du * dv

        # Use only valid corners
        w00[~V00] = 0.0; w10[~V10] = 0.0
        w01[~V01] = 0.0; w11[~V11] = 0.0
        wsum = w00 + w10 + w01 + w11
        wsum[wsum == 0.0] = 1.0
        w00 /= wsum; w10 /= wsum
        w01 /= wsum; w11 /= wsum

        Xw_ok = (w00 * X00 + w10 * X10 + w01 * X01 + w11 * X11).astype(np.float32)
        Yw_ok = (w00 * Y00 + w10 * Y10 + w01 * Y01 + w11 * Y11).astype(np.float32)

        whole = np.zeros_like(ok, dtype=bool)
        whole[ok] = allow
        Xw[whole] = Xw_ok[allow]
        Yw[whole] = Yw_ok[allow]
        valid[whole] = True

    return Xw, Yw, valid


def _bilinear_lut_xyz(lut, u, v, min_valid_corners: int = 3, boundary_eps: float = 1e-3):
    """XY 보간에 Z까지 확장. LUT에 Z가 없으면 Z는 NaN."""
    Xw, Yw, valid = _bilinear_lut_xy(
        lut, u, v,
        min_valid_corners=min_valid_corners,
        boundary_eps=boundary_eps
    )
    Z = lut.get("Z", None)
    if Z is None:
        Zw = np.full_like(Xw, np.nan, dtype=np.float32)
        return Xw, Yw, Zw, valid

    Z = np.asarray(Z)
    H, W = Z.shape
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()

    eps = float(boundary_eps) if np.isfinite(boundary_eps) and boundary_eps > 0 else 1e-3
    u = np.clip(u, 0.0, W - 1 - eps)
    v = np.clip(v, 0.0, H - 1 - eps)

    Zw = np.full(u.shape, np.nan, dtype=np.float32)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1
    ok = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)

    if np.any(ok):
        u0_ok = u0[ok]; v0_ok = v0[ok]
        u1_ok = u1[ok]; v1_ok = v1[ok]
        du = u[ok] - u0_ok
        dv = v[ok] - v0_ok

        Z00 = Z[v0_ok, u0_ok]; Z10 = Z[v0_ok, u1_ok]
        Z01 = Z[v1_ok, u0_ok]; Z11 = Z[v1_ok, u1_ok]

        w00 = (1.0 - du) * (1.0 - dv)
        w10 = du * (1.0 - dv)
        w01 = (1.0 - du) * dv
        w11 = du * dv

        Zw_ok = (w00 * Z00 + w10 * Z10 + w01 * Z01 + w11 * Z11).astype(np.float32)
        Zw[ok] = Zw_ok
        Zw[~valid] = np.nan  # XY invalid → Z도 무효

    return Xw, Yw, Zw, valid


def tris_img_to_bev_by_lut(tris_img: np.ndarray, lut_data: dict, bev_scale: float = 1.0,
                           min_valid_corners: int = 3, boundary_eps: float = 1e-3):
    """
    이미지 좌표의 삼각형들(tris_img: [N,3,2])을 LUT(npz)의 (X,Y,Z)로 보간해 BEV 평면으로 투영.
    returns:
        tris_bev_xy: (N,3,2)
        tris_bev_z:  (N,3)
        tri_ok:      (N,) bool
    """
    if tris_img.size == 0:
        return (np.zeros((0, 3, 2), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=bool))

    u = tris_img[:, :, 0].reshape(-1)
    v = tris_img[:, :, 1].reshape(-1)

    Xw, Yw, Zw, valid = _bilinear_lut_xyz(
        lut_data, u, v,
        min_valid_corners=min_valid_corners,
        boundary_eps=boundary_eps
    )

    valid = valid.reshape(-1, 3)
    tri_ok = np.all(valid, axis=1)

    Xw = Xw.reshape(-1, 3)
    Yw = Yw.reshape(-1, 3)
    Zw = Zw.reshape(-1, 3)

    tris_bev_xy = np.stack([Xw, Yw], axis=-1).astype(np.float32)
    tris_bev_xy *= float(bev_scale)

    return tris_bev_xy, Zw.astype(np.float32), tri_ok.astype(bool)


# =========================
# TensorRT temporal runner
# =========================
class TensorRTTemporalRunner:
    """
    TensorRT ConvLSTM/GRU engine runner.
    (Updated for TensorRT 8.5+ compatibility on Jetson Orin)
    """

    def __init__(self, engine_path: str,
                 state_stride_hint: int = 32,
                 default_hidden_ch: int = 256):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.state_stride_hint = int(state_stride_hint)
        self.default_hidden_ch = int(default_hidden_ch)

        # --- [수정됨] TensorRT 8.5+ 대응: num_bindings -> num_io_tensors ---
        # 최신 버전에서는 binding 인덱스 대신 tensor 이름을 주로 사용합니다.
        try:
            self.num_io_tensors = self.engine.num_io_tensors
            self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.num_io_tensors)]
        except AttributeError:
            # 구버전(8.4 이하) 호환용 백업
            self.num_io_tensors = self.engine.num_bindings
            self.tensor_names = [self.engine.get_binding_name(i) for i in range(self.num_io_tensors)]

        # 입력/출력 텐서 구분
        self.input_names = []
        self.output_names = []
        
        for name in self.tensor_names:
            # get_tensor_mode 혹은 binding_is_input 사용
            try:
                mode = self.engine.get_tensor_mode(name)
                is_input = (mode == trt.TensorIOMode.INPUT)
            except AttributeError:
                 is_input = self.engine.binding_is_input(self.engine.get_binding_index(name))
            
            if is_input:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # Semantic mapping
        cand_x = [n for n in self.input_names if n.lower() in ("images", "image", "input")]
        self.x_name = cand_x[0] if cand_x else self.input_names[0]

        self.h_name = next((n for n in self.input_names if "h_in" in n.lower()), None)
        self.c_name = next((n for n in self.input_names if "c_in" in n.lower()), None)

        self.ho_name = next((n for n in self.output_names if "h_out" in n.lower()), None)
        self.co_name = next((n for n in self.output_names if "c_out" in n.lower()), None)

        self.reg_names = [n for n in self.output_names if "reg" in n.lower()]
        self.obj_names = [n for n in self.output_names if "obj" in n.lower()]
        self.cls_names = [n for n in self.output_names if "cls" in n.lower()]

        def _sort_key(s):
            toks = []
            acc = ""
            for ch in s:
                if ch.isdigit():
                    acc += ch
                else:
                    if acc:
                        toks.append(int(acc))
                        acc = ""
                    toks.append(ch)
            if acc:
                toks.append(int(acc))
            return tuple(toks)

        self.reg_names.sort(key=_sort_key)
        self.obj_names.sort(key=_sort_key)
        self.cls_names.sort(key=_sort_key)

        # Host / device buffers
        self.host_buffers = {}
        self.device_buffers = {}
        self.buffer_nbytes = {}
        # execute_v2를 위한 바인딩 포인터 리스트 (이름 순서대로 정렬됨에 주의)
        self.bindings = [0] * self.num_io_tensors
        self.binding_dtypes = {}
        self.binding_shapes = {}
        self.stream = cuda.Stream()

        for i, name in enumerate(self.tensor_names):
            try:
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            except AttributeError:
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
            self.binding_dtypes[name] = dtype

        # Hidden state tracking (device-only buffers)
        self.state_shapes = {}
        self.state_reset_pending = set()
        self.state_pairs = []
        if self.h_name and self.ho_name:
            self.state_pairs.append((self.h_name, self.ho_name))
        if self.c_name and self.co_name:
            self.state_pairs.append((self.c_name, self.co_name))

        # Cached meta for state size
        self.h_shape_meta = self._shape_from_engine_meta(self.h_name)
        self.c_shape_meta = self._shape_from_engine_meta(self.c_name)

    def _shape_from_engine_meta(self, name):
        if name is None:
            return None
        try:
            dims = self.engine.get_tensor_shape(name)
        except AttributeError:
            idx = self.engine.get_binding_index(name)
            dims = self.engine.get_binding_shape(idx)

        def _to_int(val, default):
            return int(val) if (isinstance(val, int) and val > 0) else default
        N = _to_int(dims[0], 1)
        C = _to_int(dims[1], self.default_hidden_ch)
        Hs = _to_int(dims[2], 0)
        Ws = _to_int(dims[3], 0)
        return [N, C, Hs, Ws]

    def _zero_state_buffer(self, name):
        if name and name in self.device_buffers and name in self.buffer_nbytes:
            cuda.memset_d8(self.device_buffers[name], 0, self.buffer_nbytes[name])

    def reset(self):
        """Reset temporal hidden state."""
        for name in (self.h_name, self.c_name):
            if name:
                self.state_reset_pending.add(name)

    def _ensure_state(self, img_numpy_chw: np.ndarray):
        _, _, H, W = img_numpy_chw.shape
        for name, meta in (
            (self.h_name, self.h_shape_meta),
            (self.c_name, self.c_shape_meta),
        ):
            if name is None or name in self.state_shapes:
                continue
            N, C, Hs, Ws = meta
            if Hs == 0 or Ws == 0:
                Hs = max(1, H // self.state_stride_hint)
                Ws = max(1, W // self.state_stride_hint)
            shape = (N, C, Hs, Ws)
            self.state_shapes[name] = shape
            self.binding_shapes[name] = shape
            self.state_reset_pending.add(name)

    def _set_binding_shapes(self, img_numpy_chw: np.ndarray):
        """Set dynamic binding shapes using new API."""
        # 1. Input Image
        shape_x = tuple(img_numpy_chw.shape)
        try:
            self.context.set_input_shape(self.x_name, shape_x)
            self.binding_shapes[self.x_name] = tuple(self.context.get_tensor_shape(self.x_name))
        except AttributeError:
            # Old API fallback
            idx = self.engine.get_binding_index(self.x_name)
            self.context.set_binding_shape(idx, shape_x)
            self.binding_shapes[self.x_name] = tuple(self.context.get_binding_shape(idx))

        # 2. Hidden States
        if self.h_name and self.h_name in self.state_shapes:
            h_shape = self.state_shapes[self.h_name]
            try:
                self.context.set_input_shape(self.h_name, h_shape)
                self.binding_shapes[self.h_name] = tuple(self.context.get_tensor_shape(self.h_name))
            except AttributeError:
                idx = self.engine.get_binding_index(self.h_name)
                self.context.set_binding_shape(idx, h_shape)
                self.binding_shapes[self.h_name] = tuple(self.context.get_binding_shape(idx))

        if self.c_name and self.c_name in self.state_shapes:
            c_shape = self.state_shapes[self.c_name]
            try:
                self.context.set_input_shape(self.c_name, c_shape)
                self.binding_shapes[self.c_name] = tuple(self.context.get_tensor_shape(self.c_name))
            except AttributeError:
                idx = self.engine.get_binding_index(self.c_name)
                self.context.set_binding_shape(idx, c_shape)
                self.binding_shapes[self.c_name] = tuple(self.context.get_binding_shape(idx))

        # 3. Output Shapes
        for name in self.output_names:
            try:
                self.binding_shapes[name] = tuple(self.context.get_tensor_shape(name))
            except AttributeError:
                idx = self.engine.get_binding_index(name)
                self.binding_shapes[name] = tuple(self.context.get_binding_shape(idx))

    def _allocate_buffers_if_needed(self):
        for i, name in enumerate(self.tensor_names):
            if name in self.host_buffers:
                continue
                
            shape = self.binding_shapes.get(name, None)

            # shape 정보가 없으면 가져옴
            if shape is None:
                try:
                    shape = tuple(self.context.get_tensor_shape(name))
                except AttributeError:
                    shape = tuple(self.context.get_binding_shape(i))

            if any(d < 0 for d in shape):
                continue

            size = int(np.prod(shape))
            dtype = self.binding_dtypes[name]
            itemsize = np.dtype(dtype).itemsize
            size_bytes = size * itemsize

            prev_nbytes = self.buffer_nbytes.get(name)
            if name in self.device_buffers and prev_nbytes == size_bytes:
                self.bindings[i] = int(self.device_buffers[name])
                continue

            needs_host = name not in (self.h_name, self.c_name, self.ho_name, self.co_name)
            host_buf = cuda.pagelocked_empty(size, dtype=dtype) if needs_host else None
            device_buf = cuda.mem_alloc(size_bytes)

            self.host_buffers[name] = host_buf
            self.device_buffers[name] = device_buf
            self.buffer_nbytes[name] = size_bytes
            self.bindings[i] = int(device_buf)

            if name in self.state_reset_pending:
                self._zero_state_buffer(name)
                self.state_reset_pending.discard(name)

    def _apply_pending_state_resets(self):
        if not self.state_reset_pending:
            return
        for name in list(self.state_reset_pending):
            self._zero_state_buffer(name)
            self.state_reset_pending.discard(name)

    def forward(self, img_numpy_chw: np.ndarray):
        assert img_numpy_chw.ndim == 4, "Input image must be (1,3,H,W)"

        self._ensure_state(img_numpy_chw)
        self._set_binding_shapes(img_numpy_chw)
        self._allocate_buffers_if_needed()
        self._apply_pending_state_resets()

        # Copy Inputs to Host
        x_host = self.host_buffers[self.x_name]
        x_host[...] = img_numpy_chw.ravel()

        # HtoD (Host -> Device)
        stream = self.stream
        for name in self.input_names:
            host_buf = self.host_buffers.get(name)
            if host_buf is None or name not in self.device_buffers:
                continue
            cuda.memcpy_htod_async(self.device_buffers[name], host_buf, stream)

        # Execute
        if hasattr(self.context, "execute_async_v2"):
            self.context.execute_async_v2(self.bindings, stream.handle)
        else:
            # ensure pending HtoD copies on our stream finish before sync exec
            stream.synchronize()
            self.context.execute_v2(self.bindings)

        # DtoH (Device -> Host) & Map Outputs
        out_map = {}
        for name in self.output_names:
            host_buf = self.host_buffers.get(name)
            if host_buf is None or name not in self.device_buffers:
                continue
            cuda.memcpy_dtoh_async(host_buf, self.device_buffers[name], stream)

        stream.synchronize()

        for name in self.output_names:
            host_buf = self.host_buffers.get(name)
            if host_buf is None:
                continue
            shape = self.binding_shapes[name]
            out_map[name] = host_buf.reshape(shape)

        # Update State on device (no host round-trip)
        for in_name, out_name in self.state_pairs:
            if in_name not in self.device_buffers or out_name not in self.device_buffers:
                continue
            nbytes = self.buffer_nbytes.get(out_name)
            if nbytes is None:
                shape = self.binding_shapes.get(out_name)
                if shape is None:
                    continue
                dtype = self.binding_dtypes[out_name]
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
                self.buffer_nbytes[out_name] = nbytes
            cuda.memcpy_dtod(self.device_buffers[in_name], self.device_buffers[out_name], nbytes)

        # Result formatting
        pred_list = []
        for rn, on, cn in zip(self.reg_names, self.obj_names, self.cls_names):
            pr = torch.from_numpy(out_map[rn])
            po = torch.from_numpy(out_map[on])
            pc = torch.from_numpy(out_map[cn])
            pred_list.append((pr, po, pc))
        return pred_list


# =========================
# 시퀀스 키
# =========================
def seq_key(file_path: str, mode: str) -> str:
    p = Path(file_path)
    if mode == "by_subdir":
        return p.parent.name
    stem = p.stem
    if "_" in stem:
        return stem.split("_")[0]
    if "-" in stem:
        return stem.split("-")[0]
    return "ALL"


def _sane_dims(L, W, args) -> bool:
    if not (np.isfinite(L) and np.isfinite(W)):
        return False
    if not (args.min_length <= L <= args.max_length):
        return False
    if not (args.min_width <= W <= args.max_width):
        return False
    r = L / max(W, 1e-6)
    if not (args.min_lw_ratio <= r <= args.max_lw_ratio):
        return False
    return True


# =========================
# 메인
# =========================
def main():
    ap = argparse.ArgumentParser("YOLO11 2.5D TensorRT Temporal Inference (+GT & BEV via LUT)")
    ap.add_argument("--input-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default="./inference_results_npz")
    ap.add_argument("--weights", type=str, required=True,
                    help="TensorRT engine (.engine) path")
    ap.add_argument("--img-size", type=str, default="864,1536")
    ap.add_argument("--score-mode", type=str, default="obj*cls",
                    choices=["obj", "cls", "obj*cls"])
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms-iou", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--contain-thr", type=float, default=0.85)
    ap.add_argument("--clip-cells", type=float, default=None)
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png")
    ap.add_argument("--save-images", action="store_true",
                    help="Write per-image prediction overlays (default: off)")
    ap.add_argument("--save-labels", action="store_true",
                    help="Write per-image prediction txt files (default: off)")
    ap.add_argument("--save-mixed", action="store_true",
                    help="Write prediction-vs-GT overlay images (default: off)")
    ap.add_argument("--save-bev", action="store_true",
                    help="Write BEV visualization/labels (requires --lut-path)")
    ap.add_argument("--prefetch-threads", type=int, default=1,
                    help="Background threads for image loading/resizing (0=disabled)")

    # strides for decoder
    ap.add_argument("--strides", type=str, default="8,16,32")

    # temporal & sequence (keep flags for compatibility)
    ap.add_argument("--temporal", type=str, default="lstm", choices=["none", "gru", "lstm"])
    ap.add_argument("--seq-mode", type=str, default="by_prefix", choices=["by_prefix", "by_subdir"])
    ap.add_argument("--reset-per-seq", action="store_true", default=True)

    # state feature map hint
    ap.add_argument("--state-stride-hint", type=int, default=32)
    ap.add_argument("--default-hidden-ch", type=int, default=256)

    # Eval(2D)
    ap.add_argument("--gt-label-dir", type=str, default=None)
    ap.add_argument("--eval-iou-thr", type=float, default=0.5)
    ap.add_argument("--labels-are-original-size", action="store_true", default=True)

    # BEV via LUT
    ap.add_argument(
        "--lut-path",
        type=str,
        default=None,
        help="pixel2world_lut.npz 경로 (지정하면 BEV 라벨/시각화 수행)",
    )
    ap.add_argument("--bev-scale", type=float, default=1.0)

    # LUT interpolation robustness
    ap.add_argument("--lut-min-corners", type=int, default=3,
                    help="Min number of valid bilinear corners (0..4) to accept a sample (default: 3)")
    ap.add_argument("--lut-boundary-eps", type=float, default=1e-3,
                    help="Clamp (u,v) to [0,W-1-eps]/[0,H-1-eps] (default: 1e-3)")

    # BEV 3D label options
    ap.add_argument("--bev-label-3d", action="store_true", default=True,
                    help="Write BEV labels with cz, yaw, pitch, roll (default: on)")
    ap.add_argument("--use-roll", action="store_true", default=False,
                    help="Also estimate & write roll from left-right Z difference")
    ap.add_argument("--roll-threshold-deg", type=float, default=2.0,
                    help="Absolute roll below this (deg) is snapped to 0")
    ap.add_argument("--roll-clamp-deg", type=float, default=8.0,
                    help="Clamp |roll| to this maximum (deg)")
    ap.add_argument("--pitch-clamp-deg", type=float, default=30.0,
                    help="Clamp |pitch| to this maximum (deg)")

    # --- sanity filters for 3D dims ---
    ap.add_argument("--min-length", type=float, default=0.0)
    ap.add_argument("--max-length", type=float, default=100.0)
    ap.add_argument("--min-width", type=float, default=0.0)
    ap.add_argument("--max-width", type=float, default=100.0)
    ap.add_argument("--min-lw-ratio", type=float, default=0.01)
    ap.add_argument("--max-lw-ratio", type=float, default=100)

    # kept for compatibility; not used in TensorRT path
    ap.add_argument("--no-cuda", action="store_true", help="(ignored for TensorRT)")

    args = ap.parse_args()
    H, W = map(int, args.img_size.split(","))
    strides = [float(s) for s in args.strides.split(",")]

    # TensorRT engine runner
    runner = TensorRTTemporalRunner(
        args.weights,
        state_stride_hint=args.state_stride_hint,
        default_hidden_ch=args.default_hidden_ch
    )

    out_img_dir = out_lab_dir = out_mix_dir = None
    if args.save_images:
        out_img_dir = os.path.join(args.output_dir, "images")
        os.makedirs(out_img_dir, exist_ok=True)
    if args.save_labels:
        out_lab_dir = os.path.join(args.output_dir, "labels")
        os.makedirs(out_lab_dir, exist_ok=True)
    if args.save_mixed:
        out_mix_dir = os.path.join(args.output_dir, "images_with_gt")
        os.makedirs(out_mix_dir, exist_ok=True)

    # LUT load (optional)
    use_bev = bool(args.lut_path)
    lut_data = None
    if use_bev:
        if not os.path.isfile(args.lut_path):
            raise FileNotFoundError(f"LUT not found: {args.lut_path}")
        lut_data = dict(np.load(args.lut_path))
        print(f"[BEV] Using LUT: {args.lut_path}")
    else:
        print("[BEV] LUT path not provided; skipping BEV label/visualization outputs.")

    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    names = [f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)]
    names.sort()
    target_hw = (H, W)

    prefetch_threads = max(0, int(args.prefetch_threads))
    executor = ThreadPoolExecutor(max_workers=prefetch_threads) if prefetch_threads > 0 else None
    use_prefetch = executor is not None
    next_future: Optional[Future] = None
    if use_prefetch and names:
        first_path = os.path.join(args.input_dir, names[0])
        next_future = executor.submit(_load_and_preprocess_image, first_path, target_hw)

    do_eval_2d = args.gt_label_dir is not None and os.path.isdir(args.gt_label_dir)

    metric_records = []
    total_gt = 0

    # BEV metrics
    metric_records_bev = []
    total_gt_bev = 0
    out_bev_img_dir = None
    out_bev_mix_dir = None
    out_bev_lab_dir = None
    if use_bev and args.save_bev:
        out_bev_img_dir = os.path.join(args.output_dir, "bev_images")
        out_bev_mix_dir = os.path.join(args.output_dir, "bev_images_with_gt")
        out_bev_lab_dir = os.path.join(args.output_dir, "bev_labels")
        os.makedirs(out_bev_img_dir, exist_ok=True)
        os.makedirs(out_bev_mix_dir, exist_ok=True)
        os.makedirs(out_bev_lab_dir, exist_ok=True)

    print(
        f"[Infer-TRT] imgs={len(names)}, temporal={args.temporal}, "
        f"seq={args.seq_mode}, reset_per_seq={args.reset_per_seq}, "
        f"eval2D={do_eval_2d}, use_bev={use_bev}"
    )

    prev_key = None
    for idx, name in enumerate(tqdm(names, desc="[Infer-TRT]")):
        path = os.path.join(args.input_dir, name)

        # Sequence boundary → reset temporal state
        k = seq_key(path, args.seq_mode)
        if args.reset_per_seq and k != prev_key:
            runner.reset()
        prev_key = k

        if use_prefetch:
            if next_future is None:
                break
            img_data = next_future.result()
            if idx + 1 < len(names):
                next_path = os.path.join(args.input_dir, names[idx + 1])
                next_future = executor.submit(_load_and_preprocess_image, next_path, target_hw)
            else:
                next_future = None
        else:
            img_data = _load_and_preprocess_image(path, target_hw)

        if img_data is None:
            continue

        img_bgr = img_data["img_bgr"]
        img_np = img_data["img_np"]
        H0 = img_data["H0"]
        W0 = img_data["W0"]
        scale_resize_x = img_data["scale_resize_x"]
        scale_resize_y = img_data["scale_resize_y"]
        scale_to_orig_x = img_data["scale_to_orig_x"]
        scale_to_orig_y = img_data["scale_to_orig_y"]

        # TensorRT forward
        outs = runner.forward(img_np)

        # decode predictions
        dets = decode_predictions(
            outs, strides,
            clip_cells=args.clip_cells,
            conf_th=args.conf,
            nms_iou=args.nms_iou,
            topk=args.topk,
            contain_thr=args.contain_thr,
            score_mode=args.score_mode,
            use_gpu_nms=True
        )[0]

        dets = tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)

        save_img = os.path.join(out_img_dir, name) if out_img_dir else None
        save_txt = (
            os.path.join(out_lab_dir, os.path.splitext(name)[0] + ".txt")
            if out_lab_dir
            else None
        )
        pred_tris_orig = draw_pred_only(img_bgr, dets, save_img, save_txt, W, H, W0, H0)

        # 2D Eval
        gt_for_eval = np.zeros((0, 3, 2), dtype=np.float32)
        gt_tri_orig_for_bev = np.zeros((0, 3, 2), dtype=np.float32)

        if do_eval_2d:
            lab_path = os.path.join(args.gt_label_dir, os.path.splitext(name)[0] + ".txt")
            gt_tri_raw = load_gt_triangles(lab_path)
            if gt_tri_raw.shape[0] > 0:
                if args.labels_are_original_size:
                    gt_tri_orig_for_bev = gt_tri_raw.astype(np.float32)
                    gt_for_eval = gt_tri_raw.copy()
                    gt_for_eval[:, :, 0] *= scale_resize_x
                    gt_for_eval[:, :, 1] *= scale_resize_y
                else:
                    gt_for_eval = gt_tri_raw.astype(np.float32)
                    gt_tri_orig_for_bev = gt_tri_raw.copy()
                    gt_tri_orig_for_bev[:, :, 0] *= scale_to_orig_x
                    gt_tri_orig_for_bev[:, :, 1] *= scale_to_orig_y

            save_img_mix = os.path.join(out_mix_dir, name) if out_mix_dir else None
            if save_img_mix is not None:
                draw_pred_with_gt(img_bgr, dets, gt_for_eval,
                                  save_img_mix, iou_thr=args.eval_iou_thr)

            if gt_for_eval.shape[0] > 0:
                records, _ = evaluate_single_image(dets, gt_for_eval,
                                                   iou_thr=args.eval_iou_thr)
                metric_records.extend(records)
                total_gt += gt_for_eval.shape[0]

        # ---- BEV (LUT) ----
        if use_bev:
            bev_dets = []
            if pred_tris_orig:
                pred_stack_orig = np.asarray(pred_tris_orig, dtype=np.float64)
                pred_tris_bev_xy, pred_tris_bev_z, good_mask = tris_img_to_bev_by_lut(
                    pred_stack_orig, lut_data, bev_scale=float(args.bev_scale),
                    min_valid_corners=int(args.lut_min_corners),
                    boundary_eps=float(args.lut_boundary_eps)
                )
                num_tri = min(len(dets), pred_tris_bev_xy.shape[0])
                if len(dets) != pred_tris_bev_xy.shape[0]:
                    print(
                        f"[WARN] BEV LUT triangles ({pred_tris_bev_xy.shape[0]}) "
                        f"and detections ({len(dets)}) differ; clipping to {num_tri}."
                    )
                for idx in range(num_tri):
                    det = dets[idx]
                    if not good_mask[idx]:
                        continue
                    tri_bev_xy = pred_tris_bev_xy[idx]
                    tri_bev_z = pred_tris_bev_z[idx]
                    if not np.all(np.isfinite(tri_bev_xy)):
                        continue
                    poly_bev = poly_from_tri(tri_bev_xy)
                    props = None
                    try:
                        props = compute_bev_properties_3d(
                            tri_bev_xy,
                            tri_bev_z,
                            pitch_clamp_deg=args.pitch_clamp_deg,
                            use_roll=args.use_roll,
                            roll_threshold_deg=args.roll_threshold_deg,
                            roll_clamp_deg=args.roll_clamp_deg,
                        )
                    except Exception:
                        props = None

                    if props is None:
                        props = compute_bev_properties(
                            tri_bev_xy,
                            tri_bev_z,
                            pitch_clamp_deg=args.pitch_clamp_deg,
                            use_roll=args.use_roll,
                            roll_threshold_deg=args.roll_threshold_deg,
                            roll_clamp_deg=args.roll_clamp_deg,
                        )
                    if props is None:
                        continue

                    center, length, width, yaw, front_edge, cz, pitch_deg, roll_deg = props

                    if not _sane_dims(length, width, args):
                        continue

                    bev_dets.append(
                        {
                            "score": float(det["score"]),
                            "tri": tri_bev_xy,
                            "poly": poly_bev,
                            "center": center,
                            "length": length,
                            "width": width,
                            "yaw": yaw,
                            "front_edge": front_edge,
                            "cz": cz,
                            "pitch": pitch_deg,
                            "roll": roll_deg if args.use_roll else 0.0,
                        }
                    )

            # GT → BEV (same LUT)
            if gt_tri_orig_for_bev.size > 0:
                gt_u = gt_tri_orig_for_bev[:, :, 0].reshape(-1)
                gt_v = gt_tri_orig_for_bev[:, :, 1].reshape(-1)
                Xg, Yg, Zg, Vg = _bilinear_lut_xyz(
                    lut_data,
                    gt_u,
                    gt_v,
                    min_valid_corners=int(args.lut_min_corners),
                    boundary_eps=float(args.lut_boundary_eps),
                )
                Vg = Vg.reshape(-1, 3)
                good_gt = np.all(Vg, axis=1)
                Xg = Xg.reshape(-1, 3)
                Yg = Yg.reshape(-1, 3)
                gt_tris_bev = np.stack([Xg, Yg], axis=-1).astype(np.float32)
                gt_tris_bev *= float(args.bev_scale)
                gt_tris_bev = gt_tris_bev[good_gt]
            else:
                gt_tris_bev = np.zeros((0, 3, 2), dtype=np.float32)

            # BEV visualization / labels
            if args.save_bev and out_bev_img_dir:
                bev_img_path = os.path.join(out_bev_img_dir, name)
                draw_bev_visualization(bev_dets, None,
                                       bev_img_path, f"{name} | Pred BEV")

            if args.save_bev and out_bev_mix_dir:
                bev_mix_path = os.path.join(out_bev_mix_dir, name)
                draw_bev_visualization(bev_dets, gt_tris_bev,
                                       bev_mix_path, f"{name} | Pred & GT BEV")

            if args.save_bev and out_bev_lab_dir:
                bev_label_path = os.path.join(
                    out_bev_lab_dir, os.path.splitext(name)[0] + ".txt"
                )
                write_bev_labels(bev_label_path, bev_dets,
                                 write_3d=bool(args.bev_label_3d))

            total_gt_bev += gt_tris_bev.shape[0]
            if gt_tris_bev.shape[0] > 0 and len(bev_dets) > 0:
                records_bev, _ = evaluate_single_image_bev(
                    bev_dets, gt_tris_bev, iou_thr=args.eval_iou_thr
                )
                metric_records_bev.extend(records_bev)

    if executor:
        executor.shutdown(wait=True)

    # ---- 전체 메트릭 출력 ----
    if do_eval_2d:
        metrics = compute_detection_metrics(metric_records, total_gt)
        print("== 2D Eval (dataset-wide) ==")
        print("Precision:  {:.4f}".format(metrics["precision"]))
        print("Recall:     {:.4f}".format(metrics["recall"]))
        print("mAP@50:     {:.4f}".format(metrics["map50"]))
        print("mAOE(deg):  {:.2f}".format(metrics["mAOE_deg"]))

    if use_bev:
        metrics_bev = (
            compute_detection_metrics(metric_records_bev, total_gt_bev)
            if (total_gt_bev > 0 or metric_records_bev)
            else None
        )
        if metrics_bev is not None:
            print("== BEV Eval (dataset-wide) ==")
            print("Precision:  {:.4f}".format(metrics_bev["precision"]))
            print("Recall:     {:.4f}".format(metrics_bev["recall"]))
            print("APbev@50:   {:.4f}".format(metrics_bev["map50"]))
            print("mAOE_bev:   {:.2f}".format(metrics_bev["mAOE_deg"]))
        else:
            print("[Info] BEV 평가는 GT 또는 유효 매칭이 부족해 계산하지 않았습니다.")
    else:
        print("[Info] LUT 미지정으로 BEV 라벨/시각화/평가를 생략했습니다.")

    print("Done.")


if __name__ == "__main__":
    main()
