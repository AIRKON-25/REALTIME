#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class PairPoints:
    cam_pts: List[Tuple[float, float]]
    map_pts: List[Tuple[float, float]]

    def as_arrays(self):
        cam = np.array(self.cam_pts, dtype=np.float64).reshape(-1, 2)
        mp  = np.array(self.map_pts, dtype=np.float64).reshape(-1, 2)
        return cam, mp

    def pop_last(self):
        if len(self.cam_pts) > len(self.map_pts):
            self.cam_pts.pop()
            return
        if len(self.cam_pts) == len(self.map_pts) and len(self.cam_pts) > 0:
            self.cam_pts.pop()
            self.map_pts.pop()

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_map2world_json(path: str):
    """
    기대 JSON 형식:
    {
      "points": [
        { "image_px": {"u":..., "v":...}, "world_xy": {"x":..., "y":...}},
        ...
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = data.get("points", [])
    if len(pts) < 4:
        raise ValueError("map2world.json의 points는 최소 4개가 필요합니다.")
    img_uv = []
    world_xy = []
    for p in pts[:4]:
        img_uv.append([float(p["image_px"]["u"]), float(p["image_px"]["v"])])
        world_xy.append([float(p["world_xy"]["x"]), float(p["world_xy"]["y"])])
    img_uv = np.array(img_uv, dtype=np.float64)
    world_xy = np.array(world_xy, dtype=np.float64)
    return img_uv, world_xy, data

def homography_from_4pts(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """
    src_xy: Nx2 (N>=4)
    dst_xy: Nx2
    """
    if src_xy.shape[0] < 4:
        raise ValueError("Homography에는 최소 4점이 필요합니다.")
    H, mask = cv2.findHomography(src_xy, dst_xy, method=0)
    if H is None:
        raise RuntimeError("4점 기반 Homography 계산 실패")
    return H

def reprojection_errors(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    H: 3x3, src_pts: Nx2, dst_pts: Nx2
    return: N errors (pixel distance)
    """
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=np.float64)])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-12, None)
    err = np.linalg.norm(proj - dst_pts, axis=1)
    return err

POINT_RADIUS = 12
POINT_THICKNESS = 3
SELECT_TOL = 20
MAG_POINT_RADIUS = 12

def draw_points_and_lines(canvas, cam_rect, map_rect, pairs: PairPoints, s_cam: float, s_map: float, pending_cam_pt=None):
    """
    cam_rect: (x0,y0,x1,y1) area in canvas
    map_rect: (x0,y0,x1,y1) area in canvas
    """
    x0c, y0c, x1c, y1c = cam_rect
    x0m, y0m, x1m, y1m = map_rect

    n_pairs = min(len(pairs.cam_pts), len(pairs.map_pts))

    # Draw completed pairs
    for i in range(n_pairs):
        (uc, vc) = pairs.cam_pts[i]
        (um, vm) = pairs.map_pts[i]
        pc = (int(x0c + uc * s_cam), int(y0c + vc * s_cam))
        pm = (int(x0m + um * s_map), int(y0m + vm * s_map))
        cv2.circle(canvas, pc, POINT_RADIUS, (255, 0, 255), POINT_THICKNESS)
        cv2.circle(canvas, pm, POINT_RADIUS, (255, 0, 255), POINT_THICKNESS)
        cv2.line(canvas, pc, pm, (50, 200, 50), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(i+1), (pc[0]+6, pc[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(i+1), (pm[0]+6, pm[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # Draw unmatched camera points (no map yet)
    for j in range(n_pairs, len(pairs.cam_pts)):
        uc, vc = pairs.cam_pts[j]
        pc = (int(x0c + uc * s_cam), int(y0c + vc * s_cam))
        cv2.circle(canvas, pc, POINT_RADIUS+1, (0, 0, 255), POINT_THICKNESS)
        cv2.putText(canvas, "C", (pc[0]+8, pc[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1, cv2.LINE_AA)

    # Draw unmatched map points (should rarely happen)
    for j in range(n_pairs, len(pairs.map_pts)):
        um, vm = pairs.map_pts[j]
        pm = (int(x0m + um * s_map), int(y0m + vm * s_map))
        cv2.circle(canvas, pm, POINT_RADIUS+1, (255, 0, 0), POINT_THICKNESS)
        cv2.putText(canvas, "M", (pm[0]+8, pm[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 1, cv2.LINE_AA)

    # Draw pending cam point if exists
    if pending_cam_pt is not None:
        (uc, vc) = pending_cam_pt
        pc = (int(x0c + uc * s_cam), int(y0c + vc * s_cam))
        cv2.circle(canvas, pc, POINT_RADIUS+4, (0, 128, 255), 2)
        cv2.putText(canvas, "pending -> click on MAP", (x0c+10, y0c+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2, cv2.LINE_AA)

def warp_and_overlay(cam_bgr, map_bgr, H_cam2map, alpha=0.55, margin=0):
    """
    cam을 map 좌표로 warp하고 map 위에 overlay
    """
    margin = int(max(0, margin))
    if margin > 0:
        pad_map = cv2.copyMakeBorder(
            map_bgr, margin, margin, margin, margin,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        T = np.array([[1.0, 0.0, margin],
                      [0.0, 1.0, margin],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        H_use = T @ H_cam2map
    else:
        pad_map = map_bgr
        H_use = H_cam2map
    mh, mw = pad_map.shape[:2]
    warped = cv2.warpPerspective(cam_bgr, H_use, (mw, mh), flags=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(pad_map, 1.0 - alpha, warped, alpha, 0.0)
    return warped, overlay

# -----------------------------
# Save homography as JSON
# -----------------------------
def save_H_json(path, H: np.ndarray):
    """
    요청 포맷:
    {
      "H": [
        [..,..,..],
        [..,..,..],
        [..,..,..]
      ]
    }
    """
    data = {"H": [[float(v) for v in row] for row in H.tolist()]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# Delta-H builder (GUI micro-adjust)
# -----------------------------
def make_deltaH(tx: float, ty: float, theta_deg: float, scale: float) -> np.ndarray:
    """
    ΔH: map 픽셀 좌표계에서의 2D similarity transform
      [ sR  t ]
      [ 0   1 ]
    - tx, ty: 픽셀 단위 평행이동
    - theta_deg: 회전(deg)
    - scale: 스케일(1.0 근처)
    """
    th = np.deg2rad(theta_deg)
    c = np.cos(th) * float(scale)
    s = np.sin(th) * float(scale)
    H = np.array([[c, -s, float(tx)],
                  [s,  c, float(ty)],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return H

# -----------------------------
# Main GUI tool
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True, help="camera image path")
    ap.add_argument("--map", default="utils/make_H/ces_real_map.png" ,help="global map image path")
    ap.add_argument("--map2world", default="utils/make_H/ces_real_img2world.json", help="json with 4 corners: map image px -> world xy")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ransac-th", type=float, default=3.0, help="RANSAC reprojection threshold in pixels")
    ap.add_argument("--alpha", type=float, default=0.55, help="overlay alpha for warped cam on map")
    ap.add_argument("--overlay-margin", type=int, default=200, help="margin (pixels) added around map for overlay/preview")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    cam = cv2.imread(args.cam, cv2.IMREAD_COLOR)
    mp  = cv2.imread(args.map, cv2.IMREAD_COLOR)
    if cam is None:
        raise FileNotFoundError(args.cam)
    if mp is None:
        raise FileNotFoundError(args.map)

    cam_h, cam_w = cam.shape[:2]
    map_h, map_w = mp.shape[:2]

    # Build side-by-side canvas (same height)
    target_h = max(cam_h, map_h)

    # resize each image to target_h while keeping aspect ratio for display
    def resize_to_h(img, th):
        h, w = img.shape[:2]
        if h == th:
            return img, 1.0
        scale = th / float(h)
        out = cv2.resize(img, (int(round(w*scale)), th), interpolation=cv2.INTER_AREA)
        return out, scale

    cam_disp, s_cam = resize_to_h(cam, target_h)
    map_disp, s_map = resize_to_h(mp, target_h)

    # =============================
    # Magnifier/drag state variables
    # =============================
    MAG_SIZE = 41        # 원본 ROI 크기 (홀수)
    MAG_SCALE = 8        # 확대 배율
    hover_cam_uv = None
    hover_map_uv = None
    dragging = False
    drag_target = None   # ("cam", idx) or ("map", idx)
    last_mouse_xy = None
    last_mouse_xy_raw = None

    # Drag sensitivity state
    drag_sensitivity = 1.0   # 1.0=normal, 0.25=fine, 0.1=ultra-fine
    drag_mode_name = "NORMAL"

    mag_mode = None   # None / "cam" / "map"

    gap = 8
    canvas_w = cam_disp.shape[1] + gap + map_disp.shape[1]
    canvas_h = target_h + 140  # HUD area

    # Rects for click mapping in canvas coordinates
    cam_rect = (0, 0, cam_disp.shape[1], target_h)
    map_rect = (cam_disp.shape[1] + gap, 0, cam_disp.shape[1] + gap + map_disp.shape[1], target_h)

    # Pre-build static canvas (images + borders) to avoid copying every frame
    base_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    base_canvas[0:target_h, 0:cam_disp.shape[1]] = cam_disp
    base_canvas[0:target_h, map_rect[0]:map_rect[0] + map_disp.shape[1]] = map_disp
    cv2.rectangle(base_canvas, (cam_rect[0], cam_rect[1]), (cam_rect[2]-1, cam_rect[3]-1), (80,80,80), 1)
    cv2.rectangle(base_canvas, (map_rect[0], map_rect[1]), (map_rect[2]-1, map_rect[3]-1), (80,80,80), 1)
    cv2.line(base_canvas, (map_rect[0]-gap//2, 0), (map_rect[0]-gap//2, target_h), (80,80,80), 1)

    pairs = PairPoints(cam_pts=[], map_pts=[])
    pending_cam = None  # store cam click until map click

    # -----------------------------
    # Homography states
    # -----------------------------
    H_base: Optional[np.ndarray] = None     # 점 기반으로 계산된 초기 H (cam->map)
    H_current: Optional[np.ndarray] = None  # 사용자 적용(누적)된 H (cam->map)
    H_prev_stack: List[np.ndarray] = []     # ENTER 적용 전 H_current 저장 스택

    # micro adjust params
    adj_tx = 0.0
    adj_ty = 0.0
    adj_theta = 0.0
    adj_scale = 1.0

    inlier_mask = None
    overlay_live_cache = None
    last_overlay_disp = None

    # For mapping display coords -> original coords
    def canvas_to_cam_uv(x, y):
        # CAM 영역 좌상단 기준으로 변환
        u = (x - cam_rect[0]) / s_cam
        v = (y - cam_rect[1]) / s_cam
        return float(u), float(v)

    def canvas_to_map_uv(x, y):
        # MAP 영역 좌상단 기준으로 변환
        u = (x - map_rect[0]) / s_map
        v = (y - map_rect[1]) / s_map
        return float(u), float(v)

    win = "Homography Tool (CAM | MAP)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, canvas_w), min(980, canvas_h))

    def on_mouse(event, x, y, flags, userdata):
        nonlocal pending_cam, pairs
        nonlocal hover_cam_uv, hover_map_uv, dragging, drag_target, last_mouse_xy, last_mouse_xy_raw
        # Mouse move: update hover position for magnifier, or drag point if dragging
        if event == cv2.EVENT_MOUSEMOVE:
            if dragging and last_mouse_xy is not None:
                if last_mouse_xy_raw is None:
                    last_mouse_xy_raw = (x, y)
                raw_dx = x - last_mouse_xy_raw[0]
                raw_dy = y - last_mouse_xy_raw[1]

                dx = raw_dx * drag_sensitivity
                dy = raw_dy * drag_sensitivity

                side, idx = drag_target if drag_target is not None else (None, None)

                if side == "cam":
                    u, v = pairs.cam_pts[idx]
                    pairs.cam_pts[idx] = (
                        u + dx / s_cam,
                        v + dy / s_cam
                    )
                    hover_cam_uv = pairs.cam_pts[idx]
                    hover_map_uv = None
                elif side == "map":
                    u, v = pairs.map_pts[idx]
                    pairs.map_pts[idx] = (
                        u + dx / s_map,
                        v + dy / s_map
                    )
                    hover_map_uv = pairs.map_pts[idx]
                    hover_cam_uv = None

                # 기준점 업데이트
                last_mouse_xy = (last_mouse_xy[0] + dx, last_mouse_xy[1] + dy)
                last_mouse_xy_raw = (x, y)
                return
            # Not dragging: update hover for magnifier
            if cam_rect[0] <= x < cam_rect[2] and y < target_h:
                hover_cam_uv = canvas_to_cam_uv(x, y)
                hover_map_uv = None
            elif map_rect[0] <= x < map_rect[2] and y < target_h:
                hover_map_uv = canvas_to_map_uv(x, y)
                hover_cam_uv = None
            else:
                hover_cam_uv = None
                hover_map_uv = None
            return
        # Mouse up: stop dragging
        if event == cv2.EVENT_LBUTTONUP:
            dragging = False
            drag_target = None
            last_mouse_xy = None
            last_mouse_xy_raw = None
            return
        # Only clicks valid in image area (not HUD)
        if y < 0 or y >= target_h:
            return
        # Mouse down: check for drag on existing points
        if event == cv2.EVENT_LBUTTONDOWN:
            # Try cam points
            for i, (u, v) in enumerate(pairs.cam_pts):
                px = int(cam_rect[0] + u * s_cam)
                py = int(cam_rect[1] + v * s_cam)
                if abs(px - x) < SELECT_TOL and abs(py - y) < SELECT_TOL:
                    dragging = True
                    drag_target = ("cam", i)
                    last_mouse_xy = (x, y)
                    last_mouse_xy_raw = (x, y)
                    hover_cam_uv = pairs.cam_pts[i]
                    hover_map_uv = None
                    return
            # Try map points
            for i, (u, v) in enumerate(pairs.map_pts):
                px = int(map_rect[0] + u * s_map)
                py = int(map_rect[1] + v * s_map)
                if abs(px - x) < SELECT_TOL and abs(py - y) < SELECT_TOL:
                    dragging = True
                    drag_target = ("map", i)
                    last_mouse_xy = (x, y)
                    last_mouse_xy_raw = (x, y)
                    hover_map_uv = pairs.map_pts[i]
                    hover_cam_uv = None
                    return
            # Not on existing point: continue with normal click logic
            if cam_rect[0] <= x < cam_rect[2]:
                u, v = canvas_to_cam_uv(x, y)
                if pending_cam is not None:
                    # replace the existing unmatched CAM point (only one allowed)
                    idx = len(pairs.map_pts)
                    if idx < len(pairs.cam_pts):
                        pairs.cam_pts[idx] = (u, v)
                    else:
                        pairs.cam_pts.append((u, v))
                else:
                    pairs.cam_pts.append((u, v))
                pending_cam = (u, v)
                return
            if map_rect[0] <= x < map_rect[2]:
                if pending_cam is None:
                    return
                u, v = canvas_to_map_uv(x, y)
                pairs.map_pts.append((u, v))
                pending_cam = None
                return

    cv2.setMouseCallback(win, on_mouse)

    # -----------------------------
    # Magnifier draw function
    # -----------------------------
    def draw_magnifier(img, u, v, pts=None, color=(0, 255, 255), name="magnifier"):
        h, w = img.shape[:2]
        r = MAG_SIZE // 2
        cx, cy = int(round(u)), int(round(v))
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        patch = img[y0:y1, x0:x1].copy()
        if patch.size == 0:
            return
        mag = cv2.resize(patch, None, fx=MAG_SCALE, fy=MAG_SCALE, interpolation=cv2.INTER_NEAREST)
        mh, mw = mag.shape[:2]
        cv2.line(mag, (mw // 2, 0), (mw // 2, mh), (0, 255, 255), 1)
        cv2.line(mag, (0, mh // 2), (mw, mh // 2), (0, 255, 255), 1)
        if pts:
            for px, py in pts:
                if x0 <= px < x1 and y0 <= py < y1:
                    mx = int(round((px - x0) * MAG_SCALE))
                    my = int(round((py - y0) * MAG_SCALE))
                    cv2.circle(mag, (mx, my), MAG_POINT_RADIUS, color, 2)
        cv2.imshow(name, mag)

    def compute_H_points():
        nonlocal inlier_mask
        cam_pts, map_pts = pairs.as_arrays()
        n = min(len(cam_pts), len(map_pts))
        if n < 4:
            print("[WARN] Homography 계산에는 최소 4쌍이 필요합니다.")
            return None
        cam_pts = cam_pts[:n]
        map_pts = map_pts[:n]

        H, mask = cv2.findHomography(cam_pts, map_pts, method=cv2.RANSAC, ransacReprojThreshold=float(args.ransac_th))
        if H is None:
            print("[ERROR] findHomography 실패")
            return None
        inlier_mask = mask.reshape(-1).astype(bool)

        err_all = reprojection_errors(H, cam_pts, map_pts)
        err_in = err_all[inlier_mask] if np.any(inlier_mask) else err_all

        rmse_all = np.sqrt(np.mean(err_all**2)) if len(err_all) else float("nan")
        rmse_in  = np.sqrt(np.mean(err_in**2))  if len(err_in) else float("nan")

        print(f"[OK] H_cam→map computed. pairs={n}, inliers={int(np.sum(inlier_mask))}/{n}")
        print(f"     RMSE(all)={rmse_all:.3f}px  RMSE(inlier)={rmse_in:.3f}px  max(all)={np.max(err_all):.3f}px")
        return H, cam_pts, map_pts, inlier_mask.copy()

    def save_inlier_visualization(path, base_img, cam_pts, map_pts, mask, inlier_color=(0,255,0), outlier_color=(0,0,255)):
        vis = base_img.copy()
        total = len(mask)
        inliers = int(np.sum(mask))
        for i in range(total):
            color = inlier_color if mask[i] else outlier_color
            thickness = POINT_THICKNESS + 1
            uc, vc = cam_pts[i]
            um, vm = map_pts[i]
            pc = (int(cam_rect[0] + uc * s_cam), int(cam_rect[1] + vc * s_cam))
            pm = (int(map_rect[0] + um * s_map), int(map_rect[1] + vm * s_map))
            cv2.circle(vis, pc, POINT_RADIUS, color, thickness)
            cv2.circle(vis, pm, POINT_RADIUS, color, thickness)
            cv2.line(vis, pc, pm, color, 2, cv2.LINE_AA)
            cv2.putText(vis, f"{i+1}", (pc[0]+8, pc[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            cv2.putText(vis, f"{i+1}", (pm[0]+8, pm[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        summary = f"inliers={inliers}/{total}"
        cv2.putText(vis, summary, (10, base_img.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.imwrite(path, vis)

    # map2world H
    H_map2world = None
    if args.map2world:
        img_uv, world_xy, raw = load_map2world_json(args.map2world)
        H_map2world = homography_from_4pts(img_uv, world_xy)
        print("[OK] Loaded map2world.json and computed H_map→world")

    def reset_adjust():
        nonlocal adj_tx, adj_ty, adj_theta, adj_scale
        adj_tx = 0.0
        adj_ty = 0.0
        adj_theta = 0.0
        adj_scale = 1.0

    def get_H_display() -> Optional[np.ndarray]:
        """
        화면/오버레이에 실제로 적용되는 H:
          H_display = ΔH @ H_current
        """
        if H_current is None:
            return None
        dH = make_deltaH(adj_tx, adj_ty, adj_theta, adj_scale)
        return dH @ H_current

    def save_final():
        """
        저장 파일을 2개로 단순화:
          - overlay.png
          - H_cam2world.json (H_map2world 있으면 합성)
        """
        H_disp = get_H_display()
        if H_disp is None:
            print("[WARN] 저장할 H가 없습니다. 먼저 점으로 H를 계산하세요(ENTER).")
            return

        # warp/overlay (cam->map)
        warped, overlay = warp_and_overlay(cam, mp, H_disp, alpha=float(args.alpha), margin=int(args.overlay_margin))
        cv2.imwrite(os.path.join(args.outdir, "overlay.png"), overlay)

        # compose final to world if map2world exists
        H_final = H_disp
        if H_map2world is not None:
            H_final = H_map2world @ H_disp

        save_H_json(os.path.join(args.outdir, "H_cam2world.json"), H_final)
        print("[OK] Saved H_cam2world.json and overlay.png")

    # -----------------------------
    # Main loop
    # -----------------------------
    while True:
        canvas = base_canvas.copy()

        # draw points
        draw_points_and_lines(canvas, cam_rect, map_rect, pairs, s_cam, s_map, pending_cam_pt=pending_cam)

        # HUD
        hud_y0 = target_h + 10
        n_pairs = min(len(pairs.cam_pts), len(pairs.map_pts))

        msg1 = "Click: CAM(left) then MAP(right) | Backspace: remove last point"
        msg2 = "ENTER: compute/apply H | U: undo last apply | S: save | ESC: quit"
        msg3 = "Adjust keys: I/J/K/L=move  Q/E=rotate  +/-=scale (applied on ENTER)"
        msg4 = f"Adjust values: tx={adj_tx:.1f}px  ty={adj_ty:.1f}px  theta={adj_theta:.2f}deg  scale={adj_scale:.6f}"
        msg5 = f"Pairs={n_pairs} (cam={len(pairs.cam_pts)}, map={len(pairs.map_pts)})  RANSAC_th={args.ransac_th}px"
        msg6 = f"Mouse: magnifier | Drag point | Z/X/C drag sensitivity -> {drag_mode_name} ({drag_sensitivity}x)"

        cv2.putText(canvas, msg1, (10, hud_y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
        cv2.putText(canvas, msg2, (10, hud_y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
        cv2.putText(canvas, msg3, (10, hud_y0 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
        cv2.putText(canvas, msg4, (10, hud_y0 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(canvas, msg5, (10, hud_y0 + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,220,255), 1, cv2.LINE_AA)
        cv2.putText(canvas, msg6, (10, hud_y0 + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,120), 1, cv2.LINE_AA)

        # show status
        if H_current is not None:
            cv2.putText(canvas, "H_current ready (cam->map).", (canvas_w - 360, hud_y0 + 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)

        # -----------------------------
        # Live preview: show overlay in a separate window (optional but practical)
        # -----------------------------
        H_disp = get_H_display()
        if H_disp is not None:
            if last_overlay_disp is None or not np.allclose(H_disp, last_overlay_disp):
                _, overlay_live_cache = warp_and_overlay(cam, mp, H_disp, alpha=float(args.alpha), margin=int(args.overlay_margin))
                last_overlay_disp = H_disp.copy()
            if overlay_live_cache is not None:
                cv2.imshow("overlay_preview", overlay_live_cache)
        else:
            overlay_live_cache = None
            last_overlay_disp = None
            # close if exists
            try:
                cv2.destroyWindow("overlay_preview")
            except Exception:
                pass

        # -----------------------------
        # Magnifier view (CAM / MAP) - show only one persistent window
        # -----------------------------
        if hover_cam_uv is not None:
            if mag_mode != "cam":
                cv2.namedWindow("MAG", cv2.WINDOW_NORMAL)
                mag_mode = "cam"
            draw_magnifier(cam, hover_cam_uv[0], hover_cam_uv[1], pts=pairs.cam_pts, color=(255, 0, 255), name="MAG")
        elif hover_map_uv is not None:
            if mag_mode != "map":
                cv2.namedWindow("MAG", cv2.WINDOW_NORMAL)
                mag_mode = "map"
            draw_magnifier(mp, hover_map_uv[0], hover_map_uv[1], pts=pairs.map_pts, color=(0, 200, 0), name="MAG")
        else:
            if mag_mode is not None:
                # 창 유지, 내용만 숨김
                cv2.imshow("MAG", np.zeros((MAG_SIZE*MAG_SCALE, MAG_SIZE*MAG_SCALE, 3), dtype=np.uint8))
                mag_mode = None

        cv2.imshow(win, canvas)
        key_raw = cv2.waitKeyEx(10)  # use waitKeyEx for arrows
        key = key_raw & 0xFF if key_raw != -1 else -1

        # -----------------------------
        # Key handling (single-dispatch)
        # -----------------------------
        if key in (ord('z'), ord('Z')):
            drag_sensitivity = 0.1
            drag_mode_name = "ULTRA-FINE"
            continue

        elif key in (ord('x'), ord('X')):
            drag_sensitivity = 0.25
            drag_mode_name = "FINE"
            continue

        elif key in (ord('c'), ord('C')):
            drag_sensitivity = 1.0
            drag_mode_name = "NORMAL"
            continue

        elif key in (ord('u'), ord('U')):
            # undo last apply
            if len(H_prev_stack) > 0:
                H_current = H_prev_stack.pop()
                reset_adjust()
                print("[OK] Undo: restored previous H_current.")
            else:
                print("[WARN] Undo stack is empty.")

        elif key in (ord('s'), ord('S')):
            save_final()

        elif key in (8, 127):  # Backspace/Delete: remove last clicked point
            pending_cam = None
            pairs.pop_last()
            print("[OK] Removed last clicked point.")
            continue

        elif key == 27:  # ESC
            break

        # Micro-adjust controls and ENTER, undo point pairs: keep as before (not affected by Z/X/C/R/L/U/S above)
        # Micro-adjust controls (IJKL, Q/E, +/-)
        if key in (ord('j'), ord('J')):
            adj_tx -= 1.0
        elif key in (ord('l'), ord('L')):
            adj_tx += 1.0
        elif key in (ord('i'), ord('I')):
            adj_ty -= 1.0
        elif key in (ord('k'), ord('K')):
            adj_ty += 1.0

        # rotation
        if key in (ord('q'), ord('Q')):
            adj_theta -= 0.2
        if key in (ord('e'), ord('E')):
            adj_theta += 0.2

        # scale: '+'/'-' (키보드 레이아웃 차이를 고려해 '='도 허용)
        if key in (ord('+'), ord('=')):
            adj_scale *= 1.002
        if key in (ord('-'), ord('_')):
            adj_scale /= 1.002

        # ENTER behavior:
        #  - if no H_current yet: compute from clicked pairs, set H_base/H_current
        #  - else: apply current ΔH to H_current (commit)
        if key_raw in (13, 10):  # ENTER
            if H_current is None:
                res = compute_H_points()
                if res is None:
                    continue
                H0, cam_pts_used, map_pts_used, mask_used = res
                save_inlier_visualization(os.path.join(args.outdir, "inlier_pairs.png"),
                                          base_canvas, cam_pts_used, map_pts_used, mask_used)
                H_base = H0.copy()
                H_current = H0.copy()
                reset_adjust()
                print("[OK] Initialized H_current from point correspondences. Use arrows/QE/+- to fine tune, then ENTER to apply.")
            else:
                H_display = get_H_display()
                if H_display is None:
                    continue
                # push previous
                H_prev_stack.append(H_current.copy())
                # commit
                H_current = H_display
                reset_adjust()
                print("[OK] Applied ΔH -> updated H_current (committed).")

        # (중복 U 처리 제거)

    cv2.destroyAllWindows()
    try:
        cv2.destroyWindow("overlay_preview")
    except Exception:
        pass

if __name__ == "__main__":
    main()


"""
python ./utils/make_H/homography_tool_2.py \
  --cam ./test_img.jpeg \
  --map ./utils/make_H/ces_real_map.png \
  --map2world ./utils/make_H/ces_real_img2world.json \
  --outdir ./out_h
"""
