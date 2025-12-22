#!/usr/bin/env python3
"""
Simple ROI creator/editor: load an image, paint an ROI mask with a brush, and save mask + metadata to .npz.
If --out already exists, the existing mask/meta will be loaded first so you can edit/extend it.

Usage:
  python utils/roi/create_roi_from_image.py --image path/to/frame.jpg --out roi/example_roi.npz \
      --camera-size 1536,864 --note "front lane"
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def _parse_camera_size(text: Optional[str], w_from_img: int, h_from_img: int) -> Tuple[int, int]:
    if text is None:
        return w_from_img, h_from_img
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("camera-size should be W,H (e.g., 1536,864)")
    try:
        w = int(parts[0].strip())
        h = int(parts[1].strip())
    except Exception as exc:
        raise ValueError("camera-size should be two integers (W,H)") from exc
    return w, h


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_npz(out_path: Path,
              mask: np.ndarray,
              poly_points: np.ndarray,
              meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Store meta as JSON string to keep npz pure numeric/strings.
    meta_json = json.dumps(meta, ensure_ascii=True)
    np.savez_compressed(
        out_path,
        mask=mask.astype(np.uint8),
        polygon=poly_points.astype(np.int32),
        meta=np.array([meta_json])
    )
    print(f"[OK] ROI saved to {out_path}")


def _load_roi_npz(npz_path: Path, expect_hw: Tuple[int, int]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[dict]]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "mask" not in data:
            raise ValueError("ROI npz missing 'mask' field")
        mask_raw = np.asarray(data["mask"], dtype=np.uint8)
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]
        poly = None
        if "polygon" in data:
            poly = np.asarray(data["polygon"], dtype=np.int32)
        meta = None
        if "meta" in data:
            try:
                meta = json.loads(str(data["meta"][0]))
            except Exception:
                meta = None

    H_exp, W_exp = expect_hw
    mask = mask_raw
    if mask.shape != (H_exp, W_exp):
        mask = cv2.resize(mask_raw, (W_exp, H_exp), interpolation=cv2.INTER_NEAREST)
        print(f"[ROI] Resized loaded mask from {mask_raw.shape} to {(H_exp, W_exp)}")
    return mask, poly, meta


def _blend_mask(base_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = base_bgr.copy()
    color_layer = np.zeros_like(base_bgr)
    color_layer[:, :] = (0, 200, 255)  # orange-ish overlay
    # Apply color where mask>0
    masked = cv2.addWeighted(vis, 0.4, color_layer, 0.6, 0)
    vis[mask > 0] = masked[mask > 0]
    return vis


def main():
    parser = argparse.ArgumentParser("Create ROI mask from an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the reference image")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
    parser.add_argument("--camera-size", type=str, default=None,
                        help="Camera output size W,H that this ROI matches (defaults to image size)")
    parser.add_argument("--note", type=str, default=None, help="Optional note to embed in metadata")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        print(f"[ERR] image not found: {image_path}")
        sys.exit(1)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"[ERR] failed to load image: {image_path}")
        sys.exit(1)

    H, W = img_bgr.shape[:2]
    cam_w, cam_h = _parse_camera_size(args.camera_size, W, H)

    out_path = Path(args.out).expanduser().resolve()
    mask = np.zeros((H, W), dtype=np.uint8)
    meta_base = {}
    if out_path.is_file():
        try:
            mask_loaded, poly_loaded, meta_loaded = _load_roi_npz(out_path, (H, W))
            mask = mask_loaded.copy()
            if meta_loaded:
                meta_base.update(meta_loaded)
            print(f"[ROI] Loaded existing ROI from {out_path}")
            if poly_loaded is not None:
                print(f"[ROI] Existing polygon has {len(poly_loaded)} points")
        except Exception as exc:
            print(f"[ROI] WARN: failed to load existing ROI '{out_path}': {exc}")

    print("Instructions:")
    print(" - Left click/drag: paint ROI")
    print(" - Right click/drag: erase")
    print(" - [ or - : brush smaller | ] or + : brush larger")
    print(" - r: reset mask")
    print(" - Enter/Space/s: save and quit (need at least some painted area)")
    print(" - q or ESC: quit without saving")

    window_name = "ROI Editor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    brush_size = 18
    drawing = False
    erase_mode = False
    cursor_pos: Optional[Tuple[int, int]] = None

    def _on_mouse(event, x, y, _flags, _userdata):
        nonlocal drawing, erase_mode, cursor_pos
        cursor_pos = (int(x), int(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            erase_mode = False
            cv2.circle(mask, (x, y), brush_size, 255, -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
            erase_mode = True
            cv2.circle(mask, (x, y), brush_size, 0, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            val = 0 if erase_mode else 255
            cv2.circle(mask, (x, y), brush_size, val, -1)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            drawing = False

    cv2.setMouseCallback(window_name, _on_mouse)

    while True:
        vis = _blend_mask(img_bgr, mask)
        if cursor_pos is not None:
            cv2.circle(vis, cursor_pos, brush_size, (0, 255, 0), 1, cv2.LINE_AA)
        info = f"brush:{brush_size}px  paint:L drag  erase:R drag  resize:[ ]  reset:r  save:Enter/Space/s  quit:q/ESC"
        cv2.putText(vis, info, (10, max(25, H - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            print("[Info] Quit without saving.")
            cv2.destroyAllWindows()
            sys.exit(0)
        if key in (ord("["), ord("-")):
            brush_size = max(1, brush_size - 2)
        if key in (ord("]"), ord("+"), ord("=")):
            brush_size = min(512, brush_size + 2)
        if key == ord("r"):
            mask[:] = 0
        if key in (13, 10, 32, ord("s")):  # enter, space, or s
            if int(mask.sum()) > 0:
                break
            else:
                print("[Warn] No ROI painted yet; paint some area first.")

    cv2.destroyAllWindows()

    # Extract a representative polygon (largest contour) for optional downstream use.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly = np.zeros((0, 2), dtype=np.int32)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        poly = largest.reshape(-1, 2)

    meta = dict(meta_base)
    meta.update({
        "camera_size": [cam_w, cam_h],
        "image_size": [W, H],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        "image_sha256": _sha256_file(image_path),
    })
    if args.note:
        meta["note"] = args.note
    elif "note" in meta and meta["note"] is None:
        meta.pop("note", None)

    _save_npz(out_path, mask, poly, meta)
    print(f"[Meta] camera_size={meta['camera_size']}")


if __name__ == "__main__":
    main()
