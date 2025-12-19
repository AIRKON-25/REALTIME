import argparse
import glob
import os
from typing import List

from data_io.camera_assets import load_camera_markers, safe_float
from utils.tracking.merge_dist_wbf import merge_frame_with_distance_weight

# merge_dist_wbf_cli.py --pred-dir <pred> --out-dir <out> --camera-positions <camera.json>
def _build_camera_setups(camera_positions_path: str) -> List[dict]:
    markers = load_camera_markers(camera_positions_path) if camera_positions_path else []
    setups = []
    for marker in markers:
        pos = marker.get("position") or {}
        x = safe_float(pos.get("x"))
        y = safe_float(pos.get("y"))
        if x is None or y is None:
            continue
        setups.append({"name": marker.get("name") or marker.get("key"), "pos": {"x": x, "y": y}})
    return setups


def _discover_frame_keys(pred_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(pred_dir, "cam*_frame_*.txt")))
    frame_keys = set()
    for path in files:
        name = os.path.basename(path)
        parts = name.split("_frame_")
        if len(parts) >= 2:
            key_part = parts[1]
            key = os.path.splitext(key_part)[0]
            frame_keys.add(key)
    return sorted(frame_keys)


def main():
    parser = argparse.ArgumentParser(description="Distance-weighted WBF merging for multi-camera detections.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing cam*_frame_*.txt prediction files")
    parser.add_argument("--out-dir", required=True, help="Directory to write merged_frame_*.txt files")
    parser.add_argument("--camera-positions", default=None, help="Camera position JSON (same format as server)")
    parser.add_argument("--frame-keys", nargs="*", default=None, help="Explicit frame keys to process")
    parser.add_argument("--iou-thr", type=float, default=0.25, help="AABB IoU threshold for clustering")
    parser.add_argument("--d0", type=float, default=5.0, help="Distance weight pivot")
    parser.add_argument("--p", type=float, default=2.0, help="Distance weight exponent")
    args = parser.parse_args()

    frame_keys = args.frame_keys or _discover_frame_keys(args.pred_dir)
    if not frame_keys:
        print(f"[merge_dist_wbf_cli] no frame keys found under {args.pred_dir}")
        return

    camera_setups = _build_camera_setups(args.camera_positions) if args.camera_positions else []

    for frame_key in frame_keys:
        merge_frame_with_distance_weight(
            args.pred_dir,
            frame_key,
            args.out_dir,
            camera_setups,
            iou_cluster_thr=args.iou_thr,
            d0=args.d0,
            p=args.p,
        )


if __name__ == "__main__":
    main()
