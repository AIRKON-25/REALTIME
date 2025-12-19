import argparse
import glob
import os
from typing import List

import numpy as np

from utils.tracking.tracker import SortTracker, load_detections_from_file

# tracker_demo.py --input-folder <dir> --output tracking_output.txt
def run_demo(
    input_folder: str,
    output_path: str,
    max_age: int = 10,
    min_hits: int = 3,
    iou_threshold: float = 0.15,
) -> None:
    file_pattern = os.path.join(input_folder, "merged_frame_*.txt")
    frame_files: List[str] = sorted(glob.glob(file_pattern))

    if not frame_files:
        print(f"[tracker_demo] no input files found under '{file_pattern}'")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tracker = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    all_tracking_results = []

    print(f"[tracker_demo] processing {len(frame_files)} frames...")
    for frame_idx, filepath in enumerate(frame_files):
        detections = load_detections_from_file(filepath)
        tracked_objects = tracker.update(detections, None)

        if len(tracked_objects) > 0:
            frame_id_column = np.full((tracked_objects.shape[0], 1), frame_idx)
            frame_results = np.hstack((frame_id_column, tracked_objects))
            all_tracking_results.append(frame_results)

        if (frame_idx + 1) % 50 == 0 or frame_idx == len(frame_files) - 1:
            print(f"[tracker_demo] frame {frame_idx + 1}/{len(frame_files)}, active tracks={len(tracker.tracks)}")

    if not all_tracking_results:
        print("[tracker_demo] no confirmed/lost tracks produced")
        return

    try:
        final_results = np.vstack(all_tracking_results)
        header = "frame_id, track_id, class, x_center, y_center, length, width, angle"
        np.savetxt(
            output_path,
            final_results,
            fmt=['%d', '%d', '%d', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f'],
            delimiter=',',
            header=header,
            comments='',
        )
        print(f"[tracker_demo] saved results -> {output_path}")
    except Exception as exc:
        print(f"[tracker_demo] failed to save results: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Run SortTracker on merged detection frames.")
    parser.add_argument("--input-folder", default="/merge_dist_wbf_drop", help="Folder containing merged_frame_*.txt")
    parser.add_argument("--output", default="tracking_output.txt", help="Output path for aggregated tracks")
    parser.add_argument("--max-age", type=int, default=10, help="Max age for tracks before deletion")
    parser.add_argument("--min-hits", type=int, default=3, help="Minimum hits to confirm a track")
    parser.add_argument("--iou-thr", type=float, default=0.15, help="IoU threshold for assignment")
    args = parser.parse_args()

    run_demo(
        input_folder=args.input_folder,
        output_path=args.output,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_thr,
    )


if __name__ == "__main__":
    main()
