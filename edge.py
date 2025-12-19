#!/usr/bin/env python3

import argparse
import json
import math
import os
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import depthai as dai
import numpy as np

from utils.edge.evaluation_utils import decode_predictions
from utils.edge.geometry_utils import parallelogram_from_triangle, tiny_filter_on_dets
from utils.edge.inference_lstm_onnx_pointcloud_tensorrt import TensorRTTemporalRunner


def preprocess_frame(frame_bgr: np.ndarray, target_hw: Tuple[int, int]) -> dict:
    target_h, target_w = target_hw
    resized_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, 0).copy()

    H0, W0 = frame_bgr.shape[:2]
    scale_to_orig_x = float(W0) / float(target_w)
    scale_to_orig_y = float(H0) / float(target_h)

    return {
        "img_np": img_np,
        "resized_bgr": resized_bgr,
        "orig_bgr": frame_bgr,
        "scale_to_orig_x": scale_to_orig_x,
        "scale_to_orig_y": scale_to_orig_y,
    }


def overlay_detections(frame_bgr: np.ndarray,
                       dets,
                       scale_to_orig_x: float,
                       scale_to_orig_y: float) -> np.ndarray:
    vis = frame_bgr.copy()
    for det in dets:
        tri = np.asarray(det["tri"], dtype=np.float32).copy()
        tri[:, 0] *= scale_to_orig_x
        tri[:, 1] *= scale_to_orig_y
        poly = parallelogram_from_triangle(tri[0], tri[1], tri[2]).astype(np.int32)
        class_id = det.get("class_id", det.get("cls"))
        color = _class_color(class_id)
        cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=2)
        cx, cy = int(tri[0][0]), int(tri[0][1])
        score = float(det.get("score", 0.0))
        cls_text = f"C{int(class_id)}" if class_id is not None else "C?"
        text = f"{cls_text} {score:.2f}"
        cv2.putText(vis, text, (cx, max(0, cy - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def draw_pred_pseudo3d(
    image_bgr: np.ndarray,
    tri_records: List[dict],
    *,
    dy: Optional[int] = None,
    height_scale: float = 0.5,
    min_dy: int = 8,
    max_dy: int = 80,
) -> Optional[np.ndarray]:
    """
    tri_records: [{\"tri\": (3,2) np.ndarray, \"color_info\": dict|None, \"class_id\": int|None, \"score\": float}, ...]
    """
    if not tri_records:
        return None

    img = image_bgr.copy()
    H, W = image_bgr.shape[:2]

    records: List[Tuple[np.ndarray, Optional[dict], Optional[int], float]] = []
    for rec in tri_records:
        tri = np.asarray(rec.get("tri"), dtype=np.float32)
        if tri.shape != (3, 2) or not np.all(np.isfinite(tri)):
            continue
        poly4 = parallelogram_from_triangle(tri[0], tri[1], tri[2]).astype(np.float32)
        color_info = rec.get("color_info")
        class_id = rec.get("class_id")
        score = float(rec.get("score", 0.0))
        records.append((poly4, color_info, class_id, score))

    if not records:
        return None

    records.sort(key=lambda rec: rec[0][:, 1].max())

    for poly, color_info, class_id, score in records:
        x_min, x_max = poly[:, 0].min(), poly[:, 0].max()
        y_min, y_max = poly[:, 1].min(), poly[:, 1].max()
        w = max(1.0, float(x_max - x_min))
        h = max(1.0, float(y_max - y_min))
        diag = math.sqrt(w * w + h * h)

        off = int(np.clip(height_scale * diag, min_dy, max_dy)) if dy is None else int(dy)

        base = poly.astype(int)
        top = base.copy()
        top[:, 1] -= off

        overlay = img.copy()
        base_color = _class_color(class_id)
        cv2.fillPoly(overlay, [top], (0, 200, 255))
        cv2.fillPoly(overlay, [base], base_color)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        for i in range(len(base)):
            cv2.line(img, tuple(base[i]), tuple(top[i]), base_color, 2)
        cv2.polylines(img, [base], True, base_color, 2)
        cv2.polylines(img, [top], True, base_color, 2)

        cls_text = f"C{int(class_id)}" if class_id is not None else "C?"
        score_text = f"{score:.2f}"
        color_hex = None
        if color_info:
            color_hex = color_info.get("hex")
        if color_info and color_info.get("bgr"):
            b, g, r = [int(np.clip(v, 0, 255)) for v in color_info["bgr"]]
            x_center = int(base[:, 0].mean())
            y_top_min = int(top[:, 1].min())
            cy = y_top_min - 12
            patch_w, patch_h = 18, 18
            x1 = max(0, min(W - patch_w, x_center - patch_w // 2))
            y1 = max(0, min(H - patch_h, cy - patch_h // 2))
            x2 = min(W - 1, x1 + patch_w)
            y2 = min(H - 1, y1 + patch_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), thickness=-1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)
            text_lines = []
            if color_hex:
                text_lines.append(color_hex.upper())
            text_lines.append(f"{cls_text} {score_text}")
            for i, line in enumerate(text_lines):
                text_org = (x1, max(0, y1 - 6 - i * 12))
                cv2.putText(img, line, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            x_center = int(base[:, 0].mean())
            y_top_min = int(top[:, 1].min())
            text_org = (x_center, max(0, y_top_min - 6))
            cv2.putText(img, f"{cls_text} {score_text}", text_org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return img


CLASS_COLOR_TABLE = {
    0: (0, 255, 0),      # lime
    1: (0, 255, 255),    # yellow
    2: (255, 0, 0),      # blue (BGR)
    3: (255, 0, 255),    # magenta
    4: (0, 128, 255),    # orange
}


def _class_color(class_id: Optional[int]) -> Tuple[int, int, int]:
    if class_id is None:
        return (0, 255, 0)
    return CLASS_COLOR_TABLE.get(int(class_id), (0, 200, 255))


class VisualizationWorker(threading.Thread):
    def __init__(self, show_window: bool):
        super().__init__(daemon=True)
        self.queue: queue.Queue = queue.Queue(maxsize=2)
        self.show_window = show_window
        self.stop_evt = threading.Event()
        self._last_key = -1

    def submit(self, payload):
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(payload)
            except queue.Full:
                pass

    def get_key(self) -> int:
        key = self._last_key
        self._last_key = -1
        return key

    def stop(self):
        self.stop_evt.set()

    def run(self):
        while not self.stop_evt.is_set():
            try:
                frame_bgr, tri_records, dets, sx, sy, infer_ms, save_path = self.queue.get(timeout=0.01)
            except queue.Empty:
                if self.show_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        self._last_key = key
                continue

            vis_frame = None
            if tri_records:
                vis_frame = draw_pred_pseudo3d(frame_bgr, tri_records)
            if vis_frame is None:
                vis_frame = overlay_detections(frame_bgr, dets, sx, sy)
            cv2.putText(vis_frame, f"Infer {infer_ms:.1f} ms | dets {len(dets)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            if save_path:
                try:
                    cv2.imwrite(save_path, vis_frame)
                except Exception as exc:
                    print(f"[VisWorker] WARN: failed to save {save_path}: {exc}")

            if self.show_window:
                if vis_frame is not None:
                    vis_frame = cv2.resize(vis_frame, (500, 500)) 
                cv2.imshow("DepthAI TRT Live", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._last_key = key


def extract_tris_and_colors(dets,
                            orig_bgr: np.ndarray,
                            scale_to_orig_x: float,
                            scale_to_orig_y: float) -> Tuple[List[dict], List[Optional[dict]]]:
    tri_records: List[dict] = []
    det_color_infos: List[Optional[dict]] = []
    Hc, Wc = orig_bgr.shape[:2]
    for det in dets:
        tri = det.get("tri")
        color_info: Optional[dict] = None
        tri_orig = None
        if tri is not None:
            tri = np.asarray(tri, dtype=np.float32)
            if tri.shape == (3, 2):
                tri_orig = tri.copy()
                tri_orig[:, 0] *= scale_to_orig_x
                tri_orig[:, 1] *= scale_to_orig_y

                poly = parallelogram_from_triangle(tri_orig[0], tri_orig[1], tri_orig[2]).astype(np.float32)
                poly_int = np.round(poly).astype(np.int32)
                x, y, w, h = cv2.boundingRect(poly_int)
                if w > 0 and h > 0:
                    x0 = max(0, x)
                    y0 = max(0, y)
                    x1 = min(Wc, x0 + w)
                    y1 = min(Hc, y0 + h)
                    if x1 > x0 and y1 > y0:
                        roi_img = orig_bgr[y0:y1, x0:x1]
                        local_poly = (poly_int - np.array([x0, y0], dtype=np.int32))
                        local_poly[:, 0] = np.clip(local_poly[:, 0], 0, x1 - x0 - 1)
                        local_poly[:, 1] = np.clip(local_poly[:, 1], 0, y1 - y0 - 1)
                        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                        cv2.fillConvexPoly(mask, local_poly, 255)
                        mean_bgr = cv2.mean(roi_img, mask=mask)
                        mb = int(np.clip(mean_bgr[0], 0, 255))
                        mg = int(np.clip(mean_bgr[1], 0, 255))
                        mr = int(np.clip(mean_bgr[2], 0, 255))
                        hexcol = f"{mr:02x}{mg:02x}{mb:02x}"
                        color_info = {
                            "hex": hexcol,
                            "bgr": (mb, mg, mr),
                        }
        det_color_infos.append(color_info)
        if tri_orig is not None:
            tri_records.append({
                "tri": tri_orig,
                "color_info": color_info,
                "class_id": det.get("class_id", det.get("cls")),
                "score": float(det.get("score", 0.0)),
            })
    return tri_records, det_color_infos
def load_homography_matrix(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    matrix_path = Path(path).expanduser()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Homography file not found: {matrix_path}")
    if matrix_path.suffix.lower() == ".npy":
        H = np.load(str(matrix_path))
    else:
        with open(matrix_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "H" in data:
            H = np.asarray(data["H"], dtype=np.float64)
        elif isinstance(data, dict) and "points" in data:
            img_pts = []
            world_pts = []
            for item in data["points"]:
                img = item.get("image_px", {})
                world = item.get("world_xy", {})
                if not {"u", "v"} <= img.keys() or not {"x", "y"} <= world.keys():
                    continue
                img_pts.append([float(img["u"]), float(img["v"])])
                world_pts.append([float(world["x"]), float(world["y"])])
            if len(img_pts) < 4:
                raise ValueError("Need at least 4 correspondences to compute homography.")
            img_pts = np.asarray(img_pts, dtype=np.float64)
            world_pts = np.asarray(world_pts, dtype=np.float64)
            H, status = cv2.findHomography(img_pts, world_pts, method=0)
            if H is None:
                raise RuntimeError("Failed to compute homography from provided points.")
        else:
            raise ValueError(f"Unsupported homography file structure: {matrix_path}")
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")
    return H


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    pts = pts.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return projected.reshape(-1, 2)


def dets_to_bev_entries(dets,
                        homography: Optional[np.ndarray],
                        color_infos: Optional[List[Optional[dict]]] = None) -> List[dict]:
    if homography is None:
        return []
    tri_list: List[np.ndarray] = []
    det_indices: List[int] = []
    for idx, det in enumerate(dets):
        tri = det.get("tri")
        if tri is None:
            continue
        tri_np = np.asarray(tri, dtype=np.float64)
        if tri_np.shape != (3, 2):
            continue
        tri_list.append(tri_np)
        det_indices.append(idx)

    if not tri_list:
        return []

    tri_stack = np.stack(tri_list, axis=0)
    pts = tri_stack.reshape(-1, 1, 2)
    tri_world_all = cv2.perspectiveTransform(pts, homography.astype(np.float64))
    tri_world_all = tri_world_all.reshape(-1, 3, 2)

    entries: List[dict] = []
    for tri_world, det_idx in zip(tri_world_all, det_indices):
        if not np.all(np.isfinite(tri_world)):
            continue
        center = tri_world[0]
        front_left = tri_world[1]
        front_right = tri_world[2]
        front_center = (front_left + front_right) * 0.5
        forward_vec = front_center - center
        length_half = np.linalg.norm(forward_vec)
        width = np.linalg.norm(front_right - front_left)
        if length_half < 1e-6 or width < 1e-6:
            continue
        yaw = math.degrees(math.atan2(forward_vec[1], forward_vec[0]))
        yaw = (yaw + 180.0) % 360.0 - 180.0
        det = dets[det_idx]
        cls_raw = det.get("class_id", det.get("cls"))
        entry = {
            "center": [float(center[0]), float(center[1])],
            "length": float(length_half * 2.0),
            "width": float(width),
            "yaw": float(yaw),
            "score": float(det.get("score", 0.0)),
            "class_id": int(cls_raw) if cls_raw is not None else -1,
        }
        if color_infos is not None and det_idx < len(color_infos):
            color_info = color_infos[det_idx]
            if color_info:
                color_value = color_info.get("hex")
                if color_value:
                    entry["color_hex"] = color_value
        entries.append(entry)
    return entries


class UDPSender:
    def __init__(
        self,
        host: str,
        port: int,
        fmt: str = "json",
        max_bytes: int = 65000,
        fixed_length: Optional[float] = None,
        fixed_width: Optional[float] = None,
        on_send: Optional[Callable[[bytes, int, float, Optional[float], List[dict]], None]] = None,
    ):
        self.addr = (host, int(port))
        self.fmt = fmt
        self.max_bytes = max_bytes
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.fixed_length = float(fixed_length) if fixed_length is not None else None
        self.fixed_width = float(fixed_width) if fixed_width is not None else None
        self.on_send = on_send

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def _pack_json(self, cam_id: int, ts: float, bev_dets, capture_ts: Optional[float]):
        items = []
        for d in bev_dets or []:
            length = self.fixed_length if self.fixed_length is not None else d["length"]
            width = self.fixed_width if self.fixed_width is not None else d["width"]
            items.append({
                "center": [float(d["center"][0]), float(d["center"][1])],
                "length": float(length),
                "width": float(width),
                "yaw": float(d["yaw"]),
                "score": float(d["score"]),
                "class_id": int(d.get("class_id", -1)),
            })
            color_hex = d.get("color_hex")
            if color_hex:
                items[-1]["color_hex"] = color_hex
        msg = {
            "type": "bev_labels",
            "camera_id": cam_id,
            "timestamp": ts,
            "capture_ts": capture_ts,
            "items": items,
        }
        return json.dumps(msg, ensure_ascii=False).encode("utf-8")

    def send(self, cam_id: int, ts: float, bev_dets=None, capture_ts: Optional[float] = None):
        bev_list = bev_dets or []
        if self.fmt == "json":
            payload = self._pack_json(cam_id, ts, bev_list, capture_ts)
            try:
                print(f"[UDP] {payload.decode('utf-8')}")
            except Exception:
                print(f"[UDP] bytes={payload[:200]}{'...' if len(payload) > 200 else ''}")
        else:
            raise ValueError(f"Unsupported UDP payload fmt: {self.fmt}")
        if self.on_send is not None:
            try:
                self.on_send(payload, cam_id, ts, capture_ts, bev_list)
            except Exception as exc:
                print(f"[UDP] on_send error: {exc}")
        if len(payload) <= self.max_bytes:
            self.sock.sendto(payload, self.addr)


def run_live_inference(args) -> None:
    target_h, target_w = map(int, args.img_size.split(","))
    target_hw = (target_h, target_w)
    strides = [float(s) for s in args.strides.split(",")]

    runner = TensorRTTemporalRunner(
        args.weights,
        state_stride_hint=args.state_stride_hint,
        default_hidden_ch=args.default_hidden_ch
    )

    cam_w, cam_h = map(int, args.camera_size.split(","))

    save_dir = None
    if args.save_dir:
        save_dir = os.path.abspath(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)

    homography = load_homography_matrix(args.homography) if args.homography else None
    if args.udp_enable and homography is None:
        raise ValueError("UDP output requested but no homography (--homography) provided.")
    udp_sender = None
    if args.udp_enable:
        udp_sender = UDPSender(
            args.udp_host,
            args.udp_port,
            fmt=args.udp_format,
            fixed_length=args.udp_fixed_length,
            fixed_width=args.udp_fixed_width,
        )

    resize_mode = None
    if args.resize_mode is not None:
        mode_key = args.resize_mode.strip().upper()
        if mode_key not in ("", "NONE"):
            if not hasattr(dai.ImgResizeMode, mode_key):
                raise ValueError(f"Unsupported resize mode: {args.resize_mode}")
            resize_mode = getattr(dai.ImgResizeMode, mode_key)

    vis_worker = None
    if args.show_vis or save_dir is not None:
        vis_worker = VisualizationWorker(show_window=args.show_vis)
        vis_worker.start()

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        request_kwargs = {}
        if resize_mode is not None:
            request_kwargs["resizeMode"] = resize_mode
        if args.enable_undistort:
            request_kwargs["enableUndistortion"] = True
        video_queue = (
            cam.requestOutput((cam_w, cam_h), **request_kwargs)
            .createOutputQueue(maxSize=1, blocking=False)
        )

        pipeline.start()
        print(f"[DepthAI] Streaming {cam_w}x{cam_h} → inference size {target_w}x{target_h}")
        print("[Info] Press 'q' to quit, 'r' to reset temporal state.")

        frame_idx = 0
        last_fps_ts = time.time()
        fps = 0.0
        try:
            while pipeline.isRunning():
                img_packet = video_queue.tryGet()
                if img_packet is None:
                    continue
                cam_latency_ms = (
                    dai.Clock.now() - img_packet.getTimestamp()
                ).total_seconds() * 1000.0
                frame_bgr = img_packet.getCvFrame()

                prep = preprocess_frame(frame_bgr, target_hw)

                t0 = time.time()
                outputs = runner.forward(prep["img_np"])
                infer_ms = (time.time() - t0) * 1000.0

                t_post0 = time.time()
                dets = decode_predictions(
                    outputs,
                    strides,
                    clip_cells=args.clip_cells,
                    conf_th=args.conf,
                    nms_iou=args.nms_iou,
                    topk=args.topk,
                    contain_thr=args.contain_thr,
                    score_mode=args.score_mode,
                    use_gpu_nms=True
                )[0]
                dets = tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)
                tri_records, det_color_infos = [], []
                if dets:
                    tri_records, det_color_infos = extract_tris_and_colors(
                        dets,
                        prep["orig_bgr"],
                        prep["scale_to_orig_x"],
                        prep["scale_to_orig_y"]
                    )
                bev_entries = dets_to_bev_entries(dets, homography, det_color_infos)
                post_ms = (time.time() - t_post0) * 1000.0

                t_vis0 = time.time()
                vis_ms = 0.0
                key = -1
                vis_save_path = None
                if save_dir is not None:
                    vis_save_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
                if vis_worker is not None and (args.show_vis or vis_save_path is not None):
                    vis_payload = (
                        prep["orig_bgr"].copy(),
                        tri_records,
                        dets,
                        prep["scale_to_orig_x"],
                        prep["scale_to_orig_y"],
                        infer_ms,
                        vis_save_path,
                    )
                    vis_worker.submit(vis_payload)
                elif vis_save_path is not None:
                    try:
                        cv2.imwrite(vis_save_path, prep["orig_bgr"])
                    except Exception as exc:
                        print(f"[Vis] WARN: failed to save {vis_save_path}: {exc}")
                if vis_worker is not None and args.show_vis:
                    key = vis_worker.get_key()
                else:
                    key = cv2.waitKey(1) & 0xFF
                vis_ms = (time.time() - t_vis0) * 1000.0
                frame_idx += 1

                capture_ts = None
                ts_raw = img_packet.getTimestamp()
                if ts_raw is not None and hasattr(ts_raw, "total_seconds"):
                    capture_ts = ts_raw.total_seconds()

                if udp_sender is not None:
                    try:
                        udp_sender.send(
                            cam_id=args.camera_id,
                            ts=time.time(),
                            bev_dets=bev_entries,
                            capture_ts=capture_ts
                        )
                    except Exception as exc:
                        print(f"[UDP] send error: {exc}")

                total_ms = cam_latency_ms + infer_ms + post_ms + vis_ms
                now = time.time()
                dt = now - last_fps_ts
                if dt > 0:
                    fps = 1.0 / dt
                last_fps_ts = now
                print(
                    f"[Timing] cam {cam_latency_ms:.1f} ms | infer {infer_ms:.1f} ms | "
                    f"post {post_ms:.1f} ms | vis {vis_ms:.1f} ms | total {total_ms:.1f} ms | fps {fps:.1f}"
                )

                if key == ord("q"):
                    break
                if key == ord("r"):
                    print("[Info] State reset requested.")
                    runner.reset()
        finally:
            if vis_worker is not None:
                vis_worker.stop()
                vis_worker.join(timeout=1.0)
            cv2.destroyAllWindows()
            if udp_sender is not None:
                udp_sender.close()


def parse_args():
    parser = argparse.ArgumentParser("DepthAI live inference with TensorRT temporal model")
    parser.add_argument("--weights", type=str, required=True, help="TensorRT engine (.engine) path")
    parser.add_argument("--img-size", type=str, default="864,1536", help="Model input H,W")
    parser.add_argument("--camera-size", type=str, default="1920,1080", help="DepthAI output width,height")
    parser.add_argument("--score-mode", type=str, default="obj*cls", choices=["obj", "cls", "obj*cls"])
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--contain-thr", type=float, default=0.85)
    parser.add_argument("--clip-cells", type=float, default=None)
    parser.add_argument("--strides", type=str, default="8,16,32")
    parser.add_argument("--state-stride-hint", type=int, default=32)
    parser.add_argument("--default-hidden-ch", type=int, default=256)
    parser.add_argument("--show-vis", dest="show_vis", action="store_true", default=True,
                        help="Enable OpenCV visualization window (default: on)")
    parser.add_argument("--no-show-vis", dest="show_vis", action="store_false",
                        help="Disable OpenCV visualization window")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Optional directory to save each processed frame (after inference)")
    parser.add_argument("--enable-undistort", dest="enable_undistort", action="store_true",
                        help="Enable on-device undistortion for camera output")
    parser.add_argument("--disable-undistort", dest="enable_undistort", action="store_false",
                        help="Disable on-device undistortion")
    parser.set_defaults(enable_undistort=True)
    parser.add_argument("--resize-mode", type=str, default="LETTERBOX",
                        help="DepthAI resize mode (CROP|STRETCH|LETTERBOX|NONE)")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Camera identifier included in UDP payloads")
    parser.add_argument("--homography", type=str, default=None,
                        help="Path to 3x3 homography (.npy or JSON with correspondences)")
    parser.add_argument("--udp-enable", action="store_true",
                        help="Enable UDP streaming of BEV detections (same format as main.py)")
    parser.add_argument("--udp-host", type=str, default="192.168.0.165")
    parser.add_argument("--udp-port", type=int, default=50050)
    parser.add_argument("--udp-format", choices=["json", "text"], default="json")
    parser.add_argument("--udp-fixed-length", type=float, default=4.4,
                        help="Override vehicle length in UDP payloads (meters)")
    parser.add_argument("--udp-fixed-width", type=float, default=2.7,
                        help="Override vehicle width in UDP payloads (meters)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_live_inference(args)


if __name__ == "__main__":
    main()
