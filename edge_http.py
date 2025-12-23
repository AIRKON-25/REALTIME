#!/usr/bin/env python3

import argparse
import json
import os
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Set, Tuple

import cv2
import depthai as dai
import numpy as np

from edge import (
    UDPSender,
    dets_to_bev_entries,
    draw_pred_pseudo3d,
    extract_tris_and_colors,
    load_homography_matrix,
    overlay_detections,
    preprocess_frame,
)
from utils.edge.evaluation_utils import decode_predictions
from utils.edge.geometry_utils import parallelogram_from_triangle, tiny_filter_on_dets
from utils.edge.inference_lstm_onnx_pointcloud_tensorrt import TensorRTTemporalRunner


class FrameBroadcaster:
    def __init__(self):
        self._condition = threading.Condition()
        self._frame: Optional[bytes] = None
        self._frame_id = 0
        self._stopped = False

    def update(self, frame_bytes: bytes) -> None:
        with self._condition:
            self._frame = frame_bytes
            self._frame_id += 1
            self._condition.notify_all()

    def get_latest(self) -> Optional[bytes]:
        with self._condition:
            return self._frame

    def get_next(self, last_frame_id: int) -> Tuple[Optional[bytes], int]:
        with self._condition:
            self._condition.wait_for(lambda: self._frame_id != last_frame_id or self._stopped)
            if self._stopped:
                return None, last_frame_id
            return self._frame, self._frame_id

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


class UDPRecorder(threading.Thread):
    def __init__(self, record_path: str):
        super().__init__(daemon=True)
        self.record_path = os.path.abspath(record_path)
        self.queue: queue.Queue = queue.Queue(maxsize=1024)
        self.stop_evt = threading.Event()
        self._last_sent_ts: Optional[float] = None
        os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)

    def handle_send(self,
                    payload: bytes,
                    cam_id: int,
                    ts: float,
                    capture_ts: Optional[float],
                    bev_dets) -> None:
        delay_ms = None
        if self._last_sent_ts is not None:
            delay_ms = (ts - self._last_sent_ts) * 1000.0
        self._last_sent_ts = ts
        record = {
            "cam_id": cam_id,
            "sent_ts": ts,
            "capture_ts": capture_ts,
            "delay_ms": delay_ms,
            "payload": payload.decode("utf-8", errors="replace"),
        }
        if bev_dets is not None:
            record["bev_count"] = len(bev_dets)
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                pass

    def stop(self) -> None:
        self.stop_evt.set()

    def run(self):
        try:
            with open(self.record_path, "w", encoding="utf-8") as fp:
                while not self.stop_evt.is_set() or not self.queue.empty():
                    try:
                        rec = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    line = json.dumps(rec, ensure_ascii=False)
                    fp.write(line + "\n")
                    fp.flush()
        except Exception as exc:
            print(f"[UDPRecorder] WARN: recording stopped due to error: {exc}")


class HttpVisualizationWorker(threading.Thread):
    def __init__(self,
                 broadcaster: FrameBroadcaster,
                 jpeg_quality: int,
                 show_window: bool,
                 front_highlight: bool,
                 vehicle_cls_set: Optional[Set[int]],
                 front_color: Tuple[int, int, int],
                 warp_cfg: Optional[dict] = None,
                 warp_broadcaster: Optional[FrameBroadcaster] = None):
        super().__init__(daemon=True)
        self.queue: queue.Queue = queue.Queue(maxsize=3)
        self.broadcaster = broadcaster
        self.jpeg_quality = jpeg_quality
        self.show_window = show_window
        self.front_highlight = front_highlight
        self.vehicle_cls_set = vehicle_cls_set
        self.front_color = front_color
        self.warp_cfg = warp_cfg
        self.warp_broadcaster = warp_broadcaster
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
        while not self.stop_evt.is_set() or not self.queue.empty():
            try:
                frame_bgr, tri_records, dets, sx, sy, infer_ms, save_path = self.queue.get(timeout=0.01)
            except queue.Empty:
                if self.show_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        self._last_key = key
                continue

            vis_frame = render_vis_frame(
                frame_bgr,
                tri_records,
                dets,
                sx,
                sy,
                infer_ms,
                self.front_highlight,
                self.vehicle_cls_set,
                self.front_color
            )

            if self.warp_cfg is not None and self.warp_broadcaster is not None:
                try:
                    warped = cv2.warpPerspective(
                        frame_bgr,
                        self.warp_cfg["H"],
                        self.warp_cfg["dsize"],
                        flags=cv2.INTER_LINEAR
                    )
                    mask_keep = self.warp_cfg.get("mask_keep")
                    if mask_keep is not None:
                        warped[~mask_keep] = 0
                    rotate_code = self.warp_cfg.get("rotate_code")
                    if rotate_code is not None:
                        warped = cv2.rotate(warped, rotate_code)
                    success_warp, encoded_warp = cv2.imencode(
                        ".jpg",
                        warped,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                    )
                    if success_warp:
                        self.warp_broadcaster.update(encoded_warp.tobytes())
                except Exception as exc:
                    print(f"[Warp] WARN: warp render failed: {exc}")

            if save_path:
                try:
                    cv2.imwrite(save_path, vis_frame)
                except Exception as exc:
                    print(f"[VisWorker] WARN: failed to save {save_path}: {exc}")

            success, encoded = cv2.imencode(
                ".jpg",
                vis_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if success:
                self.broadcaster.update(encoded.tobytes())

            if self.show_window:
                preview = vis_frame
                if preview is not None:
                    preview = cv2.resize(preview, (500, 500))
                cv2.imshow("DepthAI TRT HTTP", preview)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._last_key = key

        if self.show_window:
            cv2.destroyAllWindows()


def build_handler(broadcaster: FrameBroadcaster, page_title: str):
    class StreamingHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self._handle_root()
            elif self.path == "/stream":
                self._handle_stream()
            elif self.path == "/snapshot":
                self._handle_snapshot()
            else:
                self.send_error(404, "Not Found")

        def _handle_root(self):
            html = (
                "<html><head>"
                f"<title>{page_title}</title>"
                "<style>body{font-family:sans-serif;background:#111;color:#eee;text-align:center}"
                "img{max-width:95vw;height:auto;border:1px solid #444}</style>"
                "<script>"
                "function saveFrame(){"
                "fetch('/snapshot').then(r=>{if(!r.ok)throw new Error('No frame');return r.blob();})"
                ".then(blob=>{const url=URL.createObjectURL(blob);"
                "const a=document.createElement('a');"
                "const ts=new Date().toISOString().replace(/[:.]/g,'-');"
                "a.href=url;a.download=`edge_frame_${ts}.jpg`;"
                "document.body.appendChild(a);a.click();a.remove();"
                "URL.revokeObjectURL(url);})"
                ".catch(err=>alert('프레임 저장 실패: '+err));}"
                "</script></head>"
                f"<body><h1>{page_title}</h1>"
                '<img src="/stream" alt="stream" />'
                '<p><button onclick="saveFrame()">현재 프레임 저장</button></p>'
                "</body></html>"
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_snapshot(self):
            frame = broadcaster.get_latest()
            if frame is None:
                self.send_error(503, "Frame not available")
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-cache, private")
            self.end_headers()
            self.wfile.write(frame)

        def _handle_stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            frame_id = -1
            while True:
                frame, frame_id = broadcaster.get_next(frame_id)
                if frame is None:
                    break

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break

        def log_message(self, format, *args):
            return

    return StreamingHandler


def guess_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip_addr = sock.getsockname()[0]
    except OSError:
        ip_addr = "127.0.0.1"
    finally:
        sock.close()
    return ip_addr


def render_vis_frame(frame_bgr: np.ndarray,
                     tri_records: List[dict],
                     dets,
                     scale_to_orig_x: float,
                     scale_to_orig_y: float,
                     infer_ms: float,
                     front_highlight: bool,
                     vehicle_cls_set: Optional[Set[int]],
                     front_color: Tuple[int, int, int]) -> np.ndarray:
    vis_frame = None
    if tri_records:
        vis_frame = draw_pred_pseudo3d(frame_bgr, tri_records)
    if vis_frame is None:
        vis_frame = overlay_detections(frame_bgr, dets, scale_to_orig_x, scale_to_orig_y)
    cv2.putText(
        vis_frame,
        f"Infer {infer_ms:.1f} ms | dets {len(dets)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if front_highlight:
        highlight_vehicle_front_faces(
            vis_frame,
            tri_records,
            vehicle_cls_set,
            front_color
        )
    return vis_frame


def highlight_vehicle_front_faces(vis_frame: np.ndarray,
                                  tri_records: List[dict],
                                  vehicle_cls_set: Optional[Set[int]],
                                  front_color: Tuple[int, int, int],
                                  height_scale: float = 0.5,
                                  min_dy: int = 8,
                                  max_dy: int = 80) -> None:
    if not tri_records:
        return
    for rec in tri_records:
        cls_raw = rec.get("class_id")
        if cls_raw is None:
            continue
        cls_val = int(cls_raw)
        if vehicle_cls_set is not None and cls_val not in vehicle_cls_set:
            continue
        tri = rec.get("tri")
        if tri is None:
            continue
        tri_np = np.asarray(tri, dtype=np.float32)
        if tri_np.shape != (3, 2):
            continue
        poly = parallelogram_from_triangle(tri_np[0], tri_np[1], tri_np[2]).astype(np.float32)
        base = np.round(poly).astype(np.int32)
        x_min, x_max = poly[:, 0].min(), poly[:, 0].max()
        y_min, y_max = poly[:, 1].min(), poly[:, 1].max()
        w = max(1.0, float(x_max - x_min))
        h = max(1.0, float(y_max - y_min))
        diag = float(np.hypot(w, h))
        off = int(np.clip(height_scale * diag, min_dy, max_dy))
        top = base.copy()
        top[:, 1] -= off
        front_poly = np.array([base[0], base[1], top[1], top[0]], dtype=np.int32)
        cv2.polylines(vis_frame, [front_poly], True, front_color, 2, cv2.LINE_AA)
        for idx in range(2):
            cv2.line(vis_frame, tuple(base[idx]), tuple(top[idx]), front_color, 2, cv2.LINE_AA)


def load_roi_mask(npz_path: str,
                  expect_hw: Tuple[int, int]) -> Tuple[np.ndarray, List[np.ndarray], Optional[dict]]:
    """
    Load ROI mask from .npz (fields: mask, optional polygon, meta).
    Resizes to expected H,W if needed.
    """
    H_exp, W_exp = expect_hw
    with np.load(npz_path, allow_pickle=True) as data:
        if "mask" not in data:
            raise ValueError("ROI npz missing 'mask' field")
        mask_raw = data["mask"]
        mask = np.asarray(mask_raw, dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape != (H_exp, W_exp):
            mask = cv2.resize(mask, (W_exp, H_exp), interpolation=cv2.INTER_NEAREST)
            print(f"[ROI] Resized mask from {mask_raw.shape} to {(H_exp, W_exp)}")
        polygon = None
        if "polygon" in data:
            polygon = np.asarray(data["polygon"], dtype=np.int32)
        meta = None
        if "meta" in data:
            try:
                meta = json.loads(str(data["meta"][0]))
            except Exception:
                meta = None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours, meta


def filter_dets_by_roi(dets,
                       roi_mask: Optional[np.ndarray],
                       scale_to_orig_x: float,
                       scale_to_orig_y: float) -> List[dict]:
    """Keep detections whose P0 lies inside roi_mask."""
    if roi_mask is None:
        return dets
    H, W = roi_mask.shape[:2]
    kept = []
    for det in dets:
        tri = det.get("tri")
        if tri is None:
            continue
        tri_np = np.asarray(tri, dtype=np.float32)
        if tri_np.shape != (3, 2):
            continue
        p0 = tri_np[0]
        x = int(round(float(p0[0] * scale_to_orig_x)))
        y = int(round(float(p0[1] * scale_to_orig_y)))
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        if roi_mask[y, x] > 0:
            kept.append(det)
    return kept


def load_world_points_from_json(json_path: str) -> Optional[np.ndarray]:
    """
    Load world_xy points from a homography JSON file.
    Expected schema:
    {
      "points": [
        {"world_xy": {"x": .., "y": ..}, ...},
        ...
      ]
    }
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[Warp] WARN: failed to read world points from {json_path}: {exc}")
        return None

    pts = []
    for item in data.get("points", []):
        world = item.get("world_xy")
        if not world:
            continue
        if not {"x", "y"} <= world.keys():
            continue
        try:
            pts.append([float(world["x"]), float(world["y"])])
        except Exception:
            continue
    if len(pts) < 3:
        print(f"[Warp] WARN: not enough world points in {json_path}")
        return None
    return np.asarray(pts, dtype=np.float64)


def build_warp_config(h_img2world: np.ndarray,
                      world_pts: np.ndarray,
                      target_w: Optional[int],
                      target_h: Optional[int],
                      rotate_deg: int):
    """
    Create warp homography (image->pixel topdown), mask, and rotation.
    Aspect ratio is preserved using world span; height/width are auto-adjusted if needed.
    """
    min_x = float(np.min(world_pts[:, 0]))
    max_x = float(np.max(world_pts[:, 0]))
    min_y = float(np.min(world_pts[:, 1]))
    max_y = float(np.max(world_pts[:, 1]))
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x <= 0.0 or span_y <= 0.0:
        raise ValueError("[Warp] Invalid world span (zero or negative)")

    world_ratio = span_x / span_y
    if target_w is None and target_h is None:
        # Default to a TV-like aspect using world ratio and ~37.6 px/m (matches 1889x1397 for CES points)
        default_scale = 37.6
        target_w = int(round(span_x * default_scale))
        target_h = int(round(span_y * default_scale))
    elif target_w is None:
        target_w = int(round(float(target_h) * world_ratio))
    elif target_h is None:
        target_h = int(round(float(target_w) / world_ratio))

    # Enforce aspect ratio; adjust height to match width-driven ratio if needed
    desired_ratio = target_w / float(target_h)
    if abs(desired_ratio - world_ratio) > 1e-3:
        adjusted_h = int(round(float(target_w) / world_ratio))
        print(
            f"[Warp] Adjusting height {target_h} -> {adjusted_h} to preserve aspect ratio "
            f"(world_ratio={world_ratio:.4f}, requested_ratio={desired_ratio:.4f})"
        )
        target_h = adjusted_h

    scale = target_w / span_x
    # World y is mapped so that larger y goes downward in the image (top-left origin)
    S_world2px = np.array([
        [scale, 0.0, -min_x * scale],
        [0.0, -scale, max_y * scale],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    H_out = S_world2px @ h_img2world

    # Build polygon mask in output pixel space
    poly_world = world_pts.reshape(-1, 1, 2).astype(np.float64)
    poly_px = cv2.perspectiveTransform(poly_world, S_world2px.astype(np.float64))
    poly_px = np.round(poly_px.reshape(-1, 2)).astype(np.int32)
    mask = np.zeros((int(target_h), int(target_w)), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_px], 255)
    mask_keep = mask > 0

    rotate_code = None
    if rotate_deg in (90, -270):
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotate_deg in (-90, 270):
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    elif rotate_deg in (180, -180):
        rotate_code = cv2.ROTATE_180
    elif rotate_deg not in (0,):
        raise ValueError(f"[Warp] Unsupported rotate_deg {rotate_deg}; use 0/90/-90/180")

    print(f"[Warp] Output size {target_w}x{target_h}, rotate={rotate_deg} deg")
    return {
        "H": H_out,
        "dsize": (int(target_w), int(target_h)),
        "mask_keep": mask_keep,
        "rotate_code": rotate_code,
    }


def run_live_inference_http(args) -> None:
    target_h, target_w = map(int, args.img_size.split(","))
    target_hw = (target_h, target_w)
    strides = [float(s) for s in args.strides.split(",")]
    jpeg_quality = int(np.clip(args.jpeg_quality, 1, 100))
    class_conf_map = None
    if args.class_conf:
        class_conf_map = {}
        for tok in args.class_conf.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                cid, thr = tok.split(":")
                class_conf_map[int(cid.strip())] = float(thr.strip())
            except Exception:
                print(f"[Warn] invalid --class-conf token '{tok}', expected format 'cls:thr'")

    runner = TensorRTTemporalRunner(
        args.weights,
        state_stride_hint=args.state_stride_hint,
        default_hidden_ch=args.default_hidden_ch
    )

    cam_w, cam_h = map(int, args.camera_size.split(","))
    roi_mask = None
    roi_meta = None
    if args.roi_path:
        try:
            roi_mask, _, roi_meta = load_roi_mask(args.roi_path, (cam_h, cam_w))
            print(f"[ROI] Loaded mask {roi_mask.shape} from {args.roi_path}")
            if roi_meta:
                print(f"[ROI] meta: {roi_meta}")
        except Exception as exc:
            print(f"[ROI] WARN: failed to load ROI '{args.roi_path}': {exc}")
            roi_mask = None

    save_dir = None
    if args.save_dir:
        save_dir = os.path.abspath(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)

    homography = load_homography_matrix(args.homography) if args.homography else None
    if args.udp_enable and homography is None:
        raise ValueError("UDP output requested but no homography (--homography) provided.")

    warp_cfg = None
    warp_broadcaster: Optional[FrameBroadcaster] = None
    warp_server: Optional[ThreadingHTTPServer] = None
    warp_server_thread: Optional[threading.Thread] = None
    if homography is None:
        raise ValueError("Warp stream requires homography (--homography).")
    world_json_path = args.warp_world_json or args.homography
    world_pts = None
    if world_json_path and str(world_json_path).lower().endswith(".json"):
        world_pts = load_world_points_from_json(world_json_path)
    if world_pts is None:
        raise ValueError("Warp stream requires world_xy points; provide --warp-world-json with that data.")
    warp_cfg = build_warp_config(
        homography,
        world_pts,
        None,
        None,
        args.warp_rotate
    )
    warp_broadcaster = FrameBroadcaster()
    warp_handler_cls = build_handler(warp_broadcaster, "Warp Stream")
    warp_server = ThreadingHTTPServer((args.http_host, args.warp_port), warp_handler_cls)
    warp_server_thread = threading.Thread(target=warp_server.serve_forever, daemon=True)
    warp_server_thread.start()
    bound_warp_ip = args.http_host if args.http_host != "0.0.0.0" else guess_local_ip()
    print(f"[Warp] Streaming warped MJPEG at http://{bound_warp_ip}:{args.warp_port}/ (Ctrl+C to stop)")

    udp_recorder: Optional[UDPRecorder] = None
    if args.record_udp:
        udp_recorder = UDPRecorder(args.record_udp)
        udp_recorder.start()
        print(f"[UDP] Recording payloads to {udp_recorder.record_path}")
    udp_sender = None
    if args.udp_enable:
        udp_sender = UDPSender(
            args.udp_host,
            args.udp_port,
            fmt=args.udp_format,
            fixed_length=args.udp_fixed_length,
            fixed_width=args.udp_fixed_width,
            on_send=udp_recorder.handle_send if udp_recorder is not None else None,
        )

    resize_mode = None
    if args.resize_mode is not None:
        mode_key = args.resize_mode.strip().upper()
        if mode_key not in ("", "NONE"):
            if not hasattr(dai.ImgResizeMode, mode_key):
                raise ValueError(f"Unsupported resize mode: {args.resize_mode}")
            resize_mode = getattr(dai.ImgResizeMode, mode_key)

    vehicle_cls_set: Optional[Set[int]] = {0}
    front_color = (0, 0, 255)

    broadcaster = FrameBroadcaster()
    handler_cls = build_handler(broadcaster, args.page_title)
    server = ThreadingHTTPServer((args.http_host, args.http_port), handler_cls)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    bound_ip = args.http_host if args.http_host != "0.0.0.0" else guess_local_ip()
    print(f"[HTTP] Streaming MJPEG at http://{bound_ip}:{args.http_port}/ (Ctrl+C to stop)")

    vis_worker: Optional[HttpVisualizationWorker] = None
    try:
        vis_worker = HttpVisualizationWorker(
            broadcaster,
            jpeg_quality,
            args.show_vis,
            args.front_highlight,
            vehicle_cls_set,
            front_color,
            warp_cfg=warp_cfg,
            warp_broadcaster=warp_broadcaster
        )
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
                    if class_conf_map:
                        def _keep(d):
                            cid = int(d.get("class_id", d.get("cls", 0)))
                            thr = class_conf_map.get(cid, args.conf)
                            return float(d.get("score", 0.0)) >= thr
                        dets = [d for d in dets if _keep(d)]
                    dets = tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)
                    dets = filter_dets_by_roi(
                        dets,
                        roi_mask,
                        prep["scale_to_orig_x"],
                        prep["scale_to_orig_y"]
                    )

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

                    vis_start = time.time()
                    vis_save_path = None
                    if save_dir is not None:
                        vis_save_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
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
                    vis_ms = (time.time() - vis_start) * 1000.0
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

                    key = vis_worker.get_key() if args.show_vis else -1
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        print("[Info] State reset requested.")
                        runner.reset()

            except KeyboardInterrupt:
                print("[Info] Keyboard interrupt received, stopping.")
    finally:
        if vis_worker is not None:
            vis_worker.stop()
            vis_worker.join(timeout=1.0)
        if warp_server is not None:
            warp_server.shutdown()
            warp_server.server_close()
        if warp_server_thread is not None:
            warp_server_thread.join(timeout=1.0)
        if warp_broadcaster is not None:
            warp_broadcaster.stop()
        broadcaster.stop()
        server.shutdown()
        server.server_close()
        if udp_sender is not None:
            udp_sender.close()
        if udp_recorder is not None:
            udp_recorder.stop()
            udp_recorder.join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser("DepthAI inference with HTTP visualization stream")
    parser.add_argument("--weights", type=str, required=True, help="TensorRT engine (.engine) path")
    parser.add_argument("--img-size", type=str, default="864,1536", help="Model input H,W")
    parser.add_argument("--camera-size", type=str, default="1536,864", help="DepthAI output width,height")
    parser.add_argument("--score-mode", type=str, default="obj*cls", choices=["obj", "cls", "obj*cls"])
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--class-conf", type=str, default=None,
                        help="Optional per-class conf threshold, e.g., '0:0.2,1:0.35,2:0.35'")
    parser.add_argument("--nms-iou", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--contain-thr", type=float, default=0.85)
    parser.add_argument("--clip-cells", type=float, default=None)
    parser.add_argument("--strides", type=str, default="8,16,32")
    parser.add_argument("--state-stride-hint", type=int, default=32)
    parser.add_argument("--default-hidden-ch", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Optional directory to save per-frame visualizations")
    parser.add_argument("--show-vis", dest="show_vis", action="store_true", default=False,
                        help="Also open a local OpenCV window for debugging")
    parser.add_argument("--no-show-vis", dest="show_vis", action="store_false")
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
    parser.add_argument("--udp-fixed-length", type=float, default=None,
                        help="Override vehicle length in UDP payloads (meters)")
    parser.add_argument("--udp-fixed-width", type=float, default=None,
                        help="Override vehicle width in UDP payloads (meters)")
    parser.add_argument("--roi-path", type=str, default=None,
                        help="Optional ROI npz (mask) path; only detections inside ROI (P0) are kept")
    parser.add_argument("--front-highlight", dest="front_highlight", action="store_true", default=True,
                        help="Highlight vehicle front faces in visualization (default: on)")
    parser.add_argument("--no-front-highlight", dest="front_highlight", action="store_false",
                        help="Disable vehicle front-face highlight")
    parser.add_argument("--http-host", type=str, default="0.0.0.0",
                        help="Interface/IP to bind the MJPEG server to")
    parser.add_argument("--http-port", type=int, default=8080,
                        help="Port to serve the MJPEG stream on")
    parser.add_argument("--page-title", type=str, default="DepthAI TensorRT Stream",
                        help="Title shown on the HTTP landing page")
    parser.add_argument("--warp-port", type=int, default=8081,
                        help="Port to serve the warped MJPEG stream on")
    parser.add_argument("--warp-rotate", type=int, default=0,
                        help="Rotate warped output by degrees (0, 90, -90, or 180)")
    parser.add_argument("--warp-world-json", type=str, default="utils/make_H/ces_real_img2world.json",
                        help="JSON with world_xy points; defaults to utils/make_H/ces_real_img2world.json (or --homography if that is a JSON file)")
    parser.add_argument("--jpeg-quality", type=int, default=85,
                        help="JPEG quality (1-100) used for streaming frames")
    parser.add_argument("--record-udp", type=str, default=None,
                        help="If set, write each UDP payload to this NDJSON file for offline replay")
    return parser.parse_args()


def main():
    args = parse_args()
    run_live_inference_http(args)


if __name__ == "__main__":
    main()
