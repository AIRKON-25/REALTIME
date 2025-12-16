#!/usr/bin/env python3

import argparse
import socket
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import depthai as dai


def parse_args():
    parser = argparse.ArgumentParser(description="Stream undistorted camera output over HTTP.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface/IP to bind the HTTP server to.")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve the MJPEG stream on.")
    parser.add_argument(
        "--fps",
        type=float,
        default=10,
        help="Camera capture FPS. Lower values reduce bandwidth and CPU usage.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save every encoded frame as JPEG.",
    )
    return parser.parse_args()


class FrameBroadcaster:
    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None
        self._frame_id = 0
        self._stopped = False

    def update(self, frame_bytes):
        with self._condition:
            self._frame = frame_bytes
            self._frame_id += 1
            self._condition.notify_all()

    def get_latest(self):
        with self._condition:
            return self._frame

    def get_next(self, last_frame_id):
        with self._condition:
            self._condition.wait_for(lambda: self._frame_id != last_frame_id or self._stopped)
            if self._stopped:
                return None, last_frame_id
            return self._frame, self._frame_id

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


def build_handler(broadcaster: FrameBroadcaster):
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
                "<html><head><title>OAK Stream</title>"
                "<script>"
                "function saveFrame(){"
                "fetch('/snapshot').then(r=>{if(!r.ok)throw new Error('No frame');return r.blob();})"
                ".then(blob=>{const url=URL.createObjectURL(blob);"
                "const a=document.createElement('a');"
                "const ts=new Date().toISOString().replace(/[:.]/g,'-');"
                "a.href=url;a.download=`frame_${ts}.jpg`;"
                "document.body.appendChild(a);a.click();a.remove();"
                "URL.revokeObjectURL(url);})"
                ".catch(err=>alert('프레임 저장 실패: '+err));}"
                "</script></head>"
                "<body><h1>Letterboxed Undistorted Stream</h1>"
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


def guess_local_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        sock.close()
    return ip


def main():
    args = parse_args()
    save_dir = args.save_dir
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving encoded frames to {save_dir.resolve()}")
    broadcaster = FrameBroadcaster()
    handler_cls = build_handler(broadcaster)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    bound_ip = args.host if args.host != "0.0.0.0" else guess_local_ip()
    print(f"Streaming MJPEG at http://{bound_ip}:{args.port}/ (Ctrl+C to stop)")

    try:
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            cam.setFps(float(args.fps))
            videoQueue = cam.requestOutput(
                (1536,864),
                resizeMode=dai.ImgResizeMode.LETTERBOX,
                enableUndistortion=True,
            ).createOutputQueue()

            pipeline.start()
            while pipeline.isRunning():
                videoIn = videoQueue.get()
                assert isinstance(videoIn, dai.ImgFrame)
                frame = videoIn.getCvFrame()
                success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if success:
                    frame_bytes = encoded.tobytes()
                    broadcaster.update(frame_bytes)
                    if save_dir:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        output_path = save_dir / f"frame_{timestamp}.jpg"
                        with open(output_path, "wb") as fh:
                            fh.write(frame_bytes)
    except KeyboardInterrupt:
        pass
    finally:
        broadcaster.stop()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
