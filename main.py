from realtime.v1.realtime_edge import IPCameraStreamerUltraLL as Streamer
from realtime.v1.batch_infer import BatchedTemporalRunner
from src.inference_lstm_onnx_pointcloud_add_color import (
    decode_predictions,
    tiny_filter_on_dets,
    tris_img_to_bev_by_lut,
    poly_from_tri,
    compute_bev_properties,
    draw_pred_only,
    draw_pred_pseudo3d,
)
from realtime_show_result.viz_utils import VizSizeConfig, prepare_visual_item
from pathlib import Path
from dataclasses import dataclass
import argparse, signal, time, threading, io
import onnxruntime as ort
import numpy as np
import cv2
import queue
from typing import Optional, Dict, List, Tuple
import socket, json
import os
from contextlib import contextmanager
import open3d as o3d
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

@dataclass
class CameraAssets:
    camera_id: int
    name: str
    lut: Optional[dict]
    lut_path: Optional[str]
    undistort_map: Optional[Tuple[np.ndarray, np.ndarray]]
    visible_cloud: Optional[Dict[str, np.ndarray]]
    visible_source: Optional[str]
    raw_config: Dict

COLOR_LABELS = ("red", "pink", "green", "white", "yellow", "purple")
_COLOR_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(COLOR_LABELS)}
_COLOR_RGB_RULES = {
    "purple": {
        "min": (120.0, 46.0, 184.0),
        "max": (174.0, 133.0, 237.0),
    },
    "white": {
        "min": (218.0, 200.0, 223.0),
        "max": (218.0, 200.0, 223.0),
    },
    "red": {
        "min": (72.0, 27.0, 20.0),
        "max": (197.0, 67.0, 49.0),
    },
    "pink": {
        "min": (199.0, 135.0, 192.0),
        "max": (237.0, 189.0, 239.0),
    },
    "green": {
        "min": (39.0, 196.0, 141.0),
        "max": (127.0, 229.0, 232.0),
    },
    "yellow": {
        "min": (218.0, 203.0, 74.0),
        "max": (218.0, 203.0, 74.0),
    },
}
def _hex_to_rgb_unit(hex_color: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not hex_color:
        return None
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        return None
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return None
    return r / 255.0, g / 255.0, b / 255.0


def _channel_score(value: float, low: float, high: float) -> float:
    if value < low or value > high:
        return 0.0
    span = max(high - low, 1e-3)
    mid = (low + high) * 0.5
    return max(0.0, 1.0 - abs(value - mid) / (0.5 * span))


def _classify_hex_color(hex_color: Optional[str]):
    rgb = _hex_to_rgb_unit(hex_color)
    if rgb is None:
        return None, 0.0, None

    # White: 모든 채널이 밝고 균일한 경우 우선 처리
    max_rgb = max(rgb)
    min_rgb = min(rgb)
    if min_rgb >= 0.7 and (max_rgb - min_rgb) <= 0.2:
        brightness = max(0.0, min(1.0, (sum(rgb) / 3.0 - 0.7) / 0.3))
        flatness = max(0.0, min(1.0, 1.0 - (max_rgb - min_rgb) / 0.2))
        confidence = float(0.5 * brightness + 0.5 * flatness)
        embedding = [0.0] * len(COLOR_LABELS)
        embedding[_COLOR_LABEL_TO_INDEX["white"]] = confidence
        return "white", confidence, embedding

    # RGB 범위 기반 색상 분류
    best_label = None
    best_score = 0.0
    intensity = max(0.0, min(1.0, (sum(rgb) / 3.0 - 0.15) / 0.85))
    for label, rule in _COLOR_RGB_RULES.items():
        mins = rule["min"]
        maxs = rule["max"]
        channel_scores = [_channel_score(v, lo, hi) for v, lo, hi in zip(rgb, mins, maxs)]
        base_score = sum(channel_scores) / 3.0
        score = float(base_score * 0.7 + intensity * 0.3)
        if score > best_score:
            best_score = score
            best_label = label
    if not best_label or best_score < 0.2:
        return None, float(best_score), None
    embedding = [0.0] * len(COLOR_LABELS)
    embedding[_COLOR_LABEL_TO_INDEX[best_label]] = best_score
    return best_label, float(best_score), embedding


def _classify_hex_color_strict(hex_color: Optional[str]):
    """
    RGB 범위 박스에 정확히 들어갈 때만 라벨을 반환한다.
    - 겹치거나 한 개도 매칭되지 않으면 (None, 0.0, None)
    - 기존 _classify_hex_color와 동일한 반환 형태를 유지한다.
    """
    rgb = _hex_to_rgb_unit(hex_color)
    if rgb is None:
        return None, 0.0, None

    max_rgb = max(rgb)
    min_rgb = min(rgb)
    candidates = []

    # 화이트는 밝고 편차가 작은 경우만 허용
    if min_rgb >= 0.7 and (max_rgb - min_rgb) <= 0.2:
        candidates.append("white")

    for label, rule in _COLOR_RGB_RULES.items():
        mins = rule["min"]
        maxs = rule["max"]
        if all(lo <= v <= hi for v, lo, hi in zip(rgb, mins, maxs)):
            candidates.append(label)

    if len(candidates) != 1:
        return None, 0.0, None

    label = candidates[0]
    embedding = [0.0] * len(COLOR_LABELS)
    embedding[_COLOR_LABEL_TO_INDEX[label]] = 1.0
    return label, 1.0, embedding

def _lut_mask(lut: Optional[dict]) -> Optional[np.ndarray]:
    if lut is None:
        return None
    for key in ("ground_valid_mask", "valid_mask", "floor_mask"):
        if key in lut:
            mask = np.asarray(lut[key]).astype(bool)
            if mask.shape == lut["X"].shape:
                return mask
    x = np.asarray(lut["X"])
    y = np.asarray(lut["Y"])
    z = np.asarray(lut["Z"])
    return np.isfinite(x) & np.isfinite(y) & np.isfinite(z)


def _build_visible_cloud(lut: Optional[dict]) -> Optional[Dict[str, np.ndarray]]:
    if lut is None:
        return None
    mask = _lut_mask(lut)
    if mask is None:
        return None
    X = np.asarray(lut["X"], dtype=np.float32)
    Y = np.asarray(lut["Y"], dtype=np.float32)
    Z = np.asarray(lut["Z"], dtype=np.float32)
    xyz = np.stack([X[mask], Y[mask], Z[mask]], axis=1)
    if xyz.size == 0:
        return None
    return {"xyz": xyz.astype(np.float32), "rgb": None}


def _load_visible_from_ply(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[EdgeInfer] WARN: visible ply missing -> {p}")
        return None
    try:
        cloud = o3d.io.read_point_cloud(str(p))
        if cloud.is_empty():
            print(f"[EdgeInfer] WARN: empty visible ply -> {p}")
            return None
        pts = np.asarray(cloud.points, dtype=np.float32)
        colors = None
        if cloud.has_colors():
            colors = np.asarray(cloud.colors, dtype=np.float32)
            if colors.shape != pts.shape:
                colors = None
        return {"xyz": pts.copy(), "rgb": colors.copy() if colors is not None else None}
    except Exception as exc:
        print(f"[EdgeInfer] WARN: failed to load visible ply {p}: {exc}")
        return None


def draw_detections(image: Optional[np.ndarray], dets, text_scale: float = 0.6):
    if image is None:
        return None
    vis = image.copy()
    for d in dets or []:
        tri = np.asarray(d.get("tri"), dtype=np.int32)
        if tri.shape != (3, 2):
            continue
        poly4 = poly_from_tri(tri).astype(np.int32)
        cv2.polylines(vis, [poly4], isClosed=True, color=(0, 255, 0), thickness=2)
        s = f"{d.get('score', 0.0):.2f}"
        p0 = (int(tri[0, 0]), int(tri[0, 1]))
        cv2.putText(vis, s, p0, cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def _lut_candidates(calib_dir: Optional[str], cam_id: int) -> List[Path]:
    if not calib_dir:
        return []
    root = Path(calib_dir)
    if not root.exists():
        return []
    stems = [f"cam{cam_id}", f"{cam_id}"]
    cands: List[Path] = []
    for stem in stems:
        cands += sorted(root.glob(f"{stem}_*.npz"))
        cands += sorted(root.glob(f"{stem}.npz"))
    return cands


def load_lut(explicit_path: Optional[str], calib_dir: Optional[str], cam_id: int) -> Tuple[Optional[dict], Optional[str]]:
    path = None
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            path = p
    if path is None:
        cands = _lut_candidates(calib_dir, cam_id)
        path = cands[0] if cands else None
    if path is None:
        print(f"[EdgeInfer] WARN: cam{cam_id} LUT not found")
        return None, None
    try:
        lut = dict(np.load(str(path), allow_pickle=False))
        if not all(k in lut for k in ("X", "Y", "Z")):
            raise ValueError("Missing XYZ keys")
        print(f"[EdgeInfer] cam{cam_id}: Using LUT {path}")
        return lut, str(path)
    except Exception as exc:
        print(f"[EdgeInfer] WARN: cam{cam_id} failed to load LUT {path}: {exc}")
        return None, str(path)


def _pick_matrix(data: dict, keys: List[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in data:
            arr = np.asarray(data[k])
            if arr.size:
                return arr
    return None


def load_undistort_map(desc, frame_hw: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not desc:
        return None
    data = None
    if isinstance(desc, (str, Path)):
        path = Path(desc)
        if not path.exists():
            print(f"[EdgeInfer] WARN: undistort file missing - {path}")
            return None
        try:
            data = dict(np.load(str(path), allow_pickle=False))
        except Exception as exc:
            print(f"[EdgeInfer] WARN: failed to load undistort npz {path}: {exc}")
            return None
    elif isinstance(desc, dict):
        data = desc
    else:
        return None

    K = _pick_matrix(data, ["K", "camera_matrix", "intrinsic", "intrinsics"])
    dist = _pick_matrix(data, ["dist", "distCoeffs", "dist_coeffs", "distortion", "D"])
    if K is None or dist is None:
        print("[EdgeInfer] WARN: undistort config missing K/dist")
        return None
    K = K.astype(np.float32).reshape(3, 3)
    dist = dist.astype(np.float32).ravel()
    newK = _pick_matrix(data, ["new_K", "K_new", "rectified_K"])
    if newK is not None:
        newK = newK.astype(np.float32).reshape(3, 3)
    else:
        newK = K

    height, width = frame_hw
    if "size" in data:
        size = data["size"]
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            width = int(size[0])
            height = int(size[1])
        elif isinstance(size, dict):
            width = int(size.get("width", width))
            height = int(size.get("height", height))
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, newK, (int(width), int(height)), cv2.CV_32FC1
    )
    return map1, map2
 

def load_camera_config_file(path: str, default_hw: Tuple[int, int]) -> List[Dict]:
    cfg_path = Path(path)
    raw = json.loads(cfg_path.read_text())
    entries = raw.get("cameras") if isinstance(raw, dict) and "cameras" in raw else raw
    if not isinstance(entries, list):
        raise ValueError("camera_config must be a list or contain 'cameras'")
    result = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        cid = item.get("camera_id", item.get("id"))
        if cid is None:
            raise ValueError("camera entry missing camera_id")
        cid = int(cid)
        rtsp = item.get("rtsp", {})
        ip = item.get("ip", rtsp.get("ip"))
        if not ip:
            raise ValueError(f"camera {cid} missing IP")
        port = int(item.get("port", rtsp.get("port", 554)))
        username = item.get("username", rtsp.get("username", "admin"))
        password = item.get("password", rtsp.get("password", ""))
        w = int(item.get("width", default_hw[1]))
        h = int(item.get("height", default_hw[0]))
        lut_rel = item.get("lut")
        undist_rel = item.get("undistort") or item.get("undistort_npz")
        visible_ply_rel = item.get("visible_ply") or item.get("visible_ply_path")
        cfg = {
            "ip": ip,
            "port": port,
            "username": username,
            "password": password,
            "camera_id": cid,
            "width": w,
            "height": h,
            "force_tcp": bool(item.get("force_tcp", False) or (rtsp.get("transport") == "tcp")),
            "lut_path": str((cfg_path.parent / lut_rel).resolve()) if lut_rel else None,
            "undistort_path": str((cfg_path.parent / undist_rel).resolve()) if undist_rel else None,
            "visible_ply": str((cfg_path.parent / visible_ply_rel).resolve()) if visible_ply_rel else None,
            "name": item.get("name") or item.get("label") or f"cam{cid}",
            "group": item.get("group"),
            "conf": item.get("conf"),
        }
        result.append(cfg)
    return result


def build_camera_assets(camera_cfgs: List[Dict], lut_root: Optional[str]) -> Dict[int, CameraAssets]:
    assets: Dict[int, CameraAssets] = {}
    for cfg in camera_cfgs:
        cid = int(cfg["camera_id"])
        lut, lut_path = load_lut(cfg.get("lut_path"), lut_root, cid)
        undist = load_undistort_map(cfg.get("undistort_path"), (cfg["height"], cfg["width"]))
        visible = None
        source_ply = None
        if cfg.get("visible_ply"):
            visible = _load_visible_from_ply(cfg["visible_ply"])
            if visible is not None:
                source_ply = cfg["visible_ply"]
        if visible is None:
            visible = _build_visible_cloud(lut)
        assets[cid] = CameraAssets(
            camera_id=cid,
            name=cfg.get("name", f"cam{cid}"),
            lut=lut,
            lut_path=lut_path,
            undistort_map=undist,
            visible_cloud=visible,
            visible_source=source_ply,
            raw_config=cfg,
        )
    return assets

# ---
class InferWorker(threading.Thread):
    def __init__(self, *, streamer, camera_assets: Dict[int, CameraAssets], img_hw, strides, onnx_path,
                 score_mode="obj*cls", conf={1:0.3}, nms_iou=0.2, topk=50,
                 bev_scale=1.0, providers=None,
                 gui_queue=None, udp_sender=None, web_publisher=None,
                 save_image_root=None):
        super().__init__(daemon=True)
        self.streamer = streamer
        self.camera_assets = camera_assets
        self.cam_ids = sorted(camera_assets.keys())
        self.H, self.W = img_hw
        self.strides = list(map(float, strides))
        self.score_mode = score_mode
        self.conf = conf
        self.nms_iou = float(nms_iou)
        self.topk = int(topk)
        self.gui_queue = gui_queue
        self.udp_sender = udp_sender
        self.web_publisher = web_publisher
        self.stop_evt = threading.Event()
        self.bev_scale = float(bev_scale)
        self.LUT_cache: Dict[int, Optional[dict]] = {cid: camera_assets[cid].lut for cid in self.cam_ids}
        self.save_dirs: Dict[str, Path] = {}
        if save_image_root:
            root = Path(save_image_root).expanduser().resolve()
            for tag in ("raw", "undist", "overlay"):
                tag_root = root / f"_{tag}"
                tag_root.mkdir(parents=True, exist_ok=True)
                self.save_dirs[tag] = tag_root

        if providers is None:
            providers = ["CUDAExecutionProvider","CPUExecutionProvider"] \
                if (ort.get_device().upper() == "GPU") else ["CPUExecutionProvider"]

        self.runner = BatchedTemporalRunner(
            onnx_path=onnx_path,
            cam_ids=self.cam_ids,
            img_size=(self.H, self.W),
            temporal="lstm", 
            providers=providers,
            state_stride_hint=32,
            default_hidden_ch=256,
        )
        for cid in self.cam_ids:
            if self.LUT_cache[cid] is None:
                print(f"[EdgeInfer] WARN: cam{cid} has no LUT - BEV/3D disabled")

    def _frame_basename(self, cam_id: int, capture_ts: Optional[float]) -> str:
        ts_val = capture_ts if capture_ts is not None else time.time()
        return f"cam{cam_id}_{int(ts_val * 1000):d}"

    def _save_image(self, img: Optional[np.ndarray], root: Optional[Path], cam_id: int,
                    capture_ts: Optional[float], tag: str):
        if root is None or img is None:
            return
        subdir = root / f"cam{cam_id}"
        subdir.mkdir(parents=True, exist_ok=True)
        name = f"{self._frame_basename(cam_id, capture_ts)}_{tag}.jpg"
        try:
            cv2.imwrite(str(subdir / name), img)
        except Exception as exc:
            print(f"[EdgeInfer] WARN: failed to save {tag} image for cam{cam_id}: {exc}")

    def _preprocess(self, cam_id: int, frame_bgr):
        assets = self.camera_assets.get(cam_id)
        if assets and assets.undistort_map is not None:
            map1, map2 = assets.undistort_map
            frame_bgr = cv2.remap(frame_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)
        orig_h, orig_w = frame_bgr.shape[:2]
        if (orig_h, orig_w) != (self.H, self.W):
            bgr = cv2.resize(frame_bgr, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        else:
            bgr = frame_bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2,0,1).astype(np.float32) / 255.0
        return chw, bgr, (orig_h, orig_w)

    def _decode(self, cam_id, outs):
        dets = decode_predictions(
            cam_id,
            outs, self.strides,
            clip_cells=None,
            conf_th=self.conf, nms_iou=self.nms_iou,
            topk=self.topk, score_mode=self.score_mode,
            use_gpu_nms=True
        )[0]
        return tiny_filter_on_dets(dets, min_area=20.0, min_edge=3.0)

    def _make_bev(self, dets, lut_data: Optional[dict], tris_img_orig=None, colors_hex=None) -> Tuple[List[dict], List[int]]:
        if lut_data is None or not dets:
            return [], []

        if tris_img_orig is not None:
            tris_img = np.asarray(tris_img_orig, dtype=np.float32)
        else:
            tris_img = np.asarray([d["tri"] for d in dets], dtype=np.float32)
        if tris_img.size == 0:
            return [], []

        tris_bev_xy, tris_bev_z, tri_ok = tris_img_to_bev_by_lut(
            tris_img, lut_data, bev_scale=self.bev_scale
        )

        bev: List[dict] = []
        valid_indices: List[int] = []

        color_list = colors_hex or []
        for idx, (d, tri_xy, tri_z, ok) in enumerate(zip(dets, tris_bev_xy, tris_bev_z, tri_ok)):
            if not ok or (not np.all(np.isfinite(tri_xy))):
                continue

            poly_bev = poly_from_tri(tri_xy)
            props = compute_bev_properties(tri_xy, tri_z)
            if props is None:
                continue
            center, length, width, yaw, front_edge, cz, pitch_deg, roll_deg = props

            entry = {
                "score":      float(d["score"]),
                "tri":        tri_xy,          
                "z3":         tri_z,          
                "poly":       poly_bev,
                "center":     center,
                "length":     float(length),
                "width":      float(width),
                "yaw":        float(yaw),
                "front_edge": front_edge,
                "cz":         float(cz),
                "pitch":      float(pitch_deg),
                "roll":       float(roll_deg),
            }
            if idx < len(color_list):
                color_hex = color_list[idx]
                entry["color_hex"] = color_hex
                label, confidence, embedding = _classify_hex_color_strict(color_hex) # _classify_hex_color로 둘중 뭐가낫나 색 none이너무많으면에바긴해 
                if label:
                    entry["color"] = label
                    entry["color_confidence"] = confidence
                if embedding is not None:
                    entry["color_embedding"] = embedding
            bev.append(entry)
            valid_indices.append(idx)
        return bev, valid_indices


    def run(self):
        wrk = StageTimer(name="WORKER", print_every=5.0)

        while not self.stop_evt.is_set():
            frame_meta: Dict[int, Dict] = {}
            # 1) 최신 프레임 수집
            with wrk.span("grab"):
                ready = True
                imgs_chw = {}
                for cid in self.cam_ids:
                    fr= self.streamer.get_latest(cid)
                    # ts_capture = time.time() 
                    if fr is None:
                        ready = False
                        break
                    with wrk.span("preproc"):
                        fr, ts_capture = fr
                        self._save_image(fr, self.save_dirs.get("raw"), cid, ts_capture, "raw")
                        chw, bgr, orig_hw = self._preprocess(cid, fr)
                    imgs_chw[cid] = chw
                    frame_meta[cid] = {
                        "bgr": bgr,
                        "orig_hw": orig_hw,
                        "capture_ts": ts_capture,
                    }

            if not ready:
                time.sleep(0.002)
                wrk.bump()
                continue

            # 2) 배치 주입
            with wrk.span("enqueue"):
                for cid, chw in imgs_chw.items():
                    self.runner.enqueue_frame(cid, chw)

            # 3) 배치 실행
            with wrk.span("infer"):
                per_cam_outs = self.runner.run_if_ready()
            if per_cam_outs is None:
                time.sleep(0.001)
                wrk.bump()
                continue

            # 4) 디코드/BEV/UDP/GUI 큐
            ts = time.time()
            for cid, outs in per_cam_outs.items():
                meta = frame_meta.get(cid)
                if meta is None:
                    continue
                capture_ts = meta.get("capture_ts")
                frame_bgr = meta.get("bgr")
                orig_hw = meta.get("orig_hw", (self.H, self.W))
                orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
                self._save_image(frame_bgr, self.save_dirs.get("undist"), cid, capture_ts, "undist")
                with wrk.span("decode"):
                    dets = self._decode(cid,outs)

                tris_img_orig = None
                colors_hex = None
                if dets and frame_bgr is not None:
                    tris_img_orig, colors_hex = draw_pred_only(
                        frame_bgr,
                        dets,
                        None,
                        None,
                        self.W,
                        self.H,
                        orig_w,
                        orig_h,
                        draw_visual=False,
                    )

                with wrk.span("bev"):
                    bev, bev_valid_indices = self._make_bev(
                        dets,
                        self.LUT_cache.get(cid),
                        tris_img_orig=tris_img_orig,
                        colors_hex=colors_hex,
                    )
                valid_indices = bev_valid_indices if bev_valid_indices else []
                filtered_dets = [dets[i] for i in valid_indices] if valid_indices else []

                overlay_frame = None
                if frame_bgr is not None and filtered_dets:
                    with wrk.span("draw"):
                        tris_for_gui = [
                            np.asarray(d.get("tri"), dtype=np.float32)
                            for d in filtered_dets
                            if d.get("tri") is not None
                        ]
                        pseudo3d = None
                        if tris_for_gui:
                            pseudo3d = draw_pred_pseudo3d(
                                frame_bgr,
                                tris_for_gui,
                                save_path_img=None,
                                dy=None,
                                height_scale=0.25,
                                min_dy=8,
                                max_dy=80,
                                return_image=True,
                            )
                        overlay_frame = pseudo3d
                if overlay_frame is None:
                    overlay_frame = draw_detections(frame_bgr, filtered_dets)
                self._save_image(overlay_frame, self.save_dirs.get("overlay"), cid, capture_ts, "overlay")

                if self.web_publisher is not None:
                    with wrk.span("web"):
                        try:
                            self.web_publisher.update(
                                cam_id=cid,
                                ts=ts,
                                capture_ts=capture_ts,
                                bev_dets=bev,
                                overlay_bgr=overlay_frame,
                            )
                        except Exception as e:
                            print(f"[WEB] publish error cam{cid}: {e}")

                if self.udp_sender is not None:
                    with wrk.span("udp"):
                        try:
                            self.udp_sender.send(cam_id=cid, ts=ts, bev_dets=bev, capture_ts=capture_ts)
                        except Exception as e:
                            print(f"[UDP] send error cam{cid}: {e}")

                if self.gui_queue is not None:
                    with wrk.span("gui.put"):
                        try:
                            self.gui_queue.put_nowait((cid, overlay_frame, capture_ts))
                        except queue.Full:
                            try:
                                _ = self.gui_queue.get_nowait()
                                self.gui_queue.put_nowait((cid, overlay_frame, capture_ts))
                            except Exception:
                                pass

            wrk.bump()

    def stop(self):
        self.stop_evt.set()

class UDPSender:
    def __init__(
        self,
        host: str,
        port: int,
        fmt: str = "json",
        max_bytes: int = 65000,
        fixed_length: Optional[float] = None,
        fixed_width: Optional[float] = None,
    ):
        self.addr = (host, int(port))
        self.fmt = fmt
        self.max_bytes = max_bytes
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.fixed_length = float(fixed_length) if fixed_length is not None else None
        self.fixed_width = float(fixed_width) if fixed_width is not None else None

    def close(self):
        try: self.sock.close()
        except: pass

    def _pack_json(self, cam_id: int, ts: float, bev_dets, capture_ts: Optional[float]):
        items = []
        for d in bev_dets or []:
            length = self.fixed_length if self.fixed_length is not None else d["length"]
            width = self.fixed_width if self.fixed_width is not None else d["width"]
            item = {
                "center": [float(d["center"][0]), float(d["center"][1])],
                "length": float(length),
                "width": float(width),
                "yaw": float(d["yaw"]),
                "score": float(d["score"]),
                "cz":     float(d.get("cz", 0.0)),
                "pitch":  float(d.get("pitch", 0.0)),
                "roll":   float(d.get("roll", 0.0)),
            }
            color_hex = d.get("color_hex")
            if color_hex:
                item["color_hex"] = color_hex
            color_label = d.get("color")
            if color_label:
                item["color"] = color_label
            color_conf = d.get("color_confidence")
            if color_conf is not None:
                item["color_confidence"] = float(color_conf)
            items.append(item)
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
        payload = None
        if self.fmt == "json":
            payload = self._pack_json(cam_id, ts, bev_list, capture_ts)
        else:
            raise ValueError(f"Unsupported UDP payload fmt: {self.fmt}")
        #print(payload)
        if len(payload) <= self.max_bytes:
            self.sock.sendto(payload, self.addr)
            return

        chunk_id = os.urandom(4).hex()
        total = (len(payload) + self.max_bytes - 1) // self.max_bytes
        for idx in range(total):
            part = payload[idx*self.max_bytes:(idx+1)*self.max_bytes]
            prefix = f"CHUNK {chunk_id} {total} {idx}\n".encode("utf-8")
            self.sock.sendto(prefix + part, self.addr)
        # print(payload)

class EdgeWebBridge:
    def __init__(self, *, host: str, port: int, global_ply: str, vehicle_glb: str,
                 camera_assets: Dict[int, CameraAssets], site_name: str = "edge",
                 jpeg_quality: int = 85, static_root: Optional[str] = None,
                 viz_config: VizSizeConfig):
        self.FastAPI = FastAPI
        self.HTTPException = HTTPException
        self.Response = Response
        self.FileResponse = FileResponse
        self.JSONResponse = JSONResponse
        self.uvicorn = uvicorn
        self.host = host
        self.port = int(port)
        self.site_name = site_name
        self.jpeg_quality = int(jpeg_quality)
        self.global_ply = str(Path(global_ply).resolve())
        self.vehicle_glb = str(Path(vehicle_glb).resolve())
        self.camera_assets = camera_assets
        self.started_at = time.time()
        self.viz_cfg = viz_config
        self._lock = threading.Lock()
        self._latest: Dict[int, Dict] = {
            cid: {"timestamp": None, "capture_ts": None, "detections": [], "overlay": None}
            for cid in camera_assets.keys()
        }
        self._visible_bytes: Dict[int, Optional[bytes]] = {}
        self._visible_arrays: Dict[int, Optional[np.ndarray]] = {}
        self._visible_meta: Dict[int, Optional[dict]] = {}
        for cid, asset in camera_assets.items():
            if asset.visible_cloud is None:
                self._visible_bytes[cid] = None
                self._visible_arrays[cid] = None
                self._visible_meta[cid] = None
                continue
            xyz = np.asarray(asset.visible_cloud.get("xyz"), dtype=np.float32)
            rgb = asset.visible_cloud.get("rgb")
            rgb = np.asarray(rgb, dtype=np.float32) if rgb is not None else None
            buf = io.BytesIO()
            if rgb is not None:
                np.savez_compressed(buf, xyz=xyz, rgb=rgb)
            else:
                np.savez_compressed(buf, xyz=xyz)
            self._visible_bytes[cid] = buf.getvalue()
            if rgb is not None and rgb.shape == xyz.shape:
                arr32 = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
                meta = {"count": int(xyz.shape[0]), "has_rgb": True, "stride": 6}
            else:
                arr32 = xyz.astype(np.float32)
                meta = {"count": int(xyz.shape[0]), "has_rgb": False, "stride": 3}
            self._visible_arrays[cid] = arr32
            self._visible_meta[cid] = meta
        self.static_root = Path(static_root) if static_root else (Path(__file__).resolve().parent / "realtime_show_result" / "static")
        self.index_html = None
        self.app = self.FastAPI(title="Edge Web Bridge", version="0.1.0")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        if self.static_root.exists():
            self.index_html = self.static_root / "live.html"
            self.app.mount("/static", StaticFiles(directory=str(self.static_root), html=True), name="static")
        self._configure_routes()
        self._server = None
        self._thread = None

    def _configure_routes(self):
        @self.app.get("/")
        def _root():
            if self.index_html and self.index_html.exists():
                return self.FileResponse(str(self.index_html))
            return {"status": "ok", "message": "Edge Web Bridge"}

        @self.app.get("/healthz")
        def _health():
            return {"ok": True, "since": self.started_at}

        @self.app.get("/api/site")
        def _site():
            return {
                "site": self.site_name,
                "started_at": self.started_at,
                "cameras": list(self.camera_assets.keys()),
                "config": self.viz_cfg.as_client_dict(),
            }

        @self.app.get("/api/cameras")
        def _cameras():
            return [
                {
                    "camera_id": cid,
                    "name": asset.name,
                    "lut": asset.lut_path,
                    "resolution": [asset.raw_config.get("width"), asset.raw_config.get("height")],
                }
                for cid, asset in sorted(self.camera_assets.items())
            ]

        @self.app.get("/api/cameras/{camera_id}/detections")
        def _detections(camera_id: int):
            entry = self._latest.get(camera_id)
            if entry is None:
                raise self.HTTPException(status_code=404, detail="unknown camera id")
            payload = {k: v for k, v in entry.items() if k != "overlay"}
            return self.JSONResponse(content=self._safe_json(payload))

        @self.app.get("/api/cameras/{camera_id}/overlay.jpg")
        def _overlay(camera_id: int):
            entry = self._latest.get(camera_id)
            if entry is None or entry.get("overlay") is None:
                raise self.HTTPException(status_code=404, detail="overlay not ready")
            return self.Response(content=entry["overlay"], media_type="image/jpeg")

        @self.app.get("/api/cameras/{camera_id}/visible-cloud.npz")
        def _visible(camera_id: int):
            data = self._visible_bytes.get(camera_id)
            if data is None:
                raise self.HTTPException(status_code=404, detail="visible cloud missing")
            return self.Response(content=data, media_type="application/octet-stream")

        @self.app.get("/api/cameras/{camera_id}/visible-meta")
        def _visible_meta(camera_id: int):
            meta = self._visible_meta.get(camera_id)
            if meta is None:
                raise self.HTTPException(status_code=404, detail="visible cloud missing")
            return self.JSONResponse(content=self._safe_json(meta))

        @self.app.get("/api/cameras/{camera_id}/visible.bin")
        def _visible_bin(camera_id: int):
            arr = self._visible_arrays.get(camera_id)
            if arr is None:
                raise self.HTTPException(status_code=404, detail="visible cloud missing")
            return self.Response(
                content=arr.tobytes(),
                media_type="application/octet-stream",
                headers={"X-Point-Count": str(arr.shape[0])},
            )

        @self.app.get("/assets/global.ply")
        def _global():
            if not os.path.exists(self.global_ply):
                raise self.HTTPException(status_code=404, detail="global PLY missing")
            return self.FileResponse(self.global_ply, filename="global.ply")

        @self.app.get("/assets/vehicle.glb")
        def _vehicle():
            if not os.path.exists(self.vehicle_glb):
                raise self.HTTPException(status_code=404, detail="vehicle mesh missing")
            return self.FileResponse(self.vehicle_glb, filename="vehicle.glb")

    def _serialize_bev(self, bev_dets):
        items = []
        for d in bev_dets or []:
            center = d.get("center")
            if center is None or len(center) < 2:
                continue
            vis = prepare_visual_item(
                class_id=int(d.get("class_id", 0)),
                cx=float(center[0]),
                cy=float(center[1]),
                cz=float(d.get("cz", 0.0)),
                length=float(d.get("length", 0.0)),
                width=float(d.get("width", 0.0)),
                yaw_deg=float(d.get("yaw", 0.0)),
                pitch_deg=float(d.get("pitch", 0.0)),
                roll_deg=float(d.get("roll", 0.0)),
                cfg=self.viz_cfg,
                score=d.get("score"),
            )
            tri = d.get("tri")
            if hasattr(tri, "tolist"):
                tri = tri.tolist()
            front_edge = d.get("front_edge")
            if hasattr(front_edge, "tolist"):
                front_edge = front_edge.tolist()
            payload = {
                **vis,
                "yaw": float(d.get("yaw", 0.0)),
                "pitch": float(d.get("pitch", 0.0)),
                "roll": float(d.get("roll", 0.0)),
                "front_edge": front_edge,
                "tri": tri,
            }
            color_hex = d.get("color_hex")
            if color_hex:
                payload["color_hex"] = color_hex
            color_label = d.get("color")
            if color_label:
                payload["color"] = color_label
            color_embedding = d.get("color_embedding")
            if color_embedding is not None:
                payload["color_embedding"] = color_embedding
            color_conf = d.get("color_confidence")
            if color_conf is not None:
                payload["color_confidence"] = float(color_conf)
            items.append(payload)
        return items

    def _safe_json(self, obj):
        if isinstance(obj, dict):
            return {k: self._safe_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._safe_json(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    def update(self, cam_id: int, ts: float, capture_ts: float, bev_dets, overlay_bgr: Optional[np.ndarray]):
        overlay_bytes = None
        if overlay_bgr is not None:
            ok, buf = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                overlay_bytes = buf.tobytes()
        payload = {
            "timestamp": ts,
            "capture_ts": capture_ts,
            "detections": self._serialize_bev(bev_dets),
        }
        with self._lock:
            state = self._latest.setdefault(cam_id, {})
            state.update(payload)
            if overlay_bytes is not None:
                state["overlay"] = overlay_bytes

    def start(self):
        if self._server is not None:
            return
        config = self.uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info", access_log=False)
        self._server = self.uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        print(f"[WEB] serving http://{self.host}:{self.port}")

    def stop(self):
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

class FPSTicker:
    def __init__(self, target_fps=30, print_every=5.0, name="GUI"):
        self.target_fps = max(1, int(target_fps))
        self.spf = 1.0 / self.target_fps  # seconds per frame
        self.next_deadline = time.time()
        self.running = True
        self._name = str(name)
        self._cnt = 0
        self._t0 = time.time()
        self._print_every = print_every
        
    def set_fps(self, target_fps: int):
        self.target_fps = max(1, int(target_fps))
        self.spf = 1.0 / self.target_fps

    def tick(self):
        if not self.running:
            return False

        now = time.time()

        if now < self.next_deadline:
            time.sleep(self.next_deadline - now)
        self.next_deadline = time.time() + self.spf

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[FPSTicker] 'q' pressed ??stopping.")
            self.running = False
            return False
        
        if self._print_every > 0:
            t = time.time()
            if (t - self._t0) >= self._print_every:
                fps_out = self._cnt / (t - self._t0)
                print(f"[FPSTicker-{self._name}] render fps = {fps_out:.2f} (target={self.target_fps})")
                self._cnt = 0
                self._t0 = t

        return True

class StageTimer:
    def __init__(self, name="prof", print_every=5.0):
        self.name = name
        self.print_every = float(print_every)
        self.t0 = time.perf_counter()
        self.last_print = self.t0
        self.sum = {} 
        self.cnt = {}  
        self.tmp = {} 

    @contextmanager
    def span(self, stage: str):
        t = time.perf_counter()
        self.tmp[stage] = t
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            self.sum[stage] = self.sum.get(stage, 0.0) + dt
            self.cnt[stage] = self.cnt.get(stage, 0) + 1

    def bump(self):
        now = time.perf_counter()
        if (now - self.last_print) >= self.print_every:
            parts = []
            for k in sorted(self.sum.keys()):
                s = self.sum[k]
                n = max(1, self.cnt.get(k, 1))
                parts.append(f"{k}={1000.0*s/n:.2f}ms")
                self.sum[k] = 0.0
                self.cnt[k] = 0
            if parts:
                print(f"[{self.name}] " + " | ".join(parts))
            self.last_print = now

def main():
    ap = argparse.ArgumentParser("Realtime Edge Infer")
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--img-size", default="864,1536", type=str)
    ap.add_argument("--strides", default="8,16,32", type=str)
    ap.add_argument("--conf", default=0.3, type=float)
    ap.add_argument("--nms-iou", default=0.20, type=float)
    ap.add_argument("--topk", default=50, type=int)
    ap.add_argument("--score-mode", default="obj*cls", choices=["obj","cls","obj*cls"])
    ap.add_argument("--bev-scale", default=1.0, type=float)
    ap.add_argument("--lut-path", default="./calib", type=str)
    ap.add_argument("--lut-dir", dest="lut_path", type=str)
    ap.add_argument("--camera-config", type=str, default ="camera_config.json")
    ap.add_argument("--transport", default="tcp", choices=["tcp","udp"])
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--udp-enable", action="store_true")
    ap.add_argument("--udp-host", default="192.168.0.165")
    ap.add_argument("--udp-port", type=int, default=50050)
    ap.add_argument("--udp-format", choices=["json","text"], default="json")
    ap.add_argument("--udp-fixed-length", type=float, default=4.4,
                    help="Override the length value in UDP payloads when set")
    ap.add_argument("--udp-fixed-width", type=float, default=2.7,
                    help="Override the width value in UDP payloads when set")
    ap.add_argument("--visual-size", default="216,384", type=str)
    ap.add_argument("--target-fps", default=30, type=int)
    ap.add_argument("--no-gui", action="store_true",
                    help="Disable OpenCV visualization windows")
    ap.add_argument("--save-image-root", type=str, default=None,
                    help="Set root dir to save frames; creates _raw/_undist/_overlay subfolders")
    # visualization scaling (shared across web/3d)
    ap.add_argument("--size-mode", choices=["bbox","fixed","mesh"], default="mesh")
    ap.add_argument("--fixed-length", type=float, default=4.4)
    ap.add_argument("--fixed-width", type=float, default=2.7)
    ap.add_argument("--height-scale", type=float, default=0.5)
    ap.add_argument("--mesh-scale", type=float, default=1.0)
    ap.add_argument("--mesh-height", type=float, default=0.0)
    ap.add_argument("--z-offset", type=float, default=0.0)
    ap.add_argument("--invert-bev-y", dest="invert_bev_y", action="store_true")
    ap.add_argument("--no-invert-bev-y", dest="invert_bev_y", action="store_false")
    ap.set_defaults(no_invert_bev_y=True)
    ap.add_argument("--normalize-vehicle", dest="normalize_vehicle", action="store_true")
    ap.add_argument("--no-normalize-vehicle", dest="normalize_vehicle", action="store_false")
    ap.set_defaults(normalize_vehicle=True)
    ap.add_argument("--vehicle-y-up", dest="vehicle_y_up", action="store_true")
    ap.add_argument("--vehicle-z-up", dest="vehicle_y_up", action="store_false")
    ap.set_defaults(vehicle_y_up=True)
    # web bridge
    ap.add_argument("--web-enable", action="store_true")
    ap.add_argument("--web-host", default="0.0.0.0")
    ap.add_argument("--web-port", type=int, default=10000)
    ap.add_argument("--web-site-name", default="edge-site")
    ap.add_argument("--web-jpeg-quality", type=int, default=85)
    ap.add_argument("--global-ply", type=str, default="pointcloud/merged_05.ply")
    ap.add_argument("--vehicle-glb", type=str, default="pointcloud/car.glb")
    args = ap.parse_args()

    # GUI 
    def gui_available() -> bool:
        try:
            cv2.namedWindow("TEST"); cv2.destroyWindow("TEST")
            return True
        except Exception:
            return False
    USE_GUI = gui_available()
    if args.no_gui:
        USE_GUI = False

    # 웹
    viz_cfg = VizSizeConfig(
        size_mode=args.size_mode,
        fixed_length=args.fixed_length,
        fixed_width=args.fixed_width,
        height_scale=args.height_scale,
        mesh_scale=args.mesh_scale,
        mesh_height=args.mesh_height,
        z_offset=args.z_offset,
        invert_bev_y=args.invert_bev_y,
        normalize_vehicle=args.normalize_vehicle,
        vehicle_y_up=args.vehicle_y_up,
    )
    
    H, W = map(int, args.img_size.split(","))
    strides = tuple(float(s) for s in args.strides.split(","))
    vis_H, vis_W = map(int, args.visual_size.split(","))
    Target_fps = args.target_fps
    
    camera_configs: List[Dict] = []
    if args.camera_config:
        try:
            camera_configs = load_camera_config_file(args.camera_config, (H, W))
        except Exception as exc:
            raise SystemExit(f"[Main] failed to parse camera_config: {exc}")
    if not camera_configs:
        raise SystemExit("[Main] no cameras configured")

    conf_dict = {
        int(cfg["camera_id"]): float(cfg["conf"]) if cfg.get("conf") is not None else float(args.conf)
        for cfg in camera_configs
    }

    cam_cfg_map = {int(cfg["camera_id"]): cfg for cfg in camera_configs} 
    camera_assets = build_camera_assets(camera_configs, args.lut_path)
    for cid, asset in camera_assets.items():
        cfg = asset.raw_config
        visible_info = asset.visible_source if asset.visible_source else "lut-derived"
        #print(f"[Main] slot cam{cid} ({asset.name}): rtsp={cfg['ip']}:{cfg['port']} size={cfg['width']}x{cfg['height']} LUT={asset.lut_path} visible={visible_info}")
    
    cam_ids  = sorted(camera_assets.keys())
    shutdown_evt = threading.Event()
    streamer = Streamer(
        camera_configs,
        show_windows=False,
        target_fps=Target_fps,
        snapshot_dir=None,
        snapshot_interval_sec=None,
        catchup_seconds=0.5,
        overlay_ts=False,
        laytency_check=False
    )
    
    providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
    if args.no_cuda or ort.get_device().upper() != "GPU":
        providers = ["CPUExecutionProvider"]
    
    udp_sender = None
    if args.udp_enable:
        udp_sender = UDPSender(
            args.udp_host,
            args.udp_port,
            fmt=args.udp_format,
            fixed_length=args.udp_fixed_length,
            fixed_width=args.udp_fixed_width,
        )

    web_bridge = None
    if args.web_enable:
        try:
            web_bridge = EdgeWebBridge(
                host=args.web_host,
                port=args.web_port,
                global_ply=args.global_ply,
                vehicle_glb=args.vehicle_glb,
                camera_assets=camera_assets,
                site_name=args.web_site_name,
                jpeg_quality=args.web_jpeg_quality,
                viz_config=viz_cfg,
            )
            web_bridge.start()
        except Exception as e:
            print(f"[WEB] disabled: {e}")
            web_bridge = None

    streamer.start()
    gui_queue = queue.Queue(maxsize=128)
    worker = InferWorker(
        streamer=streamer,
        camera_assets=camera_assets,
        img_hw=(H, W),
        strides=strides,
        onnx_path=args.weights,
        score_mode=args.score_mode,
        conf=conf_dict,
        nms_iou=args.nms_iou,
        topk=args.topk,
        bev_scale=args.bev_scale,
        providers=providers,
        gui_queue=gui_queue,
        udp_sender=udp_sender,
        web_publisher=web_bridge,
        save_image_root=args.save_image_root,
    )
    worker.start()
    if USE_GUI:
        for cid in cam_ids:
            cv2.namedWindow(f"cam{cid}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"cam{cid}", vis_W, vis_H)

    running = True
    def _sigint(sig, frm):
        try:
            ticker.running = False
        except Exception:
            pass
        shutdown_evt.set()
        try:
            if USE_GUI:
                cv2.waitKey(1)
        except Exception:
            pass

    signal.signal(signal.SIGINT, _sigint)
    ticker = FPSTicker(target_fps=Target_fps)
    gui_timer = StageTimer(name="GUI", print_every=5.0)

    try:
        while ticker.running:
            with gui_timer.span("gui.frame"):
                with gui_timer.span("gui.get"):
                    try:
                        cid, overlay_bgr, ts_capture = gui_queue.get(timeout=0.005) 
                    except queue.Empty:
                        if USE_GUI: cv2.waitKey(1)
                        if not ticker.tick():
                            break
                        gui_timer.bump()
                        continue


                if not USE_GUI:
                    if not ticker.tick():
                        break
                    gui_timer.bump()
                    continue
                vis = overlay_bgr
                if vis is None:
                    vis = np.zeros((H, W, 3), dtype=np.uint8)

                with gui_timer.span("gui.show"):
                    cv2.imshow(f"cam{cid}", vis)
                    e2e_ms = (time.time() - ts_capture) * 1000.0
                    print(f"+++++++++++++프레임받아서시각화까지: {e2e_ms}")
                    if not ticker.tick():
                        break
            gui_timer.bump()
    finally:
        worker.stop()
        worker.join(timeout=1)
        streamer.stop()
        if udp_sender: udp_sender.close()
        if web_bridge:
            web_bridge.stop()
        if USE_GUI:
            try: cv2.destroyAllWindows()
            except: pass
        print("[Main] stopped.")


if __name__ == "__main__":
    main()
