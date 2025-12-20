import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def slugify_label(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", str(text)).strip("-").lower()
    return slug or "cam"


def dedup_slug(base: str, used: set) -> str:
    slug = base
    idx = 2
    while slug in used:
        slug = f"{base}-{idx}"
        idx += 1
    used.add(slug)
    return slug


def filter_candidates_by_cam_id(candidates, cam_id: Optional[int]):
    if cam_id is None:
        return candidates
    pattern = re.compile(rf"^cam_?{cam_id}(?!\d)", re.IGNORECASE)
    filtered = [c for c in candidates if pattern.search(c.name)]
    return filtered or candidates


def guess_local_ply(root: Path, cam_id: Optional[int], name: Optional[str]) -> Optional[Path]:
    if cam_id is None:
        return None
    candidates = []
    patterns = [
        f"cam_{cam_id}_*.ply",
        f"cam{cam_id}_*.ply",
    ]
    if name:
        name_slug = slugify_label(name).replace("-", "_")
        patterns.append(f"{name_slug}*.ply")
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    if not candidates:
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    candidates = filter_candidates_by_cam_id(candidates, cam_id)
    candidates.sort()
    return candidates[0].resolve()


def guess_local_lut(root: Path, cam_id: Optional[int], name: Optional[str]) -> Optional[Path]:
    if cam_id is None:
        return None
    candidates = []
    patterns = [
        f"cam_{cam_id}_*.npz",
        f"cam{cam_id}_*.npz",
    ]
    if name:
        name_slug = slugify_label(name).replace("-", "_")
        patterns.append(f"{name_slug}*.npz")
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    if not candidates:
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    candidates = filter_candidates_by_cam_id(candidates, cam_id)
    candidates.sort()
    return candidates[0].resolve()


def _lut_mask_from_obj(obj: dict) -> Optional[np.ndarray]:
    for key in ("ground_valid_mask", "valid_mask", "floor_mask"):
        if key in obj:
            mask = np.asarray(obj[key]).astype(bool)
            if "X" in obj and mask.shape == np.asarray(obj["X"]).shape:
                return mask
    if all(k in obj for k in ("X", "Y", "Z")):
        X = np.asarray(obj["X"])
        Y = np.asarray(obj["Y"])
        Z = np.asarray(obj["Z"])
        return np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    return None


def load_visible_xyz_from_lut(path: Path) -> Optional[Tuple[np.ndarray, bool]]:
    try:
        with np.load(str(path)) as data:
            if not {"X", "Y", "Z"}.issubset(set(data.files)):
                print(f"[Fusion] LUT missing XYZ -> {path}")
                return None
            mask = _lut_mask_from_obj(data)
            if mask is None:
                print(f"[Fusion] LUT mask missing -> {path}")
                return None
            X = np.asarray(data["X"], dtype=np.float32)
            Y = np.asarray(data["Y"], dtype=np.float32)
            Z = np.asarray(data["Z"], dtype=np.float32)
            xyz = np.stack([X[mask], Y[mask], Z[mask]], axis=1).astype(np.float32)
            colors = None
            has_rgb = False
            if {"R", "G", "B"}.issubset(set(data.files)):
                R = np.asarray(data["R"], dtype=np.float32)
                G = np.asarray(data["G"], dtype=np.float32)
                B = np.asarray(data["B"], dtype=np.float32)
                try:
                    colors = np.stack([R[mask], G[mask], B[mask]], axis=1).astype(np.float32)
                    max_val = float(np.nanmax(colors)) if colors.size else 0.0
                    if max_val > 1.01:
                        colors /= 255.0
                    has_rgb = True
                except Exception as exc:  # pragma: no cover
                    print(f"[Fusion] LUT RGB load failed for {path}: {exc}")
                    colors = None
                    has_rgb = False
    except Exception as exc:
        print(f"[Fusion] failed to load LUT {path}: {exc}")
        return None
    if xyz.size == 0:
        return None
    if colors is not None and colors.shape[0] == xyz.shape[0]:
        pts = np.concatenate([xyz, colors], axis=1)
        return pts, True
    return xyz, False


def load_camera_markers(path: Optional[str], local_ply_root: Optional[str] = None,
                        local_lut_root: Optional[str] = None) -> List[dict]:
    markers: List[dict] = []
    if not path:
        return markers
    json_path = Path(path)
    if not json_path.exists():
        print(f"[Fusion] camera position file not found: {json_path}")
        return markers
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Fusion] failed to read camera position file {json_path}: {exc}")
        return markers

    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        cams = raw.get("cameras")
        if isinstance(cams, list):
            entries = cams
        else:
            entries = [raw]
    else:
        print(f"[Fusion] camera position file {json_path} has unexpected format")
        return markers

    base_dir = json_path.parent
    ply_root = Path(local_ply_root).resolve() if local_ply_root else None
    lut_root = Path(local_lut_root).resolve() if local_lut_root else None
    used_slugs = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("label")
        camera_id = entry.get("camera_id", entry.get("id"))
        if camera_id is not None:
            try:
                camera_id = int(camera_id)
            except (TypeError, ValueError):
                camera_id = None
        if camera_id is None and name:
            nums = re.findall(r"\d+", str(name))
            if nums:
                try:
                    camera_id = int(nums[-1])
                except ValueError:
                    camera_id = None

        pos_src = entry.get("pos") if isinstance(entry.get("pos"), dict) else {}
        x = safe_float(pos_src.get("x", entry.get("x")))
        y = safe_float(pos_src.get("y", entry.get("y")))
        if x is None or y is None:
            continue
        z = safe_float(pos_src.get("z", entry.get("z")), 0.0)

        rot_src = entry.get("rot") if isinstance(entry.get("rot"), dict) else {}
        rotation = {
            "pitch": safe_float(rot_src.get("pitch", entry.get("pitch")), 0.0),
            "yaw": safe_float(rot_src.get("yaw", entry.get("yaw")), 0.0),
            "roll": safe_float(rot_src.get("roll", entry.get("roll")), 0.0),
        }

        local_ref = entry.get("local_ply") or entry.get("visible_ply")
        ply_path = None
        if isinstance(local_ref, str) and local_ref:
            candidate = Path(local_ref)
            ply_path = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        if (ply_path is None or not ply_path.exists()) and ply_root and ply_root.exists():
            guessed = guess_local_ply(ply_root, camera_id, name)
            if guessed and guessed.exists():
                ply_path = guessed
        if ply_path and not ply_path.exists():
            print(f"[Fusion] WARN: local ply for {name or camera_id} missing: {ply_path}")
            ply_path = None

        lut_ref = entry.get("local_lut") or entry.get("lut") or entry.get("lut_npz")
        lut_path = None
        if isinstance(lut_ref, str) and lut_ref:
            candidate = Path(lut_ref)
            lut_path = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        if (lut_path is None or not lut_path.exists()) and lut_root and lut_root.exists():
            guessed_lut = guess_local_lut(lut_root, camera_id, name)
            if guessed_lut and guessed_lut.exists():
                lut_path = guessed_lut
        if lut_path and not lut_path.exists():
            print(f"[Fusion] WARN: local LUT for {name or camera_id} missing: {lut_path}")
            lut_path = None

        display_name = str(name or (f"cam{camera_id}" if camera_id is not None else f"marker{idx+1}"))
        slug = dedup_slug(slugify_label(display_name), used_slugs)
        overlay_base_url = entry.get("overlay_base_url") or entry.get("overlay_url") or entry.get("overlay_host")
        if overlay_base_url is not None:
            overlay_base_url = str(overlay_base_url).strip()
            if overlay_base_url:
                overlay_base_url = overlay_base_url.rstrip("/")
            else:
                overlay_base_url = None
        markers.append({
            "key": slug,
            "name": display_name,
            "camera_id": camera_id,
            "position": {"x": float(x), "y": float(y), "z": float(z)},
            "rotation": rotation,
            "local_ply": str(ply_path) if ply_path else None,
            "local_lut": str(lut_path) if lut_path else None,
            "overlay_base_url": overlay_base_url,
        })

    if not markers:
        print(f"[Fusion] camera position file {json_path} contained no usable entries")
    return markers


# Backward-compatible aliases
_safe_float = safe_float
_slugify_label = slugify_label
_dedup_slug = dedup_slug
_filter_candidates_by_cam_id = filter_candidates_by_cam_id
_guess_local_ply = guess_local_ply
_guess_local_lut = guess_local_lut
