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

# Backward-compatible aliases
_safe_float = safe_float
_slugify_label = slugify_label
_dedup_slug = dedup_slug
_filter_candidates_by_cam_id = filter_candidates_by_cam_id
_guess_local_ply = guess_local_ply
_guess_local_lut = guess_local_lut
