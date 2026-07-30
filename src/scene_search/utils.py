"""Shared helpers used across the scene_search pipeline."""

import json
import os
import re
from pathlib import Path

import yaml


# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    """Load src/config.yaml."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)

import numpy as np
import torchvision.transforms as T

# ── CLIP image preprocessing ──────────────────────────────────────────────────

def preprocess_image(image):
    """CLIP-compatible preprocessing for a PIL image."""
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std =[0.26862954, 0.26130258, 0.27577711],
        ),
    ])(image)


# ── Camera helpers ────────────────────────────────────────────────────────────

_POSE_PATTERNS = ["pose.npy", "pose.txt", "extrinsic_matrix.npy"]


def load_pose(frame_dir: str) -> np.ndarray | None:
    """Load a camera-to-world 4×4 pose from a frame directory.

    Tries pose.npy → pose.txt → extrinsic_matrix.npy.
    Validates shape, finiteness, and det(R) ≈ 1. Returns None if nothing valid found.
    """
    for pat in _POSE_PATTERNS:
        p = os.path.join(frame_dir, pat)
        if not os.path.exists(p):
            continue
        try:
            mat = np.loadtxt(p).reshape(4, 4) if pat.endswith(".txt") else np.load(p)
            if mat.shape != (4, 4) or not np.isfinite(mat).all():
                continue
            if abs(np.linalg.det(mat[:3, :3]) - 1.0) > 0.1:
                continue
            return mat.astype(np.float64)
        except Exception:
            continue
    return None


def load_intrinsics(
    base_dir: str,
    fallback_fx: float = 577.87,
    fallback_fy: float = 577.87,
    fallback_cx: float = 319.5,
    fallback_cy: float = 239.5,
) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) from intrinsics.json, or fallback values."""
    p = Path(base_dir) / "intrinsics.json"
    if p.exists():
        d = json.loads(p.read_text())
        fx, fy, cx, cy = d["fx"], d["fy"], d["cx"], d["cy"]
        print(f"Intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}")
        return fx, fy, cx, cy
    print("Warning: no intrinsics.json found — using fallback intrinsics")
    return fallback_fx, fallback_fy, fallback_cx, fallback_cy


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def sanitize_collection_name(name: str) -> str:
    """Return a ChromaDB-safe collection name (alphanumeric + . _ -)."""
    s = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    s = re.sub(r"^[^a-zA-Z0-9]+", "", s)
    s = re.sub(r"[^a-zA-Z0-9]+$", "", s)
    return s or "collection"
