"""
Build a dense RGB point cloud by unprojecting all depth frames.

Intrinsics are loaded from intrinsics.json in the scene directory (written by
prepare_scene.py / unpack_images.py).  If not found, falls back to the
default_intrinsics section of src/config.yaml.

Usage (from repo root):
    uv run python -m scene_search.dense_pointcloud \
        --base_dir /path/to/scene_frames \
        --save_path output/dense_pointcloud.npz
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from . import _config

_cfg      = _config.load()
_min_depth = _cfg.get("min_depth", 0.5)
_max_depth = _cfg.get("max_depth", 6.0)

# All pose filename conventions we accept (ScanNet + legacy)
_POSE_PATTERNS = ["pose.npy", "pose.txt", "extrinsic_matrix.npy"]

# Fallback intrinsics when no intrinsics.json is present
_fallback_intr = _cfg.get("default_intrinsics", {})
_FALLBACK_FX  = _fallback_intr.get("fx",  577.87)
_FALLBACK_FY  = _fallback_intr.get("fy",  577.87)
_FALLBACK_PPX = _fallback_intr.get("cx",  319.5)
_FALLBACK_PPY = _fallback_intr.get("cy",  239.5)


def load_scene_intrinsics(base_dir: str) -> dict | None:
    """Return per-scene intrinsics from intrinsics.json, or None."""
    p = Path(base_dir) / "intrinsics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _load_pose(frame_dir: str) -> np.ndarray | None:
    for pat in _POSE_PATTERNS:
        p = os.path.join(frame_dir, pat)
        if not os.path.exists(p):
            continue
        try:
            mat = np.loadtxt(p).reshape(4, 4) if pat.endswith(".txt") else np.load(p)
            return mat
        except Exception:
            continue
    return None


def create_dense_pointcloud(
    base_dir: str,
    save_path: str,
    stride: int = 10,
    min_depth: float = _min_depth,
    max_depth: float = _max_depth,
):
    scene_intr = load_scene_intrinsics(base_dir)
    if scene_intr is not None:
        fx, fy   = scene_intr["fx"], scene_intr["fy"]
        ppx, ppy = scene_intr["cx"], scene_intr["cy"]
    else:
        fx, fy, ppx, ppy = _FALLBACK_FX, _FALLBACK_FY, _FALLBACK_PPX, _FALLBACK_PPY
        print("Warning: no intrinsics.json found — using fallback intrinsics")

    print(f"Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={ppx:.1f} cy={ppy:.1f}")

    frame_dirs = sorted(
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for frame_dir in tqdm(frame_dirs, desc="Dense pointcloud"):
        rgb_path = os.path.join(frame_dir, "rgb_image.png")
        dep_path = os.path.join(frame_dir, "depth_image.png")
        if not os.path.exists(rgb_path) or not os.path.exists(dep_path):
            continue

        pose = _load_pose(frame_dir)
        if pose is None:
            continue

        depth = np.array(Image.open(dep_path), dtype=np.float32) / 1000.0  # mm → m
        color = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0

        H, W = depth.shape
        ys, xs = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
        ys, xs = ys.flatten(), xs.flatten()
        zs = depth[ys, xs]

        valid = (zs > min_depth) & (zs < max_depth)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        if len(zs) == 0:
            continue

        x_cam = (xs - ppx) / fx * zs
        y_cam = (ys - ppy) / fy * zs
        pts_cam = np.stack([x_cam, y_cam, zs, np.ones_like(zs)], axis=1)  # (N, 4)
        pts_world = (pose @ pts_cam.T).T[:, :3]

        fin = np.isfinite(pts_world).all(axis=1)
        all_points.append(pts_world[fin].astype(np.float32))
        all_colors.append(color[ys[fin], xs[fin]].astype(np.float32))

    if not all_points:
        raise ValueError("No valid points found in the scene")

    pts  = np.concatenate(all_points)
    cols = np.concatenate(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    pcd = pcd.uniform_down_sample(every_k_points=10)

    pts  = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)
    mask = np.isfinite(pts).all(axis=1)

    np.savez(save_path, points=pts[mask], colors=cols[mask])
    print(f"Saved {save_path} ({mask.sum()} points)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",  required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    create_dense_pointcloud(args.base_dir, args.save_path)
