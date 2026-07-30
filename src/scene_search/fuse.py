"""
Dense RGBD point cloud from all frames in a scene directory.

Reprojects every depth pixel to world space using camera intrinsics and
camera-to-world pose, concatenates across frames, uniform-downsamples,
and saves (points, colors) to a .npz file.

Usage (from repo root):
    uv run python -m src.scene_search.fuse \\
        --base_dir ScanNet/RGBD/scene0001_00 \\
        --save_path output/dense_pointcloud.npz
"""

import argparse
import os

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from .utils import load_config, load_intrinsics, load_pose

_cfg       = load_config()
_min_depth = _cfg.get("min_depth", 0.5)
_max_depth = _cfg.get("max_depth", 6.0)
_fallback  = _cfg.get("default_intrinsics", {})


def create_dense_pointcloud(
    base_dir: str,
    save_path: str,
    stride: int = 10,
    min_depth: float = _min_depth,
    max_depth: float = _max_depth,
) -> None:
    fx, fy, cx, cy = load_intrinsics(base_dir, **_fallback)

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

        pose = load_pose(frame_dir)
        if pose is None:
            continue

        color = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float32) / 1000.0  # mm → m

        H, W = color.shape[:2]
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
        ys, xs = ys.flatten(), xs.flatten()
        zs = depth[ys, xs]

        valid = (zs > min_depth) & (zs < max_depth)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        if len(zs) == 0:
            continue

        pts_cam = np.stack([(xs - cx) / fx * zs,
                            (ys - cy) / fy * zs,
                            zs,
                            np.ones_like(zs)], axis=1)
        pts_world = (pose @ pts_cam.T).T[:, :3]

        fin = np.isfinite(pts_world).all(axis=1)
        all_points.append(pts_world[fin].astype(np.float32))
        all_colors.append(color[ys[fin], xs[fin]].astype(np.float32))

    if not all_points:
        raise ValueError(f"No valid depth frames found in {base_dir}")

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
    print(f"Saved {save_path}  ({mask.sum():,} points)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense RGBD depth fusion")
    parser.add_argument("--base_dir",  required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--stride",    type=int, default=10)
    args = parser.parse_args()
    create_dense_pointcloud(args.base_dir, args.save_path, stride=args.stride)
