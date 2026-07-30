"""
Build a dense RGB point cloud by unprojecting all depth frames.

Usage (from repo root):
    uv run python -m scene_search.dense_pointcloud \
        --base_dir /path/to/scene_frames \
        --save_path output/dense_pointcloud.npz \
        --camera_type scannet
"""

import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from . import _config
from .camera_configs import get_camera_config, list_available_cameras, get_pose_file_patterns

_cfg = _config.load()
_min_depth = _cfg.get("min_depth", 0.5)
_max_depth = _cfg.get("max_depth", 6.0)


def detect_camera_type(base_dir: str) -> str:
    dirs = sorted(
        d for d in (Path(base_dir) / e for e in os.listdir(base_dir))
        if d.is_dir()
    )
    if not dirs:
        return "realsense"
    first = dirs[0]
    if (first / "extrinsic_matrix.npy").exists():
        return "realsense"
    if (first / "pose.npy").exists() or (first / "pose.txt").exists():
        return "scannet"
    return "realsense"


def _load_pose(frame_dir: str, pose_patterns) -> np.ndarray | None:
    for pat in pose_patterns:
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
    camera_type: str | None = None,
):
    if camera_type is None:
        camera_type = detect_camera_type(base_dir)

    intrinsics, cam_to_imu = get_camera_config(camera_type)
    pose_patterns = get_pose_file_patterns(camera_type)

    fx   = intrinsics["fx"]
    fy   = intrinsics["fy"]
    ppx  = intrinsics["ppx"]
    ppy  = intrinsics["ppy"]

    print(f"Camera: {camera_type}  fx={fx:.1f} fy={fy:.1f} cx={ppx:.1f} cy={ppy:.1f}")

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

        extrinsic = _load_pose(frame_dir, pose_patterns)
        if extrinsic is None:
            continue

        depth = np.array(Image.open(dep_path), dtype=np.float32) / 1000.0  # mm → m
        color = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0

        # Subsample and filter
        H, W = depth.shape
        ys, xs = np.meshgrid(
            np.arange(0, H, stride),
            np.arange(0, W, stride),
            indexing="ij",
        )
        ys, xs = ys.flatten(), xs.flatten()
        zs = depth[ys, xs]

        valid = (zs > min_depth) & (zs < max_depth)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        if len(zs) == 0:
            continue

        # Vectorised unproject (camera coords)
        x_cam = (xs - ppx) / fx * zs
        y_cam = (ys - ppy) / fy * zs
        pts_cam = np.stack([x_cam, y_cam, zs, np.ones_like(zs)], axis=1)  # (N, 4)

        # Camera → world
        pts_world = (extrinsic @ cam_to_imu @ pts_cam.T).T[:, :3]

        fin = np.isfinite(pts_world).all(axis=1)
        all_points.append(pts_world[fin].astype(np.float32))
        all_colors.append(color[ys[fin], xs[fin]].astype(np.float32))

    if not all_points:
        raise ValueError("No valid points found in the scene")

    pts = np.concatenate(all_points)
    cols = np.concatenate(all_colors)

    # Downsample with Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    pcd = pcd.uniform_down_sample(every_k_points=10)

    pts  = np.asarray(pcd.points,  dtype=np.float32)
    cols = np.asarray(pcd.colors,  dtype=np.float32)

    valid_mask = np.isfinite(pts).all(axis=1)
    pts, cols = pts[valid_mask], cols[valid_mask]

    np.savez(save_path, points=pts, colors=cols)
    print(f"Saved {save_path} ({len(pts)} points)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",    required=True)
    parser.add_argument("--save_path",   required=True)
    parser.add_argument("--camera_type", choices=["auto"] + list_available_cameras(), default="auto")
    args = parser.parse_args()

    create_dense_pointcloud(
        args.base_dir, args.save_path,
        camera_type=None if args.camera_type == "auto" else args.camera_type,
    )
