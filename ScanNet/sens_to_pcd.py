"""
Convert a ScanNet .sens file directly to a voxel-downsampled .npz point cloud.
Does NOT extract per-frame folders — useful for quick visualisation.

Usage:
    python ScanNet/sens_to_pcd.py --input_file path/to/scene.sens \
        --frame_skip 10 --voxel_size 0.02

Output: <input_file>.npz  (points, colors)
"""

import argparse

import cv2
import numpy as np
import open3d as o3d

from scannet_sensordata import SensorData


def _unproject(depth: np.ndarray, color: np.ndarray, K: np.ndarray,
               pose: np.ndarray, depth_trunc: float) -> tuple[np.ndarray, np.ndarray]:
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.flatten()
    mask = (z > 0.1) & (z <= depth_trunc) & np.isfinite(z)
    z = z[mask]
    pixels = np.stack([u.flatten()[mask], v.flatten()[mask], np.ones(mask.sum())], axis=0)
    pts_cam   = (np.linalg.inv(K) @ pixels) * z
    pts_h     = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])
    pts_world = (pose @ pts_h).T[:, :3]
    valid = np.isfinite(pts_world).all(axis=1)
    cols = color.reshape(-1, 3)[mask].astype(np.float32) / 255.0
    return pts_world[valid], cols[valid]


def sens_to_npz(sens_path: str, frame_skip: int = 1,
                voxel_size: float = 0.02, depth_trunc: float = 3.0) -> None:
    sd = SensorData(sens_path)
    # Intrinsics come from the .sens file — they vary across ScanNet scenes
    K = sd.intrinsic_depth[:3, :3]
    H, W = sd.depth_height, sd.depth_width
    print(f"Intrinsics: fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    all_pts, all_cols = [], []
    for i in range(0, len(sd.frames), frame_skip):
        frame = sd.frames[i]
        depth_bytes = frame.decompress_depth(sd.depth_compression_type)
        depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(H, W).astype(np.float32) / 1000.0
        color = frame.decompress_color(sd.color_compression_type)
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_LINEAR)
        pts, cols = _unproject(depth, color, K, frame.camera_to_world, depth_trunc)
        if len(pts):
            all_pts.append(pts.astype(np.float32))
            all_cols.append(cols)

    if not all_pts:
        print("No valid points found.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_pts))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_cols))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    out_path = sens_path.replace(".sens", ".npz")
    np.savez_compressed(
        out_path,
        points=np.asarray(pcd.points, dtype=np.float32),
        colors=np.asarray(pcd.colors, dtype=np.float32),
    )
    print(f"Saved {out_path}  ({len(pcd.points)} points)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--frame_skip",  type=int,   default=1)
    parser.add_argument("--voxel_size",  type=float, default=0.02)
    parser.add_argument("--depth_trunc", type=float, default=3.0)
    args = parser.parse_args()
    sens_to_npz(args.input_file, args.frame_skip, args.voxel_size, args.depth_trunc)
