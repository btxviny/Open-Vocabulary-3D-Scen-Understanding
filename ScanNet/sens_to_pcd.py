#!/usr/bin/env python3
import os
import numpy as np
import cv2
import open3d as o3d
import argparse
from scannet_sensordata import SensorData

def make_intrinsic(fx=577.870605, fy=577.870605, cx=319.5, cy=239.5):
    """Return 3x3 intrinsic matrix."""
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)

def save_full_pointcloud(depth_img, color_img, intrinsic, extrinsic, point_stride=1, min_depth=0.1, max_depth=3.0):
    H, W = depth_img.shape

    # Subsample pixels
    u, v = np.meshgrid(np.arange(0, W, point_stride), np.arange(0, H, point_stride))
    z = depth_img[v, u].reshape(-1)  # flatten
    colors = color_img[v, u].reshape(-1, 3)

    # Filter valid depths
    mask = (z > min_depth) & (z <= max_depth) & np.isfinite(z)
    z = z[mask]
    u = u.reshape(-1)[mask]
    v = v.reshape(-1)[mask]
    colors = colors[mask]

    if len(z) == 0:
        return np.empty((0,3)), np.empty((0,3))

    # Unproject to camera coordinates
    pixels = np.stack([u, v, np.ones_like(u)], axis=0)  # (3, N)
    K_inv = np.linalg.inv(intrinsic)
    pts_cam = (K_inv @ pixels) * z  # (3,N)
    pts_cam = pts_cam.T  # (N,3)

    # Transform to world coordinates
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (extrinsic @ pts_h.T).T[:, :3]

    # Remove invalid points
    valid = np.isfinite(pts_world).all(axis=1)
    return pts_world[valid], colors[valid]

def sens_to_npz(sens_path, frame_skip=1, voxel_size=0.02, depth_scale=1000.0, depth_trunc=3.0):
    sd = SensorData(sens_path)
    H, W = sd.depth_height, sd.depth_width
    K = make_intrinsic()

    all_points = []
    all_colors = []

    for i in range(0, len(sd.frames), frame_skip):
        frame = sd.frames[i]

        # Decompress depth
        depth_bytes = frame.decompress_depth(sd.depth_compression_type)
        depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(H, W).astype(np.float32)
        depth /= depth_scale
        depth[depth > depth_trunc] = 0.0

        # Decompress and resize color
        color = frame.decompress_color(sd.color_compression_type)
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_LINEAR)

        # Get points/colors
        pts, cols = save_full_pointcloud(depth, color, K, frame.camera_to_world)
        if len(pts) > 0:
            all_points.append(pts)
            all_colors.append(cols.astype(np.float32)/255.0)

    if len(all_points) == 0:
        print("No valid points found in the .sens file.")
        return

    # Merge all frames
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Optional voxel downsample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # Save as .npz
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    out_path = sens_path.replace(".sens", ".npz")
    np.savez_compressed(out_path, points=points, colors=colors)
    print(f"Saved {out_path} with {len(points)} points")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ScanNet .sens to .npz point cloud")
    parser.add_argument("--input_file", required=True, help="Path to .sens file")
    parser.add_argument("--frame_skip", type=int, default=1, help="Skip every n frames")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="Voxel downsample size in meters")
    args = parser.parse_args()

    sens_to_npz(args.input_file, frame_skip=args.frame_skip, voxel_size=args.voxel_size)
