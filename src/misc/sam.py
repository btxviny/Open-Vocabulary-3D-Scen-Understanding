import os
import glob
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
import open3d as o3d
import warnings

import yaml
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
frame_stride = config['frame_stride']
point_stride = config['point_stride']

os.chdir('../sam2/')

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

# === Intrinsics ===
fx = 386.99176025390625
fy = 386.99176025390625
ppx = 320.9659118652344
ppy = 241.470703125

# === Camera → IMU transformation ===
cam_to_imu = np.array([
    [0.894078, 0.012117, 0.447747, 0.008695],
    [-0.446418, -0.057450, 0.892979, 0.127623],
    [0.036543, -0.998275, -0.045955, 0.761847],
    [0.000000, 0.000000, 0.000000, 1.000000]
])

# Inverse of the intrinsic matrix
K_inv = np.array([
    [0.00258403, 0, -0.82938694],
    [0,  0.00258403, -0.62396859],
    [0, 0, 1]
])

def apply_transformation(points, matrix):
    points = np.asarray(points)
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack((points, ones))        # Nx4
    transformed = homogeneous @ matrix.T           # Nx4
    return transformed[:, :3]

def save_pcd(points, colors, path):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), cloud)

def main(base_dir):
    model_path = "./checkpoints/sam2.1_hiera_base_plus.pt"
    config_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = build_sam2(config_path, ckpt_path=model_path, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=16,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
        output_mode="binary_mask",
        crop_n_layers=0,
        min_mask_region_area=200,
    )

    # === Recursively find all sets ===
    rgb_paths = sorted(glob.glob(str(base_dir / "**/rgb_image.png"), recursive=True))
    depth_paths = sorted(glob.glob(str(base_dir / "**/depth_image.png"), recursive=True))
    extrinsic_paths = sorted(glob.glob(str(base_dir / "**/extrinsic_matrix.npy"), recursive=True))
    zipped_paths = list(zip(rgb_paths, depth_paths, extrinsic_paths))[::frame_stride]


    for rgb_path, depth_path, extrinsic_path in tqdm(zipped_paths,desc='Segmenting Objects', total=len(zipped_paths)):
        folder = Path(rgb_path).parent

        if not (Path(rgb_path).exists() and Path(depth_path).exists() and Path(extrinsic_path).exists()):
            continue

        # Load data
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        extrinsic = np.load(extrinsic_path)

        if rgb.shape[:2] != depth.shape:
            print(f"Skipping {folder} — shape mismatch")
            continue

        # Segment
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")
            masks = mask_generator.generate(rgb)

        for mask_id, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation']
            ys, xs = np.where(mask)

            # Apply stride
            xs = xs[::point_stride]
            ys = ys[::point_stride]

            # Unproject via inverse intrinsics
            pixels = np.stack([xs, ys, np.ones_like(xs)], axis=1).T  # (3, N)
            depths = depth[ys, xs]                                   # Shape: (N,)

            # Filter valid depth
            valid = (depths > 0.5) & (depths <= 6.0)
            pixels = pixels[:, valid]
            depths = depths[valid]
            colors_local = rgb[ys[valid], xs[valid]]

            # Apply inverse intrinsic matrix to get normalized camera coords
            points_camera = (K_inv @ pixels) * depths  # Shape: (3, N)
            points_local = points_camera.T             # Shape: (N, 3)

            if not points_local.size:
                continue

            # Transform through cam→imu→world
            points_imu = apply_transformation(points_local, cam_to_imu)
            points_world = apply_transformation(points_imu, extrinsic)

            # Save .pcd
            pcd_path = folder / f"object_{mask_id}.pcd"
            save_pcd(points_world, colors_local, pcd_path)
            cv2.imwrite(str(folder / f"mask_{mask_id}.png"), (mask.astype(np.uint8) * 255))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masks and 3D point clouds.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing data folders.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    main(base_dir)