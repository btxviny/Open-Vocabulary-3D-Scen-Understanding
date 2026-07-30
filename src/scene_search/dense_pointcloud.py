"""
Example usage:
python create_dense_pointcloud.py --base_dir ../data_from_runs/data_1744645809_90342760/ --save_path dense_pointcloud.npz
"""

import os
import yaml
import numpy as np
import argparse
import open3d as o3d
import tqdm
from pathlib import Path

from PIL import Image
from .camera_configs import get_camera_config, list_available_cameras, get_pose_file_patterns


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path('config.yaml')
    if not config_path.exists():
        # Try to find config.yaml in parent directories
        config_path = Path(__file__).parent.parent / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found. Looked in: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# Load configuration
try:
    config = load_config()
    min_depth = config.get('min_depth', 0.5)
    max_depth = config.get('max_depth', 6.0)
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}")
    min_depth = 0.5
    max_depth = 6.0


def detect_camera_type(base_dir):
    """
    Detect whether the data is from RealSense or ScanNet based on the data structure.
    This is a heuristic approach - you might need to adjust based on your specific data organization.
    """
    # Look for characteristic files or directory structure
    frames_dir = [os.path.join(base_dir, x) for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]
    
    if not frames_dir:
        return 'unknown'
    
    # Check first frame directory for clues
    first_frame = frames_dir[0]
    
    # RealSense typically has specific naming conventions
    if any(os.path.exists(os.path.join(first_frame, f)) for f in ['rgb_image.png', 'depth_image.png']):
        # Check if we have extrinsic_matrix.npy (RealSense format)
        if os.path.exists(os.path.join(first_frame, 'extrinsic_matrix.npy')):
            return 'realsense'
        # Check if we have pose.txt or similar (ScanNet format)
        elif os.path.exists(os.path.join(first_frame, 'pose.txt')) or os.path.exists(os.path.join(first_frame, 'pose.npy')):
            return 'scannet'
    
    # Default to RealSense if uncertain
    return 'realsense'


def unproject_to_3d(x, y, z, intrinsics):
    """Unproject 2D pixel coordinates to 3D camera coordinates using given intrinsics."""
    x_norm = (x - intrinsics['ppx']) / intrinsics['fx']
    y_norm = (y - intrinsics['ppy']) / intrinsics['fy']
    return [x_norm * z, y_norm * z, z]


def create_dense_pointcloud(base_dir, save_path, stride=10, min_depth=0.1, max_depth=10.0, camera_type=None):
    # Detect camera type if not provided
    if camera_type is None:
        camera_type = detect_camera_type(base_dir)
    
    # Get camera configuration
    try:
        intrinsics, cam_to_imu = get_camera_config(camera_type)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available camera types: {list_available_cameras()}")
        return
    
    print(f"Using camera type: {camera_type}")
    print(f"Using intrinsics: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
          f"ppx={intrinsics['ppx']:.2f}, ppy={intrinsics['ppy']:.2f}")
    
    frames_dir = [os.path.join(base_dir, x) for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]

    final_points = []
    final_colors = []

    for frame_dir in tqdm.tqdm(frames_dir, total=len(frames_dir), desc="Processing frames"):
        # Load RGB image
        image_path = os.path.join(frame_dir, 'rgb_image.png')
        if not os.path.exists(image_path): continue
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Load depth image
        depth_path = os.path.join(frame_dir, 'depth_image.png')
        if not os.path.exists(depth_path): continue
        depth = Image.open(depth_path)
        depth_np = np.array(depth).astype(np.float32) / 1000.0  # convert mm to meters

        # Load extrinsic matrix - handle different formats based on camera type
        extrinsic = None
        pose_patterns = get_pose_file_patterns(camera_type)
        
        for pattern in pose_patterns:
            pose_path = os.path.join(frame_dir, pattern)
            if os.path.exists(pose_path):
                try:
                    if pattern.endswith('.txt'):
                        # Load from text file (assuming space-separated values)
                        extrinsic = np.loadtxt(pose_path).reshape(4, 4)
                    else:
                        extrinsic = np.load(pose_path)
                    break
                except Exception as e:
                    print(f"Warning: Failed to load pose from {pose_path}: {e}")
                    continue
        
        if extrinsic is None:
            print(f"Warning: No valid pose found for frame {frame_dir}, skipping...")
            continue

        # Depth filtering
        valid_depth = (depth_np > min_depth) & (depth_np < max_depth)
        if not np.any(valid_depth):
            continue

        # Iterate over pixels
        height, width = depth_np.shape
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                if not valid_depth[y, x]:
                    continue

                z = depth_np[y, x]
                pt_cam = unproject_to_3d(x, y, z, intrinsics)
                pt_cam_h = np.append(pt_cam, 1.0)  # homogeneous

                # Apply camera-to-imu transformation and then world transformation
                pt_world = extrinsic @ cam_to_imu @ pt_cam_h
                final_points.append(pt_world[:3])
                final_colors.append(image_np[y, x] / 255.0)

    if len(final_points) == 0:
        raise ValueError("No valid points found in the scene")

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)

    # # Estimate normals (helps with outlier removal)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # # Remove radius outliers
    # radius = 0.01 # meters 
    # min_points = 50 #minimum number of neighbors inside 'radius'to be considered an inlier
    # pcd, _ = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)

    # # Remove statistical outliers with adaptive parameters
    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2)


    #Uniform downsampling
    pcd = pcd.uniform_down_sample(every_k_points=10)

    
    # Convert back to numpy
    final_points = np.asarray(pcd.points)
    final_colors = np.asarray(pcd.colors)

    # Final validation - remove any remaining NaN or inf values
    valid_mask = ~(np.any(np.isnan(final_points), axis=1) | np.any(np.isinf(final_points), axis=1))
    if not np.all(valid_mask):
        print(f"Warning: Found {np.sum(~valid_mask)} invalid points, removing them...")
        final_points = final_points[valid_mask]
        final_colors = final_colors[valid_mask]
    
    if len(final_points) == 0:
        raise ValueError("No valid points found after validation")

    # Save to .npz
    np.savez(save_path,
             points=np.array(final_points),
             colors=np.array(final_colors))

    print(f"Saved {save_path} with {len(final_points)} points.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate a global CLIP-colored point cloud.")
    parser.add_argument('--base_dir', type=str, required=True, help="Path to base directory with frame subfolders.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the resulting .npz file.")
    parser.add_argument('--camera_type', type=str, choices=['auto'] + list_available_cameras(), default='auto',
                       help="Camera type to use. 'auto' will attempt to detect automatically.")
    args = parser.parse_args()

    # Handle camera type selection
    camera_type = None
    if args.camera_type != 'auto':
        camera_type = args.camera_type
    
    create_dense_pointcloud(args.base_dir, args.save_path, stride=10, min_depth=min_depth, max_depth=max_depth, camera_type=camera_type)
