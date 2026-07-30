"""
Modified SAM3D Script for Custom Dataset with Individual Object Saving
Original Author: Yunhan Yang (yhyang.myron@gmail.com)
Modified by: Assistant

Example usage:
python -m scene_search.sam3d --base_dir ../data_from_runs/data_1744645809_90342760/ --save_path segmented_pointcloud.npz --camera_type auto
"""

import os
import cv2
import numpy as np
import torch
import copy
import multiprocessing as mp
import pointops
import random
import argparse
import yaml

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from PIL import Image
from os.path import join
from tqdm import tqdm

# Load configuration BEFORE changing directory
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# SAM3D parameters
voxel_size = config.get('voxel_size', 0.05)
point_stride = config.get('point_stride', 20)
frame_stride = config.get('frame_stride', 20)
th = config.get('th', 200)
min_depth = config.get('min_depth', 0.5)
max_depth = config.get('max_depth', 6.0)
group_overlap_ratio = config.get('group_overlap_ratio', 0.3)  # Default to 0.3 if not specified
voxel_search_multiplier = config.get('voxel_search_multiplier', 1.5)  # Default to 1.5 if not specified

# SAM mask generator parameters
sam_config = {
    'points_per_side': config.get('points_per_side', 16),
    'pred_iou_thresh': config.get('pred_iou_thresh', 0.7),
    'stability_score_thresh': config.get('stability_score_thresh', 0.7),
    'min_mask_region_area': config.get('min_mask_region_area', 200),
}

# Load camera configurations BEFORE changing directory
def load_camera_configs():
    """Load camera configurations from config.yaml."""
    if 'cameras' not in config:
        raise ValueError("No 'cameras' section found in config.yaml")
    return config['cameras']

# Store camera configs globally
CAMERA_CONFIGS = load_camera_configs()

def get_camera_config_for_sam3d(camera_type):
    """Get camera configuration for SAM3D processing."""
    if camera_type not in CAMERA_CONFIGS:
        raise ValueError(f"Unknown camera type: {camera_type}. Available types: {list(CAMERA_CONFIGS.keys())}")
    
    camera_config = CAMERA_CONFIGS[camera_type]
    intrinsics = camera_config['intrinsics']
    cam_to_imu = np.array(camera_config['cam_to_imu'])
    
    # Convert intrinsics to inverse matrix format
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    ppx = intrinsics['ppx']
    ppy = intrinsics['ppy']
    
    K_inv = np.array([
        [1/fx, 0, -ppx/fx],
        [0, 1/fy, -ppy/fy],
        [0, 0, 1]
    ])
    
    return K_inv, cam_to_imu

# Now change directory to sam2
import sys
os.chdir('../sam2/')  # Change directory to sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

# Import util from SegmentAnything3D using absolute path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SegmentAnything3D'))
from util import *

#------------------------------------------------------------------------------
# Configuration and Constants
#------------------------------------------------------------------------------

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"
# torch.cuda.empty_cache()
# if torch.cuda.is_available():
#     torch.cuda.set_per_process_memory_fraction(0.9)

#------------------------------------------------------------------------------
# Core Processing Pipeline
#------------------------------------------------------------------------------

def seg_pcd_custom(frame_paths, save_path, mask_generator, voxel_size, voxelizer, th, point_stride, camera_type):
    """Main processing pipeline that handles multiple frames."""
    pcd_list = []
    
    # Get camera configuration
    K_inv, cam_to_imu = get_camera_config_for_sam3d(camera_type)
    print(f"Using camera type: {camera_type}")
    print(f"Using intrinsics: fx={1/K_inv[0,0]:.2f}, fy={1/K_inv[1,1]:.2f}, ppx={-K_inv[0,2]/K_inv[0,0]:.2f}, ppy={-K_inv[1,2]/K_inv[1,1]:.2f}")
    
    # Process each frame with progress bar
    for frame_path in tqdm(frame_paths, desc="Processing frames", unit="frame"):
        pcd_dict = get_pcd_custom(frame_path, mask_generator, point_stride, K_inv, cam_to_imu, camera_type)
        if len(pcd_dict["coord"]) == 0:
            continue
        pcd_dict = voxelizer(pcd_dict)
        pcd_list.append(pcd_dict)
    
    # Merge point clouds with progress tracking
    while len(pcd_list) != 1:
        print(len(pcd_list), flush=True)
        new_pcd_list = []
        for indice in pairwise_indices(len(pcd_list)):
            pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelizer, th = th)
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list
    
    # Final processing
    print("Applying final processing...")
    seg_dict = pcd_list[0]
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))
    
    # Final validation - remove any remaining NaN or inf values
    valid_mask = ~(np.any(np.isnan(seg_dict["coord"]), axis=1) | np.any(np.isinf(seg_dict["coord"]), axis=1))
    if not np.all(valid_mask):
        print(f"Warning: Found {np.sum(~valid_mask)} invalid points in final output, removing them...")
        seg_dict["coord"] = seg_dict["coord"][valid_mask]
        seg_dict["color"] = seg_dict["color"][valid_mask]
        seg_dict["group"] = seg_dict["group"][valid_mask]
    
    if len(seg_dict["coord"]) == 0:
        raise ValueError("No valid points found after final validation")
    
    np.savez(save_path, points=seg_dict["coord"], colors=seg_dict["color"], cluster_labels=seg_dict["group"])
    print(f"\nSaved merged result to: {save_path}")

def get_pcd_custom(frame_path, mask_generator, point_stride, K_inv, cam_to_imu, camera_type):
    """Process a single frame to generate point cloud and object masks."""
    # Extract timestamp from frame path
    timestamp = os.path.basename(frame_path)
    
    # Load images and matrices
    depth_path = join(frame_path, 'depth_image.png')
    rgb_path = join(frame_path, 'rgb_image.png')
    
    # Handle different pose file formats based on camera type
    extrinsic = None
    if camera_type == 'realsense':
        extrinsic_path = join(frame_path, 'extrinsic_matrix.npy')
        if os.path.exists(extrinsic_path):
            extrinsic = np.load(extrinsic_path)  # 4x4
    elif camera_type == 'scannet':
        # Try different pose file formats for ScanNet
        pose_paths = [
            join(frame_path, 'pose.txt'),
            join(frame_path, 'pose.npy'),
            join(frame_path, 'extrinsic_matrix.npy')
        ]
        for pose_path in pose_paths:
            if os.path.exists(pose_path):
                try:
                    if pose_path.endswith('.txt'):
                        # Load from text file (assuming space-separated values)
                        extrinsic = np.loadtxt(pose_path).reshape(4, 4)
                    else:
                        extrinsic = np.load(pose_path)
                    
                    # Validate the transformation matrix
                    if extrinsic.shape != (4, 4):
                        print(f"Warning: Invalid pose matrix shape {extrinsic.shape} in {pose_path}, trying next...")
                        continue
                    
                    # Check for NaN or inf values
                    if np.any(np.isnan(extrinsic)) or np.any(np.isinf(extrinsic)):
                        print(f"Warning: Pose matrix contains NaN or inf values in {pose_path}, trying next...")
                        continue
                    
                    # Check if the matrix is a valid transformation matrix (determinant should be close to 1)
                    det = np.linalg.det(extrinsic[:3, :3])
                    if abs(det - 1.0) > 0.1:
                        print(f"Warning: Pose matrix has invalid determinant {det:.3f} in {pose_path}, trying next...")
                        continue
                    
                    break
                except Exception as e:
                    print(f"Warning: Failed to load pose from {pose_path}: {e}, trying next...")
                    continue
    
    if extrinsic is None:
        print(f"Warning: No pose found for frame {frame_path}, skipping...")
        return dict(coord=np.array([]), color=np.array([]), group=np.array([]))
    
    depth_img = cv2.imread(depth_path, -1) / 1000.0 # read 16bit grayscale image
    if depth_img is None:
        print(f"Warning: Could not read depth image from {depth_path}")
        return dict(coord=np.array([]), color=np.array([]), group=np.array([]))
        
    # Validate depth range
    depth_mask = (depth_img > min_depth) & (depth_img <= max_depth)  # Valid depth range
    if not np.any(depth_mask):
        print(f"Warning: No valid depth values in {depth_path}")
        return dict(coord=np.array([]), color=np.array([]), group=np.array([]))
    
    color_image = cv2.imread(rgb_path)
    if color_image is None:
        print(f"Warning: Could not read RGB image from {rgb_path}")
        return dict(coord=np.array([]), color=np.array([]), group=np.array([]))
    
    color_image = cv2.resize(color_image, (640, 480))

    # Get SAM masks and save them in the same directory as the input data
    if mask_generator is not None:
        masks = mask_generator.generate(color_image)
        group_ids = np.full((480, 640), -1, dtype=int)
        for i in reversed(range(len(masks))):
            group_ids[masks[i]["segmentation"]] = i
            
        # Save individual object point clouds using direct unprojection
        save_object_pointclouds_from_masks(
            masks=masks,
            depth_img=depth_img,
            color_image=color_image,
            extrinsic=extrinsic,
            save_dir=frame_path,
            point_stride=point_stride,
            min_points=50,
            K_inv=K_inv,
            cam_to_imu=cam_to_imu
        )
    else:
        # Load existing masks and combine them
        group_ids = np.full((480, 640), -1, dtype=int)
        mask_files = [f for f in os.listdir(frame_path) if f.startswith('mask_') and f.endswith('.png')]
        for mask_file in mask_files:
            idx = int(mask_file.split('_')[1].split('.')[0])
            mask = cv2.imread(join(frame_path, mask_file), cv2.IMREAD_GRAYSCALE)
            group_ids[mask > 127] = idx

    # Process color image
    color_image = np.reshape(color_image[depth_mask], [-1,3])
    group_ids = group_ids[depth_mask]
    colors = np.zeros_like(color_image)
    colors[:,0] = color_image[:,2]
    colors[:,1] = color_image[:,1]
    colors[:,2] = color_image[:,0]
    colors = colors.astype(np.float32) / 255.0  # Normalize colors to [0,1]

    # Create point cloud from depth
    height, width = depth_img.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack((x.flatten(), y.flatten(), np.ones_like(x.flatten())), axis=1)
    pixels = pixels[depth_mask.flatten()]
    
    # Convert to camera space
    points_cam = np.dot(K_inv, pixels.T).T
    depth_values = depth_img[depth_mask].flatten()
    points_cam = points_cam * depth_values[:, np.newaxis]
    
    # Remove points with invalid depth or too far
    valid_points = (depth_values > min_depth) & (depth_values < max_depth)  # Keep points between 0.1m and 6m
    points_cam = points_cam[valid_points]
    colors = colors[valid_points]
    group_ids = group_ids[valid_points]
    
    if len(points_cam) == 0:
        print(f"Warning: No valid points generated for {frame_path}")
        return dict(coord=np.array([]), color=np.array([]), group=np.array([]))
    
    # Convert to homogeneous coordinates
    points_cam_h = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)
    
    # Transform to lidar space using both transformations
    points_lidar = np.dot(extrinsic, np.dot(cam_to_imu, points_cam_h.T)).T[:, :3]
    
    # Remove points with invalid coordinates
    valid_coords = ~np.any(np.isnan(points_lidar) | np.isinf(points_lidar), axis=1)
    points_lidar = points_lidar[valid_coords]
    colors = colors[valid_coords]
    group_ids = group_ids[valid_coords]
    
    if len(points_lidar) == 0:
        print(f"Warning: No valid points after transformation for {frame_path}")
        return dict(coord=np.array([]), color=np.array([]), group=np.array([]))
    
    return dict(coord=points_lidar, color=colors, group=group_ids)

#------------------------------------------------------------------------------
# Object Point Cloud Generation
#------------------------------------------------------------------------------

def save_object_pointclouds_from_masks(masks, depth_img, color_image, extrinsic, save_dir, point_stride=5, min_points=200, K_inv=None, cam_to_imu=None):
    """Save individual point clouds for each SAM mask directly using unprojection."""
    os.makedirs(save_dir, exist_ok=True)
    
    for mask_id, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        ys, xs = np.where(mask)
        
        # Apply stride to indices
        xs = xs[::point_stride]
        ys = ys[::point_stride]
        
        # Get depths for these points
        depths = depth_img[ys, xs]
        
        # Filter valid depth
        valid = (depths > min_depth) & (depths <= max_depth)
        xs = xs[valid]
        ys = ys[valid]
        depths = depths[valid]
        colors_local = color_image[ys, xs]
        
        if len(depths) < min_points:
            continue
            
        # Unproject via inverse intrinsics
        pixels = np.stack([xs, ys, np.ones_like(xs)], axis=1).T  # (3, N)
        
        # Apply inverse intrinsic matrix to get normalized camera coords
        points_camera = (K_inv @ pixels) * depths  # Shape: (3, N)
        points_local = points_camera.T             # Shape: (N, 3)
        
        if not points_local.size:
            continue
            
        # Transform through cam→imu→world
        points_cam_h = np.concatenate([points_local, np.ones((points_local.shape[0], 1))], axis=1)
        points_world = np.dot(extrinsic, np.dot(cam_to_imu, points_cam_h.T)).T[:, :3]
        
        # Filter out invalid points
        valid_coords = ~np.any(np.isnan(points_world) | np.isinf(points_world), axis=1)
        points_world = points_world[valid_coords]
        colors_local = colors_local[valid_coords]
        
        if len(points_world) < min_points:
            continue
            
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors_local.astype(np.float32) / 255.0)
        
        mask_img = mask.astype(np.uint8) * 255
        mask_path = os.path.join(save_dir, f'mask_{mask_id}.png')
        cv2.imwrite(mask_path, mask_img)
            
        # Save as .pcd format
        pcd_path = os.path.join(save_dir, f'object_{mask_id}.pcd')
        o3d.io.write_point_cloud(pcd_path, pcd)

#------------------------------------------------------------------------------
# Point Cloud Merging and Processing
#------------------------------------------------------------------------------

def make_open3d_point_cloud(input_dict, th):
    """Create an Open3D point cloud from input dictionary."""
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def cal_2_scenes(pcd_list, index, voxel_size, voxelize, th=50):
    """Merge two point clouds and calculate their overlap."""
    if len(index) == 1:
        return(pcd_list[index[0]])
    
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]
    
    # Create point clouds and handle empty or invalid cases
    pcd0 = make_open3d_point_cloud(copy.deepcopy(input_dict_0), th)
    pcd1 = make_open3d_point_cloud(copy.deepcopy(input_dict_1), th)
    
    if pcd0 is None:
        if pcd1 is None:
            return None
        else:
            return input_dict_1
    elif pcd1 is None:
        return input_dict_0

    # Ensure groups are properly sized
    input_dict_0["group"] = input_dict_0["group"][:len(pcd0.points)]
    input_dict_1["group"] = input_dict_1["group"][:len(pcd1.points)]

    # Cal Dul-overlap
    match_inds = get_matching_indices(pcd1, pcd0, voxel_search_multiplier * voxel_size, 1)
    if not match_inds:  # If no matches found
        return input_dict_0  # Return first point cloud
    pcd1_new_group = cal_group(input_dict_0, input_dict_1, match_inds, ratio=group_overlap_ratio)

    match_inds = get_matching_indices(pcd0, pcd1, voxel_search_multiplier * voxel_size, 1)
    if not match_inds:  # If no matches found
        return input_dict_0  # Return first point cloud
    input_dict_1["group"] = pcd1_new_group
    pcd0_new_group = cal_group(input_dict_1, input_dict_0, match_inds, ratio=group_overlap_ratio)

    # Concatenate results
    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = num_to_natural(pcd_new_group)
    pcd_new_coord = np.concatenate((input_dict_0["coord"], input_dict_1["coord"]), axis=0)
    pcd_new_color = np.concatenate((input_dict_0["color"], input_dict_1["color"]), axis=0)
    pcd_dict = dict(coord=pcd_new_coord, color=pcd_new_color, group=pcd_new_group)

    return voxelize(pcd_dict)

def cal_group(input_dict, new_input_dict, match_inds, ratio=None):
    """Calculate group correspondence between two point clouds.
    
    Args:
        input_dict: Dictionary containing first point cloud data
        new_input_dict: Dictionary containing second point cloud data
        match_inds: Matching indices between point clouds
        ratio: Overlap ratio threshold for merging groups. If None, uses value from config.
    """
    if ratio is None:
        ratio = group_overlap_ratio
        
    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    
    # Ensure we're working with numpy arrays
    if isinstance(group_0, torch.Tensor):
        group_0 = group_0.cpu().numpy()
    if isinstance(group_1, torch.Tensor):
        group_1 = group_1.cpu().numpy()
    
    # Create a copy to avoid modifying the original
    group_1 = group_1.copy()
    
    # Ensure proper offset for new group IDs
    if len(np.unique(group_0)) > 0:
        group_1[group_1 != -1] += np.max(group_0) + 1
    
    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # Calculate group correspondence
    group_overlap = {}
    for i, j in match_inds:
        if i >= len(group_1) or j >= len(group_0):
            continue
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1

    # Update group information
    for group_i, overlap_count in group_overlap.items():
        if not overlap_count:  # Skip if no overlaps
            continue
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]
        if group_j in group_0_counts and group_i in group_1_counts:
            total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
            if count / total_count >= ratio:
                group_1[group_1 == group_i] = group_j
    
    return group_1


#------------------------------------------------------------------------------
# Main Entry Point
#------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='Segment Anything on Custom Dataset')
    parser.add_argument('--base_dir', type=str, required=True, help='Path to the run directory containing timestamp folders')
    parser.add_argument('--save_path', type=str, help='Path where to save the resulting npz file. If not provided, saves in base_dir')
    parser.add_argument('--voxel_size', type=float, default=voxel_size, help='Voxel size for point cloud processing')
    parser.add_argument('--th', type=int, default=th, help='Threshold for removing small groups')
    parser.add_argument('--frame_stride', type=int, default=frame_stride, help='Process every nth frame (default: 1)')
    parser.add_argument('--point_stride', type=int, default=point_stride, help='Process every nth point (default: 1)')
    parser.add_argument('--camera_type', type=str, choices=['auto', 'realsense', 'scannet'], default='auto',
                       help="Camera type to use. 'auto' will attempt to detect automatically.")
    args = parser.parse_args()
    
    # Auto-detect camera type if not specified
    camera_type = args.camera_type
    if camera_type == 'auto':
        # Simple detection based on file structure
        if os.path.exists(os.path.join(args.base_dir, os.listdir(args.base_dir)[0], 'extrinsic_matrix.npy')):
            camera_type = 'realsense'
        else:
            camera_type = 'scannet'
        print(f"Auto-detected camera type: {camera_type}")
    
    # If save_path is not provided, use base_dir
    if args.save_path is None:
        args.save_path = args.base_dir
    
    # Build SAM2 model
    model_path = "./checkpoints/sam2.1_hiera_base_plus.pt"
    config_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_sam2(config_path, ckpt_path=model_path, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=sam_config['points_per_side'],
        pred_iou_thresh=sam_config['pred_iou_thresh'],
        stability_score_thresh=sam_config['stability_score_thresh'],
        output_mode="binary_mask",
        crop_n_layers=0,
        min_mask_region_area=sam_config['min_mask_region_area'],
    )
    
    # Get all frame paths with stride
    frame_paths = [os.path.join(args.base_dir, x) for x in sorted(os.listdir(args.base_dir)) 
                  if os.path.isdir(os.path.join(args.base_dir, x))][::args.frame_stride]
   
    # Initialize voxelizer
    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))
    
    # Process frames
    seg_pcd_custom(frame_paths = frame_paths, save_path = args.save_path, mask_generator = mask_generator, voxel_size = args.voxel_size, 
                  voxelizer = voxelize, th = args.th, point_stride = args.point_stride, camera_type=camera_type)
     

if __name__ == '__main__':
    main() 