import os
import cv2
import glob
import re
import open3d as o3d
import clip
import torch
import tqdm
import numpy as np
import argparse
from PIL import Image
from sklearn.neighbors import KDTree
from . import _config as _cfg_mod

_cfg = _cfg_mod.load()
frame_stride    = _cfg['frame_stride']
downsample_rate = _cfg['downsample_rate']
weights         = _cfg.get('weights', 'ViT-B/32')
logic           = _cfg['logic']
BATCH_SIZE = 8

# --- Utility Functions --- #
def dilate_mask(mask: np.ndarray, kernel_size: int = 25, iterations: int = 5) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    return (dilated > 0).astype(np.uint8)

def patch_from_mask(mask: np.ndarray, image: np.ndarray, expand_ratio: float = 0.3) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.squeeze()
    y_idx, x_idx = np.where(mask > 0)
    if len(x_idx) == 0 or len(y_idx) == 0:
        return np.zeros_like(image)
    x_min, x_max = x_idx.min(), x_idx.max()
    y_min, y_max = y_idx.min(), y_idx.max()
    h, w = mask.shape
    x_pad, y_pad = int((x_max - x_min) * expand_ratio), int((y_max - y_min) * expand_ratio)
    x_min, x_max = max(x_min - x_pad, 0), min(x_max + x_pad, w - 1)
    y_min, y_max = max(y_min - y_pad, 0), min(y_max + y_pad, h - 1)
    return image[y_min:y_max+1, x_min:x_max+1]

# --- Pointcloud Preprocessing --- #
def preprocess_pointcloud(pcd_path, downsample_rate=10):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = pcd.uniform_down_sample(every_k_points=downsample_rate)
    # Convert points and colors to float32 numpy arrays for memory efficiency
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points, dtype=np.float32))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors, dtype=np.float32))
    return pcd

# --- Main Function --- #
def create_clip_pointcloud(base_dir, save_path, merge_radius=0.01, logic="bin"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(weights, device=device)

    frames_dir = [os.path.join(base_dir, x) for x in sorted(os.listdir(base_dir)) 
                  if os.path.isdir(os.path.join(base_dir, x))][::frame_stride]

    all_points = []
    all_embeddings = []
    all_colors = []

    for frame_dir in tqdm.tqdm(frames_dir, desc="Processing frames"):
        image_path = os.path.join(frame_dir, 'rgb_image.png')
        if not os.path.exists(image_path): continue
        image = Image.open(image_path)
        image_np = np.array(image).astype(np.uint8)

        mask_paths = sorted(glob.glob(os.path.join(frame_dir, 'mask_*.png')),
                            key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        pcd_paths = sorted(glob.glob(os.path.join(frame_dir, 'object_*.pcd')),
                           key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))

        for idx in range(0, len(mask_paths), BATCH_SIZE):
            masks = [Image.open(p).convert("L") for p in mask_paths[idx:idx + BATCH_SIZE]]
            pcds = [preprocess_pointcloud(p, downsample_rate=downsample_rate) for p in pcd_paths[idx:idx + BATCH_SIZE]]

            if logic == "bin":
                masked_imgs = [image_np * np.array(m)[:, :, None].astype(np.uint8) for m in masks]
            elif logic == "patch":
                masked_imgs = [patch_from_mask(np.array(m), image_np) for m in masks]
            elif logic == "dilated":
                dilated = [dilate_mask(np.array(m)) for m in masks]
                masked_imgs = [image_np * d[:, :, None] for d in dilated]
            else:
                raise ValueError(f"Unknown logic type: {logic}")

            imgs_tensor = [preprocess(Image.fromarray(m)).unsqueeze(0).to(device) for m in masked_imgs]
            with torch.no_grad():
                emb = model.encode_image(torch.cat(imgs_tensor)).cpu()
            emb = emb / emb.norm(dim=1, keepdim=True)
            emb_np = emb.numpy().astype(np.float32)

            for pcd, e in zip(pcds, emb_np):
                n_points = np.array(pcd.points, dtype=np.float32)
                n_colors = np.array(pcd.colors, dtype=np.float32)
                n_embeddings = np.repeat(e[None, :], len(n_points), axis=0).astype(np.float32)

                all_points.append(n_points)
                all_colors.append(n_colors)
                all_embeddings.append(n_embeddings)

    # Stack everything
    all_points = np.vstack(all_points).astype(np.float32)
    all_colors = np.vstack(all_colors).astype(np.float32)
    all_embeddings = np.vstack(all_embeddings).astype(np.float32)

    # Merge via KDTree
    tree = KDTree(all_points)
    visited = np.zeros(len(all_points), dtype=bool)
    merged_points, merged_colors, merged_embeddings = [], [], []

    indices = np.arange(len(all_points))
    for batch_start in tqdm.trange(0, len(all_points), 500, desc="Merging points (batched)"):
        batch_end = min(batch_start + 500, len(all_points))
        batch_indices = indices[batch_start:batch_end]
        unvisited_mask = ~visited[batch_indices]
        batch_indices = batch_indices[unvisited_mask]
        if len(batch_indices) == 0:
            continue

        neighbors_list = tree.query_radius(all_points[batch_indices], r=merge_radius)

        for i, neighbors in zip(batch_indices, neighbors_list):
            if visited[i]:
                continue
            visited[neighbors] = True
            merged_points.append(np.mean(all_points[neighbors], axis=0).astype(np.float32))
            merged_colors.append(np.mean(all_colors[neighbors], axis=0).astype(np.float32))
            merged_emb = np.mean(all_embeddings[neighbors], axis=0)
            merged_emb /= np.linalg.norm(merged_emb) + 1e-8
            merged_embeddings.append(merged_emb.astype(np.float32))

    # Convert merged lists to arrays
    merged_points = np.array(merged_points, dtype=np.float32)
    merged_colors = np.array(merged_colors, dtype=np.float32)
    merged_embeddings = np.array(merged_embeddings, dtype=np.float32)

    # Remove outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors / 255.0)

    _, ind_stat = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
    merged_points = merged_points[ind_stat]
    merged_colors = merged_colors[ind_stat]
    merged_embeddings = merged_embeddings[ind_stat]

    # Save
    np.savez(save_path,
             points=merged_points,
             colors=merged_colors,
             embeddings=merged_embeddings)

    print(f"Saved {save_path} with {len(merged_points)} merged points.")

# --- Entry Point --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CLIP-colored 3D point cloud with one-time merge.")
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--merge_radius', type=float, default=0.01)
    parser.add_argument('--logic', type=str, choices=["bin", "patch", "dilated"], default=logic)
    args = parser.parse_args()

    create_clip_pointcloud(args.base_dir, args.save_path, args.merge_radius, args.logic)