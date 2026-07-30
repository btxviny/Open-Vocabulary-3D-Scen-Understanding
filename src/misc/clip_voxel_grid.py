import os
import yaml
import shutil
import glob
import re
import argparse
import numpy as np
import torch
import clip
import cv2
from tqdm import tqdm
from PIL import Image
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
logic = config.get('logic', 'bin')
frame_stride = config.get('frame_stride', 20)
voxel_size = config.get('voxel_size', 0.01)
donwsample_rate = config.get('downsample_rate', 10)



def dilate_mask(mask, kernel_size=25, iterations=5):
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return (cv2.dilate(mask, kernel, iterations=iterations) > 0).astype(np.uint8)


def patch_from_mask(mask, image, expand_ratio=0.05):
    y_idx, x_idx = np.where(mask > 0)
    if len(x_idx) == 0 or len(y_idx) == 0:
        return np.zeros_like(image)
    x_min, x_max = x_idx.min(), x_idx.max()
    y_min, y_max = y_idx.min(), y_idx.max()
    h, w = mask.shape
    x_pad = int((x_max - x_min) * expand_ratio)
    y_pad = int((y_max - y_min) * expand_ratio)
    x_min, x_max = max(x_min - x_pad, 0), min(x_max + x_pad, w - 1)
    y_min, y_max = max(y_min - y_pad, 0), min(y_max + y_pad, h - 1)
    return image[y_min:y_max + 1, x_min:x_max + 1]


def preprocess_pointcloud(pcd_path, downsample_rate):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = pcd.uniform_down_sample(downsample_rate)
    return np.asarray(pcd.points, dtype=np.float32), np.asarray(pcd.colors, dtype=np.float32)


def extract_frame_pointcloud(frame_dir, model, preprocess, logic, downsample_rate, device, batch_size=8):
    image_path = os.path.join(frame_dir, 'rgb_image.png')
    if not os.path.exists(image_path):
        return None

    image = Image.open(image_path)
    image_np = np.array(image)

    mask_paths = sorted(glob.glob(os.path.join(frame_dir, 'mask_*.png')),
                        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
    pcd_paths = sorted(glob.glob(os.path.join(frame_dir, 'object_*.pcd')),
                       key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))

    if not mask_paths or not pcd_paths:
        return None

    crops, pcd_infos = [], []

    for mask_path, pcd_path in zip(mask_paths, pcd_paths):
        mask = Image.open(mask_path).convert("L")
        if logic == "bin":
            masked_img = image_np * np.array(mask)[:, :, None].astype(np.uint8)
        elif logic == "patch":
            masked_img = patch_from_mask(np.array(mask), image_np)
        elif logic == "dilated":
            dilated = dilate_mask(np.array(mask))
            masked_img = image_np * dilated[:, :, None]
        else:
            raise ValueError(f"Unknown logic: {logic}")

        try:
            crop = preprocess(Image.fromarray(masked_img))
        except Exception as e:
            continue  # skip invalid crop
        crops.append(crop)
        pcd_infos.append(pcd_path)

    if not crops:
        return None

    # Batch forward through CLIP
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch = torch.stack(crops[i:i+batch_size]).to(device)
            embs = model.encode_image(batch).cpu()
            embs = embs / embs.norm(dim=1, keepdim=True)
            all_embeddings.append(embs)
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)

    # Process point clouds and duplicate embeddings
    all_points, all_colors, all_repeated_embs = [], [], []

    for emb, pcd_path in zip(all_embeddings, pcd_infos):
        points, colors = preprocess_pointcloud(pcd_path, downsample_rate)
        if len(points) == 0:
            continue
        emb_repeated = np.repeat(emb[None, :], len(points), axis=0)
        all_points.append(points)
        all_colors.append(colors)
        all_repeated_embs.append(emb_repeated)

    if not all_points:
        return None

    return {
        'points': np.vstack(all_points),
        'colors': np.vstack(all_colors),
        'embeddings': np.vstack(all_repeated_embs)
    }

def merge_voxels(pcds, voxel_size=0.05):
    """
    Merge a list of PCDs into a voxel grid, averaging points, colors, and embeddings inside each voxel equally.
    """
    all_points = np.vstack([pcd['points'] for pcd in pcds])
    all_colors = np.vstack([pcd['colors'] for pcd in pcds])
    all_embs = np.vstack([pcd['embeddings'] for pcd in pcds])

    voxel_indices = np.floor(all_points / voxel_size).astype(np.int32)
    voxel_keys, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    voxel_points = []
    voxel_colors = []
    voxel_embs = []

    for i in tqdm(range(len(voxel_keys)), desc="Merging voxels", total=len(voxel_keys)):
        idxs = np.where(inverse_indices == i)[0]
        points = all_points[idxs]
        colors = all_colors[idxs]
        embs = all_embs[idxs]

        avg_point = points.mean(axis=0)
        avg_color = colors.mean(axis=0)
        avg_emb = embs.mean(axis=0)
        # Normalize embedding vector
        avg_emb /= np.linalg.norm(avg_emb) + 1e-8

        voxel_points.append(avg_point)
        voxel_colors.append(avg_color)
        voxel_embs.append(avg_emb)

    return {
        'points': np.array(voxel_points, dtype=np.float32),
        'colors': np.array(voxel_colors, dtype=np.float32),
        'embeddings': np.array(voxel_embs, dtype=np.float32)
    }



def save_pcd(pcd, path):
    np.savez(path, points=pcd['points'], colors=pcd['colors'], embeddings=pcd['embeddings'])


def load_pcd(path):
    npz = np.load(path)
    return {'points': npz['points'], 'colors': npz['colors'], 'embeddings': npz['embeddings']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--logic', choices=["bin", "patch", "dilated"], default=logic)
    parser.add_argument('--frame_stride', type=int, default=frame_stride)
    parser.add_argument('--downsample_rate', type=int, default=donwsample_rate)
    parser.add_argument('--voxel_size', type=int, default=voxel_size)
   
    args = parser.parse_args()

    

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Collect frame directories
    frame_dirs = [os.path.join(args.base_dir, x) for x in sorted(os.listdir(args.base_dir)) 
                  if os.path.isdir(os.path.join(args.base_dir, x))][::frame_stride]

    pcds = []

    for i, frame_dir in enumerate(tqdm(frame_dirs, desc="Processing frames")):
        result = extract_frame_pointcloud(frame_dir, model, preprocess, args.logic, args.downsample_rate, device)
        if result is not None:
            pcds.append(result)

    if not pcds:
        print("No valid pointclouds found.")
        return

    merged_pcd = merge_voxels(pcds, voxel_size=args.voxel_size)

    print(f"Saving merged pointcloud to {args.save_path} with {merged_pcd['points'].shape[0]} points.")
    save_pcd(merged_pcd, args.save_path)
    


if __name__ == "__main__":
    main()