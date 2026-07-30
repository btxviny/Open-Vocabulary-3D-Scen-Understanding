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
threshold = config.get('threshold', 0.01)
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



def merge_pointclouds(pcd1, pcd2, threshold=0.01):
    nn = NearestNeighbors(radius=threshold)
    nn.fit(pcd1['points'])
    indices = nn.radius_neighbors(pcd2['points'], return_distance=False)

    merged_points, merged_colors, merged_embeddings = [], [], []
    used = set()

    for i, neighbors in enumerate(indices):
        if len(neighbors) == 0:
            merged_points.append(pcd2['points'][i])
            merged_colors.append(pcd2['colors'][i])
            merged_embeddings.append(pcd2['embeddings'][i])
        else:
            neighbor_pts = np.vstack([pcd1['points'][j] for j in neighbors] + [pcd2['points'][i]])
            neighbor_clr = np.vstack([pcd1['colors'][j] for j in neighbors] + [pcd2['colors'][i]])
            neighbor_emb = np.vstack([pcd1['embeddings'][j] for j in neighbors] + [pcd2['embeddings'][i]])

            merged_points.append(np.mean(neighbor_pts, axis=0))
            merged_colors.append(np.mean(neighbor_clr, axis=0))
            emb = np.mean(neighbor_emb, axis=0)
            merged_embeddings.append(emb / (np.linalg.norm(emb) + 1e-8))
            used.update(neighbors)

    remaining = [i for i in range(len(pcd1['points'])) if i not in used]
    merged_points.extend(pcd1['points'][remaining])
    merged_colors.extend(pcd1['colors'][remaining])
    merged_embeddings.extend(pcd1['embeddings'][remaining])

    return {
        'points': np.array(merged_points, dtype=np.float32),
        'colors': np.array(merged_colors, dtype=np.float32),
        'embeddings': np.array(merged_embeddings, dtype=np.float32)
    }


def save_pcd(pcd, path):
    np.savez(path, points=pcd['points'], colors=pcd['colors'], embeddings=pcd['embeddings'])


def load_pcd(path):
    npz = np.load(path)
    return {'points': npz['points'], 'colors': npz['colors'], 'embeddings': npz['embeddings']}


def recursive_merge(paths, threshold, tmp_dir):
    iteration = 0

    while len(paths) > 1:
        print(f"\n🔁 Merge iteration {iteration + 1}: {len(paths)} pointclouds to merge")

        new_paths = []
        for i in tqdm(range(0, len(paths), 2), desc=f"Level {iteration + 1} merging"):
            if i + 1 >= len(paths):
                new_paths.append(paths[i])
                continue

            p1, p2 = paths[i], paths[i + 1]
            pcd1 = load_pcd(p1)
            pcd2 = load_pcd(p2)
            merged = merge_pointclouds(pcd1, pcd2, threshold=threshold)

            merged_path = os.path.join(tmp_dir, f"merged_iter{iteration}_{i//2}.npz")
            save_pcd(merged, merged_path)
            new_paths.append(merged_path)

        iteration += 1
        paths = new_paths
    print(f'\n✅ Saved merged pointcloud with {merged["points"].shape[0]} points.')

    return paths[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--logic', choices=["bin", "patch", "dilated"], default=logic)
    parser.add_argument('--frame_stride', type=int, default=frame_stride)
    parser.add_argument('--threshold', type=float, default=threshold)
    parser.add_argument('--downsample_rate', type=int, default=donwsample_rate)
    parser.add_argument('--tmp_dir', type=str, default='./tmp_merge')
    args = parser.parse_args()

    # Clean and prepare tmp dir
    if os.path.exists(args.tmp_dir):
        shutil.rmtree(args.tmp_dir)
    os.makedirs(args.tmp_dir, exist_ok=True)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Collect frame directories
    frame_dirs = [os.path.join(args.base_dir, x) for x in sorted(os.listdir(args.base_dir)) 
                  if os.path.isdir(os.path.join(args.base_dir, x))][::frame_stride]

    # Initial pairwise frame merges
    temp_paths = []
    for i in tqdm(range(0, len(frame_dirs), 2), desc="Initial pairwise merging"):
        pcd1 = extract_frame_pointcloud(frame_dirs[i], model, preprocess, args.logic, args.downsample_rate, device)
        if i + 1 < len(frame_dirs):
            pcd2 = extract_frame_pointcloud(frame_dirs[i + 1], model, preprocess, args.logic, args.downsample_rate, device)
            if pcd1 is None: pcd1 = pcd2
            elif pcd2 is not None:
                pcd1 = merge_pointclouds(pcd1, pcd2, threshold=args.threshold)
        if pcd1 is not None:
            temp_path = os.path.join(args.tmp_dir, f"pair_{i//2}.npz")
            save_pcd(pcd1, temp_path)
            temp_paths.append(temp_path)

    # Recursive merging
    final_path = recursive_merge(temp_paths, args.threshold, args.tmp_dir)
    # Save final result
    os.rename(final_path, args.save_path)
    #clean up temporary directory
    shutil.rmtree(args.tmp_dir)


if __name__ == "__main__":
    main()