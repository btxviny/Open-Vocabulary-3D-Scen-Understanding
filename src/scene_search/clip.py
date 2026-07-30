"""
Compute per-object CLIP embeddings from SAM2 masks and build a sparse point cloud.

For each frame, sam3d.py saves:
  mask_<id>.png      — binary segmentation mask
  object_<id>.pcd    — 3D points (world space, colours in 0–1) for that mask

This module reads those files, encodes the masked/cropped image region with CLIP,
and attaches the embedding to each object's points.  Points from all frames are
then merged by proximity (radius = merge_radius) and outlier-filtered.

Usage (from repo root):
    uv run python -m scene_search.clip \
        --base_dir /path/to/scene_frames \
        --save_path output/clip_pointcloud.npz \
        --logic patch
"""

import glob
import os
import re

import clip
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from sklearn.neighbors import KDTree
from tqdm import tqdm

from . import _config as _cfg_mod

_cfg = _cfg_mod.load()
_frame_stride    = _cfg["frame_stride"]
_downsample_rate = _cfg["downsample_rate"]
_weights         = _cfg.get("weights", "ViT-B/32")
_default_logic   = _cfg["logic"]

_BATCH_SIZE = 8


# ── Image crop helpers ────────────────────────────────────────────────────────

def _patch_from_mask(mask: np.ndarray, image: np.ndarray, expand: float = 0.3) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros_like(image)
    h, w = mask.shape
    xpad = int((xs.max() - xs.min()) * expand)
    ypad = int((ys.max() - ys.min()) * expand)
    x0 = max(xs.min() - xpad, 0);  x1 = min(xs.max() + xpad, w - 1)
    y0 = max(ys.min() - ypad, 0);  y1 = min(ys.max() + ypad, h - 1)
    return image[y0:y1 + 1, x0:x1 + 1]


def _dilate_mask(mask: np.ndarray, ksize: int = 25, iters: int = 5) -> np.ndarray:
    import cv2
    u8 = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return (cv2.dilate(u8, kernel, iterations=iters) > 0).astype(np.uint8)


def _masked_crop(mask_np: np.ndarray, image_np: np.ndarray, logic: str) -> np.ndarray:
    if logic == "bin":
        return image_np * mask_np[:, :, None]
    if logic == "patch":
        return _patch_from_mask(mask_np, image_np)
    if logic == "dilated":
        return image_np * _dilate_mask(mask_np)[:, :, None]
    raise ValueError(f"Unknown logic: {logic!r}")


# ── Point cloud helpers ───────────────────────────────────────────────────────

def _load_object_pcd(path: str, downsample: int) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if downsample > 1:
        pcd = pcd.uniform_down_sample(every_k_points=downsample)
    return pcd


def _sort_by_index(paths: list[str]) -> list[str]:
    return sorted(paths, key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))


# ── Main ──────────────────────────────────────────────────────────────────────

def create_clip_pointcloud(
    base_dir: str,
    save_path: str,
    merge_radius: float = 0.01,
    logic: str = _default_logic,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(_weights, device=device)

    frame_dirs = [
        os.path.join(base_dir, d)
        for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d))
    ][::_frame_stride]

    all_points:     list[np.ndarray] = []
    all_colors:     list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []

    for frame_dir in tqdm(frame_dirs, desc="CLIP: encoding frames"):
        rgb_path = os.path.join(frame_dir, "rgb_image.png")
        if not os.path.exists(rgb_path):
            continue
        image_np = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)

        mask_paths = _sort_by_index(glob.glob(os.path.join(frame_dir, "mask_*.png")))
        pcd_paths  = _sort_by_index(glob.glob(os.path.join(frame_dir, "object_*.pcd")))
        n = min(len(mask_paths), len(pcd_paths))
        if n == 0:
            continue

        for i in range(0, n, _BATCH_SIZE):
            batch_masks = [np.array(Image.open(p).convert("L")) for p in mask_paths[i:i + _BATCH_SIZE]]
            batch_pcds  = [_load_object_pcd(p, _downsample_rate) for p in pcd_paths[i:i + _BATCH_SIZE]]

            crops   = [_masked_crop(m, image_np, logic) for m in batch_masks]
            tensors = [preprocess(Image.fromarray(c)).unsqueeze(0).to(device) for c in crops]

            with torch.no_grad():
                embs = model.encode_image(torch.cat(tensors)).cpu()
            embs = (embs / embs.norm(dim=1, keepdim=True)).numpy().astype(np.float32)

            for pcd, emb in zip(batch_pcds, embs):
                pts  = np.asarray(pcd.points, dtype=np.float32)
                cols = np.asarray(pcd.colors, dtype=np.float32)  # already 0–1
                if len(pts) == 0:
                    continue
                all_points.append(pts)
                all_colors.append(cols)
                all_embeddings.append(np.tile(emb, (len(pts), 1)))

    if not all_points:
        raise ValueError(f"No object PCDs found under {base_dir}")

    pts  = np.vstack(all_points)
    cols = np.vstack(all_colors)
    embs = np.vstack(all_embeddings)

    # ── Merge nearby points (average positions, colours, embeddings) ──────────
    tree = KDTree(pts)
    visited = np.zeros(len(pts), dtype=bool)
    merged_pts, merged_cols, merged_embs = [], [], []

    for batch_start in tqdm(range(0, len(pts), 500), desc="CLIP: merging points"):
        batch_end = min(batch_start + 500, len(pts))
        candidates = np.where(~visited[batch_start:batch_end])[0] + batch_start
        if len(candidates) == 0:
            continue
        neighbor_sets = tree.query_radius(pts[candidates], r=merge_radius)
        for idx, neighbors in zip(candidates, neighbor_sets):
            if visited[idx]:
                continue
            visited[neighbors] = True
            merged_pts.append(pts[neighbors].mean(axis=0).astype(np.float32))
            merged_cols.append(cols[neighbors].mean(axis=0).astype(np.float32))
            avg_emb = embs[neighbors].mean(axis=0)
            avg_emb /= np.linalg.norm(avg_emb) + 1e-8
            merged_embs.append(avg_emb.astype(np.float32))

    merged_pts  = np.array(merged_pts,  dtype=np.float32)
    merged_cols = np.array(merged_cols, dtype=np.float32)
    merged_embs = np.array(merged_embs, dtype=np.float32)

    # ── Statistical outlier removal ───────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pts)
    _, keep = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)

    np.savez(
        save_path,
        points=merged_pts[keep],
        colors=merged_cols[keep],
        embeddings=merged_embs[keep],
    )
    print(f"Saved {save_path} ({len(keep)} points)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",     required=True)
    parser.add_argument("--save_path",    required=True)
    parser.add_argument("--merge_radius", type=float, default=0.01)
    parser.add_argument("--logic",        choices=["bin", "patch", "dilated"], default=_default_logic)
    args = parser.parse_args()
    create_clip_pointcloud(args.base_dir, args.save_path, args.merge_radius, args.logic)
