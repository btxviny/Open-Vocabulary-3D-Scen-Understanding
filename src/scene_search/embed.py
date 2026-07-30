"""
CLIP embedding for SAM2 masks + recursive disk-spill merge.

Reads per-frame mask_*.png and object_*.pcd files (written by segment.py),
encodes each mask crop with CLIP, attaches the embedding to every point in the
corresponding PCD, then merges all frames into a single .npz via pairwise
recursive voxel-averaging — at most 2 frame clouds live in RAM at any time.

Usage (from repo root):
    uv run python -m src.scene_search.embed \\
        --base_dir /path/to/scene_frames \\
        --save_path output/embedded_pointcloud.npz
"""

import argparse
import glob
import os
import re
import shutil
import tempfile
from pathlib import Path

import clip
import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm

from .utils import load_config

_cfg = load_config()
_frame_stride    = _cfg.get("frame_stride", 10)
_downsample_rate = _cfg.get("downsample_rate", 3)
_clip_weights    = _cfg.get("weights", "ViT-B/32")
_logic           = _cfg.get("logic", "patch")
_voxel_size      = _cfg.get("sam2", {}).get("merge_radius", 0.02)

_BATCH_SIZE = 8


# ── Image crop helpers ────────────────────────────────────────────────────────

def _patch_crop(mask: np.ndarray, image: np.ndarray, expand: float = 0.3) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image
    h, w = mask.shape
    xpad = int((xs.max() - xs.min()) * expand)
    ypad = int((ys.max() - ys.min()) * expand)
    x0 = max(xs.min() - xpad, 0);  x1 = min(xs.max() + xpad, w - 1)
    y0 = max(ys.min() - ypad, 0);  y1 = min(ys.max() + ypad, h - 1)
    return image[y0:y1 + 1, x0:x1 + 1]


def _dilate_mask(mask: np.ndarray, ksize: int = 25, iters: int = 5) -> np.ndarray:
    u8 = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return (cv2.dilate(u8, kernel, iterations=iters) > 0).astype(np.uint8)


def _masked_crop(mask: np.ndarray, image: np.ndarray, logic: str) -> np.ndarray:
    if logic == "bin":
        return image * mask[:, :, None].astype(np.uint8)
    if logic == "patch":
        return _patch_crop(mask, image)
    if logic == "dilated":
        return image * _dilate_mask(mask)[:, :, None]
    raise ValueError(f"Unknown logic: {logic!r}")


# ── Sort helpers ──────────────────────────────────────────────────────────────

def _sort_by_index(paths: list[str]) -> list[str]:
    return sorted(paths, key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))


# ── Voxel merge ───────────────────────────────────────────────────────────────

def _voxel_average(
    pts: np.ndarray, cols: np.ndarray, embs: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(pts) == 0:
        return pts, cols, embs

    idx = np.floor(pts / voxel_size).astype(np.int64)
    s = np.int64(1_000_003)
    keys = idx[:, 0] * s * s + idx[:, 1] * s + idx[:, 2]

    _, inverse = np.unique(keys, return_inverse=True)
    n = int(inverse.max()) + 1
    D = embs.shape[1]

    out_pts  = np.zeros((n, 3), np.float32)
    out_cols = np.zeros((n, 3), np.float32)
    out_embs = np.zeros((n, D), np.float32)
    counts   = np.zeros(n,      np.float32)

    np.add.at(out_pts,  inverse, pts)
    np.add.at(out_cols, inverse, cols)
    np.add.at(out_embs, inverse, embs)
    np.add.at(counts,   inverse, 1.0)

    out_pts  /= counts[:, None]
    out_cols /= counts[:, None]
    norms = np.linalg.norm(out_embs, axis=1, keepdims=True)
    out_embs /= norms + 1e-8
    return out_pts, out_cols, out_embs


# ── Disk-spill merge ──────────────────────────────────────────────────────────

def _save(path: str, pts, cols, embs) -> None:
    np.savez(path, points=pts, colors=cols, embeddings=embs)


def _load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["points"], d["colors"], d["embeddings"]


def _merge_two_files(a: str, b: str, out: str, voxel_size: float) -> None:
    pts_a, cols_a, embs_a = _load(a)
    pts_b, cols_b, embs_b = _load(b)
    pts  = np.vstack([pts_a,  pts_b]);  del pts_a,  pts_b
    cols = np.vstack([cols_a, cols_b]); del cols_a, cols_b
    embs = np.vstack([embs_a, embs_b]); del embs_a, embs_b
    pts, cols, embs = _voxel_average(pts, cols, embs, voxel_size)
    _save(out, pts, cols, embs)


def _recursive_merge(paths: list[str], tmp_dir: str, voxel_size: float) -> str:
    iteration = 0
    while len(paths) > 1:
        print(f"  Merge pass {iteration + 1}: {len(paths)} files")
        next_paths = []
        for i in tqdm(range(0, len(paths), 2), desc=f"  Pass {iteration + 1}"):
            if i + 1 >= len(paths):
                next_paths.append(paths[i])
                continue
            out = os.path.join(tmp_dir, f"merge_{iteration}_{i // 2}.npz")
            _merge_two_files(paths[i], paths[i + 1], out, voxel_size)
            os.remove(paths[i])
            os.remove(paths[i + 1])
            next_paths.append(out)
        paths = next_paths
        iteration += 1
    return paths[0]


# ── Per-frame processing ──────────────────────────────────────────────────────

def _process_frame(
    frame_dir: str,
    clip_model,
    clip_preprocess,
    logic: str,
    downsample_rate: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    rgb_path = os.path.join(frame_dir, "rgb_image.png")
    if not os.path.exists(rgb_path):
        return None

    image_np = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)

    mask_paths = _sort_by_index(glob.glob(os.path.join(frame_dir, "mask_*.png")))
    pcd_paths  = _sort_by_index(glob.glob(os.path.join(frame_dir, "object_*.pcd")))
    n = min(len(mask_paths), len(pcd_paths))
    if n == 0:
        return None

    all_pts:  list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_embs: list[np.ndarray] = []

    for i in range(0, n, _BATCH_SIZE):
        batch_masks = [np.array(Image.open(p).convert("L")) for p in mask_paths[i:i + _BATCH_SIZE]]
        batch_pcds  = [o3d.io.read_point_cloud(p).uniform_down_sample(downsample_rate)
                       for p in pcd_paths[i:i + _BATCH_SIZE]]

        crops = [_masked_crop(m, image_np, logic) for m in batch_masks]
        tensors = [clip_preprocess(Image.fromarray(c)).unsqueeze(0).to(device) for c in crops]

        with torch.no_grad():
            embs = clip_model.encode_image(torch.cat(tensors)).cpu()
        embs = (embs / embs.norm(dim=1, keepdim=True)).numpy().astype(np.float32)

        for pcd, emb in zip(batch_pcds, embs):
            pts  = np.asarray(pcd.points, dtype=np.float32)
            cols = np.asarray(pcd.colors, dtype=np.float32)
            if len(pts) == 0:
                continue
            all_pts.append(pts)
            all_cols.append(cols)
            all_embs.append(np.tile(emb, (len(pts), 1)))

    if not all_pts:
        return None

    return np.vstack(all_pts), np.vstack(all_cols), np.vstack(all_embs)


# ── Main ──────────────────────────────────────────────────────────────────────

def create_embedded_pointcloud(
    base_dir: str,
    save_path: str,
    logic: str = _logic,
    frame_stride: int = _frame_stride,
    downsample_rate: int = _downsample_rate,
    voxel_size: float = _voxel_size,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ({_clip_weights}) on {device}…")
    clip_model, clip_preprocess = clip.load(_clip_weights, device=device)

    frame_dirs = sorted(
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )[::frame_stride]

    tmp_dir = tempfile.mkdtemp(prefix="embed_")
    print(f"Temp dir: {tmp_dir}")
    frame_npzs: list[str] = []

    try:
        for frame_idx, frame_dir in enumerate(tqdm(frame_dirs, desc="CLIP embedding frames")):
            result = _process_frame(frame_dir, clip_model, clip_preprocess,
                                    logic, downsample_rate, device)
            if result is None:
                continue

            pts, cols, embs = result
            pts, cols, embs = _voxel_average(pts, cols, embs, voxel_size)

            out = os.path.join(tmp_dir, f"frame_{frame_idx:05d}.npz")
            _save(out, pts, cols, embs)
            frame_npzs.append(out)

        if not frame_npzs:
            raise ValueError(f"No mask/PCD files found under {base_dir}. Run segment.py first.")

        print(f"\n{len(frame_npzs)} frame files — recursive merge…")
        final_tmp = _recursive_merge(frame_npzs, tmp_dir, voxel_size)

        pts, cols, embs = _load(final_tmp)
        print(f"Merged: {len(pts):,} points — outlier removal…")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        _, keep = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)

        np.savez(save_path, points=pts[keep], colors=cols[keep], embeddings=embs[keep])
        print(f"Saved {save_path}  ({len(keep):,} points)")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP embeddings from SAM2 masks → embedded point cloud")
    parser.add_argument("--base_dir",        required=True)
    parser.add_argument("--save_path",       required=True)
    parser.add_argument("--logic",           choices=["bin", "patch", "dilated"], default=_logic)
    parser.add_argument("--frame_stride",    type=int,   default=_frame_stride)
    parser.add_argument("--downsample_rate", type=int,   default=_downsample_rate)
    parser.add_argument("--voxel_size",      type=float, default=_voxel_size)
    args = parser.parse_args()

    create_embedded_pointcloud(
        args.base_dir, args.save_path,
        logic=args.logic,
        frame_stride=args.frame_stride,
        downsample_rate=args.downsample_rate,
        voxel_size=args.voxel_size,
    )
