"""
Gaussian-weighted KNN interpolation of CLIP embeddings onto the dense point cloud.

Reads embedded_pointcloud.npz (sparse, from clip_embed) and dense_pointcloud.npz,
propagates embeddings from sparse to dense via weighted nearest-neighbour lookup,
and writes the result back into dense_pointcloud.npz (adds the 'embeddings' key).

Usage (from repo root):
    uv run python -m src.scene_search.interpolate \\
        --dense_pc_path  output/dense_pointcloud.npz \\
        --sparse_pc_path output/embedded_pointcloud.npz
"""

import argparse

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .utils import load_config

_cfg             = load_config()
_interpolation_k = _cfg.get("interpolation_k", 21)


def interpolate_embeddings(
    dense_points: np.ndarray,
    sparse_points: np.ndarray,
    sparse_embeddings: np.ndarray,
    k: int = 21,
    batch_size: int = 5012,
    sigma: float = 0.01,
) -> np.ndarray:
    """Gaussian-weighted KNN: for each dense point, average the k nearest
    sparse embeddings weighted by exp(-d²/2σ²), then L2-normalise."""
    N, D = len(dense_points), sparse_embeddings.shape[1]
    knn  = NearestNeighbors(n_neighbors=k).fit(sparse_points)
    out  = np.zeros((N, D), dtype=np.float32)

    for start in tqdm(range(0, N, batch_size), desc="Interpolating embeddings"):
        end = min(start + batch_size, N)
        dists, idxs = knn.kneighbors(dense_points[start:end])
        weights = np.exp(-(dists ** 2) / (2 * sigma ** 2))
        weights /= weights.sum(axis=1, keepdims=True) + 1e-8
        batch = (weights[:, :, None] * sparse_embeddings[idxs]).sum(axis=1)
        norms = np.linalg.norm(batch, axis=1, keepdims=True) + 1e-8
        out[start:end] = (batch / norms).astype(np.float32)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate CLIP embeddings onto dense cloud")
    parser.add_argument("--dense_pc_path",  required=True, help="dense_pointcloud.npz (updated in-place)")
    parser.add_argument("--sparse_pc_path", required=True, help="embedded_pointcloud.npz")
    parser.add_argument("--k",              type=int, default=_interpolation_k)
    parser.add_argument("--sigma",          type=float, default=0.01)
    args = parser.parse_args()

    dense  = np.load(args.dense_pc_path)
    sparse = np.load(args.sparse_pc_path)

    embeddings = interpolate_embeddings(
        dense["points"], sparse["points"], sparse["embeddings"],
        k=args.k, sigma=args.sigma,
    )

    np.savez(
        args.dense_pc_path,
        points=dense["points"],
        colors=dense["colors"],
        embeddings=embeddings,
    )
    print(f"Updated {args.dense_pc_path} with interpolated embeddings ({len(embeddings):,} points)")
