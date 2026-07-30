"""
Propagate sparse CLIP embeddings and SAM3D cluster IDs onto the dense point cloud.

Usage (from repo root):
    uv run python -m scene_search.interpolate \
        --dense_pc_path   output/dense_pointcloud.npz \
        --sparse_pc_path  output/clip_pointcloud.npz \
        --segmented_pc_path output/segmented_pointcloud.npz
"""

import argparse

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from . import _config as _cfg_mod

_cfg = _cfg_mod.load()
_interpolation_k = _cfg["interpolation_k"]


def interpolate_embeddings(
    dense_points: np.ndarray,
    sparse_points: np.ndarray,
    sparse_embeddings: np.ndarray,
    k: int = 21,
    batch_size: int = 5012,
    sigma: float = 0.01,
) -> np.ndarray:
    """Gaussian-weighted KNN interpolation of CLIP embeddings."""
    N, D = len(dense_points), sparse_embeddings.shape[1]
    knn = NearestNeighbors(n_neighbors=k).fit(sparse_points)
    out = np.zeros((N, D), dtype=np.float32)

    for start in tqdm(range(0, N, batch_size), desc="Interpolating embeddings"):
        end = min(start + batch_size, N)
        dists, idxs = knn.kneighbors(dense_points[start:end])
        weights = np.exp(-(dists ** 2) / (2 * sigma ** 2))  # (B, k)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-8
        batch = (weights[:, :, None] * sparse_embeddings[idxs]).sum(axis=1)
        norms = np.linalg.norm(batch, axis=1, keepdims=True) + 1e-8
        out[start:end] = (batch / norms).astype(np.float32)

    return out


def interpolate_clusters(
    dense_points: np.ndarray,
    segmented_points: np.ndarray,
    cluster_labels: np.ndarray,
    k: int = 21,
    batch_size: int = 5012,
) -> np.ndarray:
    """Majority-vote KNN propagation of cluster IDs."""
    N = len(dense_points)
    knn = NearestNeighbors(n_neighbors=k).fit(segmented_points)
    out = np.zeros(N, dtype=np.int32)

    for start in tqdm(range(0, N, batch_size), desc="Propagating cluster IDs"):
        end = min(start + batch_size, N)
        _, idxs = knn.kneighbors(dense_points[start:end])
        mode = stats.mode(cluster_labels[idxs], axis=1, keepdims=False)
        out[start:end] = mode.mode

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_pc_path",     required=True)
    parser.add_argument("--sparse_pc_path",    required=True)
    parser.add_argument("--segmented_pc_path", required=True)
    parser.add_argument("--k", type=int, default=_interpolation_k)
    args = parser.parse_args()

    dense     = np.load(args.dense_pc_path)
    sparse    = np.load(args.sparse_pc_path)
    segmented = np.load(args.segmented_pc_path)

    embeddings = interpolate_embeddings(
        dense["points"], sparse["points"], sparse["embeddings"], k=args.k
    )
    clusters = interpolate_clusters(
        dense["points"], segmented["points"], segmented["cluster_labels"], k=args.k
    )

    np.savez(
        args.dense_pc_path,
        points=dense["points"],
        colors=dense["colors"],
        embeddings=embeddings,
        cluster_labels=clusters,
    )
    print(f"Saved {args.dense_pc_path} with embeddings and cluster IDs.")
