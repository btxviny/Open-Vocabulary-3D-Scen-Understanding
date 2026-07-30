"""
Example Usage:
    python interpolate_clip_embeddings.py --sparse_pc_path clip_pointcloud_expanded_patches.npz --dense_pc_path dense_pointcloud.npz --segmented_pc_path segmented_pointcloud.npz --k 11
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import argparse
from scipy import stats

from . import _config as _cfg_mod

_cfg = _cfg_mod.load()
interpolation_k = _cfg['interpolation_k']
print(f"Interpolation neighbours: {interpolation_k}")


def interpolate_embeddings(dense_points, sparse_points, sparse_embeddings, k=21, batch_size=5012, sigma=0.01):
    D = sparse_embeddings.shape[1]
    N = len(dense_points)

    knn = NearestNeighbors(n_neighbors=k).fit(sparse_points)
    embeddings = np.zeros((N, D), dtype=np.float32)

    for batch_start in tqdm(range(0, N, batch_size), desc="Interpolating CLIP embeddings (attenuated)", unit="batch"):
        batch_end = min(batch_start + batch_size, N)
        batch_points = dense_points[batch_start:batch_end]

        distances, idxs = knn.kneighbors(batch_points)
        batch_embeddings = np.zeros((batch_end - batch_start, D), dtype=np.float32)

        for i in range(batch_end - batch_start):
            neighbor_embs = sparse_embeddings[idxs[i]]
            neighbor_distances = distances[i]

            # Gaussian weights
            weights = np.exp(- (neighbor_distances ** 2) / (2 * sigma ** 2))
            weights /= (np.sum(weights) + 1e-8)

            interp_emb = np.dot(weights, neighbor_embs)
            interp_emb /= np.linalg.norm(interp_emb) + 1e-8

            batch_embeddings[i] = interp_emb

        embeddings[batch_start:batch_end] = batch_embeddings

    return embeddings

def interpolate_embeddings_linear(dense_points, sparse_points, sparse_embeddings, k=21, batch_size=5012):
    D = sparse_embeddings.shape[1]
    N = len(dense_points)

    knn = NearestNeighbors(n_neighbors=k).fit(sparse_points)
    embeddings = np.zeros((N, D), dtype=np.float32)

    for batch_start in tqdm(range(0, N, batch_size), desc="Interpolating CLIP embeddings (linearly)", unit="batch"):
        batch_end = min(batch_start + batch_size, N)
        batch_points = dense_points[batch_start:batch_end]

        distances, idxs = knn.kneighbors(batch_points)
        batch_embeddings = np.zeros((batch_end - batch_start, D), dtype=np.float32)

        for i in range(batch_end - batch_start):
            neighbor_embs = sparse_embeddings[idxs[i]]

            # Uniform averaging
            interp_emb = np.mean(neighbor_embs, axis=0)
            interp_emb /= np.linalg.norm(interp_emb) + 1e-8

            batch_embeddings[i] = interp_emb

        embeddings[batch_start:batch_end] = batch_embeddings

    return embeddings


def interpolate_clusters(dense_points, segmented_points, cluster_labels, k=21, batch_size=5012):
    N = len(dense_points)
    knn = NearestNeighbors(n_neighbors=k).fit(segmented_points)
    interpolated_clusters = np.zeros(N, dtype=np.int32)

    for batch_start in tqdm(range(0, N, batch_size), desc="Interpolating cluster labels", unit="batch"):
        batch_end = min(batch_start + batch_size, N)
        batch_points = dense_points[batch_start:batch_end]

        _, idxs = knn.kneighbors(batch_points)
        
        for i in range(batch_end - batch_start):
            neighbor_clusters = cluster_labels[idxs[i]]
            # Use mode to get most common cluster ID among neighbors, explicitly set keepdims=False
            mode_result = stats.mode(neighbor_clusters, keepdims=False)
            interpolated_clusters[batch_start + i] = mode_result.mode

    return interpolated_clusters

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse_pc_path', type=str, required=True, help="Path to sparse pointcloud with embeddings.")
    parser.add_argument('--dense_pc_path', type=str, required=True, help="Path to dense pointcloud to interpolate embeddings onto.")
    parser.add_argument('--segmented_pc_path', type=str, required=True, help="Path to segmented pointcloud with cluster IDs.")
    args = parser.parse_args()

    # Load data
    dense = np.load(args.dense_pc_path)
    dense_points = dense['points']
    dense_colors = dense['colors']
    
    
    # Interpolation of CLIP embeddings
    sparse = np.load(args.sparse_pc_path)
    sparse_points = sparse['points']
    sparse_embeddings = sparse['embeddings']
    inter_embeddings = interpolate_embeddings_linear(
        dense_points,
        sparse_points,
        sparse_embeddings,
        k=interpolation_k
    )

    del sparse, sparse_points, sparse_embeddings

    # Interpolation of cluster IDs
    segmented = np.load(args.segmented_pc_path)
    segmented_points = segmented['points']
    cluster_labels = segmented['cluster_labels']
    inter_clusters = interpolate_clusters(
        dense_points,
        segmented_points,
        cluster_labels,
        k=interpolation_k
    )
    del segmented, segmented_points, cluster_labels

    # Save result
    np.savez(
        args.dense_pc_path,
        points=dense_points,
        colors=dense_colors,
        embeddings=inter_embeddings,
        cluster_labels=inter_clusters
    )