"""
Example usage:
--------------
python cluster_pointcloud.py --input_npz clip_pointcloud_colored.npz --output_npz clustered_output.npz
"""

import argparse
import json
import numpy as np
import torch
import clip
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# --------------------------
# Helper functions
# --------------------------

def smooth_labels(points, labels, k=100):
    """
    Smooth labels based on k-NN majority voting.
    Args:
        points (np.ndarray): (N, 3) point cloud coordinates
        labels (np.ndarray): (N,) initial labels
        k (int): number of neighbors
    Returns:
        np.ndarray: (N,) smoothed labels
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    smoothed_labels = []
    for idx_list in indices:
        neighbor_labels = labels[idx_list[1:]]  # exclude self
        mode_label = stats.mode(neighbor_labels, axis=None, keepdims=False)[0]
        smoothed_labels.append(mode_label)
    return np.array(smoothed_labels, dtype=labels.dtype)

def load_object_tags(json_file='../assets/ram_object_tags.json', device='cuda'):
    """
    Load object tags and encode them with CLIP.
    Returns:
        object_tags (list of str)
        tag_embeddings (np.ndarray of shape (num_objects, 512))
    """
    with open(json_file, 'r') as f:
        object_tags = json.load(f)
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    tag_embeddings = np.zeros((len(object_tags), 512), dtype=np.float32)

    for i, tag in enumerate(tqdm(object_tags, desc="Encoding object tags")):
        text = clip.tokenize([tag]).to(device)
        with torch.no_grad():
            feat = model.encode_text(text).detach().cpu().numpy()
        feat /= np.linalg.norm(feat, axis=1, keepdims=True)
        tag_embeddings[i] = feat

    return object_tags, tag_embeddings

def cluster_point_cloud(input_npz_file, output_npz_file, k_smooth=10, confidence_threshold=0.05):
    """
    Cluster point cloud using CLIP embeddings and optional k-NN smoothing.
    """
    print(f"Loading point cloud from {input_npz_file}...")
    data = np.load(input_npz_file)
    points = data['points']
    colors = data['colors']
    embeddings = data['embeddings']
    del data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    object_tags, tag_embeddings = load_object_tags(device=device)

    print("Matching points to tags...")
    similarity = embeddings @ tag_embeddings.T  # (N_points, N_tags)

    # Match each point to closest tag embedding
    similarity = embeddings @ tag_embeddings.T  # shape: (N, num_tags)
    best_match_indices = np.argmax(similarity, axis=1)
    point_tags = [object_tags[i] for i in best_match_indices]
    cluster_labels = best_match_indices.astype(np.uint16)

    print(f"Smoothing labels with k-NN (k={k_smooth})...")
    smoothed_labels = smooth_labels(points, cluster_labels, k=k_smooth)

    point_tags = [object_tags[label] if label != -1 else 'unknown' for label in smoothed_labels]

    np.savez(output_npz_file,
             points=points,
             embeddings=embeddings,
             colors=colors,
             cluster_labels=smoothed_labels,
             object_tags=np.array(point_tags))
    print(f"Saved clustered point cloud to {output_npz_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster sparse point cloud and propagate labels to dense cloud.")
    parser.add_argument('--input_npz', type=str, required=True, help='Input point cloud .npz')
    parser.add_argument('--output_npz', type=str, required=True, help='Output point cloud .npz')
    parser.add_argument('--k_smooth', type=int, default=100, help='k-NN smoothing (default=10)')
    parser.add_argument('--confidence_threshold', type=float, default=0.001, help='Confidence threshold for tag matching (default=0.05)')
    args = parser.parse_args()

    cluster_point_cloud(args.input_npz, args.output_npz, args.k_smooth, args.confidence_threshold)
