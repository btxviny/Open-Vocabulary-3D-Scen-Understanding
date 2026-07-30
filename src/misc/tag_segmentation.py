"""
Semantic tag-based point cloud segmentation with spatial consistency and instance awareness.
"""
import clip
import json
import numpy as np
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import torch.nn.functional as F


def load_clip_model(model_name="ViT-B/32"):
    """Load CLIP model and return model and device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(model_name, device=device)
    return model, device


def compute_tag_embeddings(model, object_tags, device):
    """Compute CLIP embeddings for all object tags."""
    num_objects = len(object_tags)
    tag_embeddings = np.zeros((num_objects, 512), dtype=np.float32)

    for i, tag in enumerate(object_tags):
        text = clip.tokenize([tag]).to(device)
        with torch.no_grad():
            feat = model.encode_text(text).detach().cpu().numpy()
        feat /= np.linalg.norm(feat, axis=1, keepdims=True)
        tag_embeddings[i] = feat

    return tag_embeddings


def smooth_labels(points, labels, k=100):
    """Smooth labels using KNN and mode."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1).fit(points)
    distances, indices = nbrs.kneighbors(points)

    smoothed_labels = np.zeros_like(labels)
    for i in tqdm(range(len(points)), desc="Smoothing labels using KNN..."):
        neighborhood_labels = labels[indices[i]]
        mode_result = stats.mode(neighborhood_labels, keepdims=True)
        smoothed_labels[i] = mode_result.mode[0]

    return smoothed_labels


def mark_spatial_outliers(points, labels, z_thresh=1.0):
    """
    Assign label -1 to spatial outliers in each semantic cluster using z-score.
    """
    print("\nDetecting spatial outliers...")
    updated_labels = labels.copy()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
        mask = labels == label
        cluster_points = points[mask]

        if len(cluster_points) < 10:
            continue  # skip small clusters

        z_scores = np.abs(stats.zscore(cluster_points, axis=0))
        is_outlier = (z_scores > z_thresh).any(axis=1)

        outlier_indices = np.where(mask)[0][is_outlier]
        updated_labels[outlier_indices] = -1

    return updated_labels


def remove_small_clusters(labels, min_points=200):
    """
    Remove clusters with fewer points than min_points by setting their labels to -1 (outlier).
    """
    print(f"\nRemoving clusters with less than {min_points} points...")
    updated_labels = labels.copy()
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:  # skip outliers
            continue
        mask = labels == label
        if np.sum(mask) < min_points:
            updated_labels[mask] = -1
            
    return updated_labels


def main(input_npz, output_npz, tags_path='./unique_tags.json', k_neighbors=100):
    """Main function for point cloud segmentation."""
    # Load data
    print("Loading data...")
    data = np.load(input_npz)
    points = data['points']
    colors = data['colors']
    embeddings = data['embeddings']

    # Load tags and initialize CLIP
    with open(tags_path, 'r') as f:
        object_tags = json.load(f)

    model, device = load_clip_model()
    tag_embeddings = compute_tag_embeddings(model, object_tags, device)

    # Tag each point using CLIP similarity
    print("Tagging points...")
    similarity = embeddings @ tag_embeddings.T
    labels = np.argmax(similarity, axis=1)

    # Optional spatial label smoothing
    labels = smooth_labels(points, labels, k=k_neighbors)

    # Outlier detection
    labels = mark_spatial_outliers(points, labels, z_thresh=3.0)
    
    # Remove small clusters
    labels = remove_small_clusters(labels, min_points=200)

    # Count points per tag and identify active tags
    print("\n📊 Final point counts:")
    all_labels = np.append(np.arange(len(object_tags)), -1)
    bincounts = {str(i): 0 for i in all_labels}
    for l in labels:
        bincounts[str(l)] += 1
    
    # Create filtered tag list (excluding tags that became outliers)
    active_tags = []
    for l in all_labels:
        if l == -1:
            print(f"  - outlier: {bincounts[str(l)]} points")
            continue
        name = object_tags[l]
        count = bincounts[str(l)]
        print(f"  - {name}: {count} points")
        if count > 0:  # Only keep tags that have points assigned to them
            active_tags.append(name)
    
    # Save filtered tags back to original path
    print(f"\n💾 Saving {len(active_tags)} active tags back to {tags_path}")
    with open(tags_path, 'w') as f:
        json.dump(active_tags, f, indent=2)

    # Save results
    np.savez(
        output_npz,
        points=points,
        colors=colors,
        embeddings=embeddings,
        cluster_labels=labels,
    )

    print("\n✅ Segmentation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment point cloud using semantic tags and spatial information")
    parser.add_argument('--input_npz', type=str, required=True, help='Input point cloud .npz file')
    parser.add_argument('--output_npz', type=str, required=True, help='Output segmentation .npz file')
    parser.add_argument('--tags_path', type=str, default='./unique_tags.json', help='Path to unique tags JSON file')
    parser.add_argument('--k_neighbors', type=int, default=21, help='Number of neighbors for label smoothing')
    args = parser.parse_args()

    main(args.input_npz, args.output_npz, args.tags_path, args.k_neighbors)
