import numpy as np
import argparse
import time
import rerun as rr
from sklearn.decomposition import PCA

def visualize_point_cloud(npz_file, mode='normal', alpha=0.3):
    print(f"Loading data from: {npz_file}")
    data = np.load(npz_file)
    print("Available keys in the file:", list(data.keys()))
    
    # Load points and colors
    if 'points' in data:
        points = data['points']
        colors = data['colors'].astype(float)  # assume already normalized or raw, handle below
        cluster_labels = data.get('cluster_label' if 'cluster_label' in data else 'cluster_labels')
    else:
        points = data.get('coord', None)
        if points is None:
            raise ValueError("Could not find points/coordinates in the file")
        colors = data.get('color', np.ones_like(points)).astype(float) / 255.0
        cluster_labels = data.get('group', None)

    print(f"Point cloud shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")

    if mode == 'pca':
        if 'embeddings' not in data:
            raise ValueError("PCA mode selected but 'embeddings' key not found in the file.")
        embeddings = data['embeddings']
        print(f"Embeddings shape: {embeddings.shape}")
        pca_model = PCA(n_components=3)
        embeddings_3d = pca_model.fit_transform(embeddings)
        min_vals = embeddings_3d.min(axis=0)
        max_vals = embeddings_3d.max(axis=0)
        colors = (embeddings_3d - min_vals) / (max_vals - min_vals + 1e-10)
        print("Using PCA embeddings for color visualization.")

    elif mode == 'cluster':
        if cluster_labels is None:
            raise ValueError("Cluster mode selected but no cluster labels found in the file.")
        cluster_labels = cluster_labels.flatten()
        unique_labels = np.unique(cluster_labels)
        print(f"Number of segments: {len(unique_labels)}")

        np.random.seed(42)
        label_colors = np.random.random((len(unique_labels), 3))
        if -1 in unique_labels:
            label_colors[np.where(unique_labels == -1)[0]] = [0.5, 0.5, 0.5]

        segment_colors = label_colors[np.searchsorted(unique_labels, cluster_labels)]

        if alpha < 1.0:
            colors = (1 - alpha) * colors + alpha * segment_colors
        else:
            colors = segment_colors

        unique, counts = np.unique(cluster_labels, return_counts=True)
        print("\nSegment sizes (top 5 largest):")
        sorted_indices = np.argsort(counts)[::-1]
        for i in range(min(5, len(unique))):
            print(f"Segment {unique[sorted_indices[i]]}: {counts[sorted_indices[i]]} points")

    # else mode == 'normal', no changes needed to colors

    rr.init("pointcloud_visualization", spawn=True)

    rr.log(
        "point_cloud",
        rr.Points3D(
            points,
            colors=colors,
        )
    )
    print(f'Colors max: {colors.max()}, min: {colors.min()}')
    print("\nVisualization controls:")
    print("- Left click + drag: Rotate")
    print("- Right click + drag: Pan")
    print("- Mouse wheel: Zoom")
    print("\nPress Ctrl+C to stop visualization...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Visualization stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D point cloud with different modes.")
    parser.add_argument('--npz_file', type=str, required=True, help='Path to .npz file containing point cloud data.')
    parser.add_argument('--mode', choices=['normal', 'pca', 'cluster'], default='normal', help='Visualization mode.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha blending for cluster mode (ignored otherwise).')
    args = parser.parse_args()

    visualize_point_cloud(args.npz_file, mode=args.mode, alpha=args.alpha)
