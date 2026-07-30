"""
Point cloud visualiser — serves a Rerun web viewer.

Open the printed URL in your browser to view the cloud.  The native Rerun
desktop app does not work under WSL (R32Float driver limitation); the web
viewer has no such restriction.

Usage:
    python -m tools.visualize --npz_file path/to/dense_pointcloud.npz
    python -m tools.visualize --npz_file path/to/dense_pointcloud.npz --mode pca
    python -m tools.visualize --npz_file path/to/segmented.npz       --mode cluster
"""

import argparse
import time

import numpy as np
import rerun as rr
from sklearn.decomposition import PCA


def visualize_point_cloud(npz_file: str, mode: str = "normal", alpha: float = 0.5) -> None:
    print(f"Loading {npz_file}")
    data = np.load(npz_file)
    print("Keys:", list(data.keys()))

    points = data["points"]
    colors = data["colors"].astype(np.float32)
    if colors.max() > 1.0:
        colors /= 255.0
    cluster_labels = data.get("cluster_labels", data.get("cluster_label"))

    print(f"Points: {points.shape}  Colors: {colors.shape}")

    if mode == "pca":
        if "embeddings" not in data:
            raise ValueError("--mode pca requires an 'embeddings' key in the .npz file")
        emb = data["embeddings"]
        print(f"Embeddings: {emb.shape} — running PCA…")
        e3 = PCA(n_components=3).fit_transform(emb)
        colors = ((e3 - e3.min(0)) / (e3.max(0) - e3.min(0) + 1e-10)).astype(np.float32)

    elif mode == "cluster":
        if cluster_labels is None:
            raise ValueError("--mode cluster requires a 'cluster_labels' key in the .npz file")
        cluster_labels = cluster_labels.flatten()
        unique = np.unique(cluster_labels)
        print(f"Clusters: {len(unique)}")

        rng = np.random.default_rng(42)
        cmap = rng.random((len(unique), 3)).astype(np.float32)
        if -1 in unique:
            cmap[np.searchsorted(unique, -1)] = [0.5, 0.5, 0.5]
        seg_colors = cmap[np.searchsorted(unique, cluster_labels)]
        colors = ((1 - alpha) * colors + alpha * seg_colors).astype(np.float32)

        counts = np.bincount(cluster_labels[cluster_labels >= 0])
        top5 = np.argsort(counts)[::-1][:5]
        print("Largest clusters:")
        for i in top5:
            print(f"  Cluster {i}: {counts[i]:,} points")

    rr.init("pointcloud_visualization", spawn=True)
    rr.log("point_cloud", rr.Points3D(positions=points, colors=colors))

    print("\nVisualization controls:")
    print("  Left click + drag: Rotate  |  Right click + drag: Pan  |  Scroll: Zoom")
    print("\nPress Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D point cloud with Rerun web viewer")
    parser.add_argument("--npz_file", required=True)
    parser.add_argument("--mode",  choices=["normal", "pca", "cluster"], default="normal")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Cluster colour blend (cluster mode only)")
    args = parser.parse_args()
    visualize_point_cloud(args.npz_file, mode=args.mode, alpha=args.alpha)
