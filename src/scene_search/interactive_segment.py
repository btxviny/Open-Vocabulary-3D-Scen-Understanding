"""
Interactive point-cloud segmentation via user-picked seed points.

Loads a processed dense_pointcloud.npz, opens an Open3D window where the user
Shift-clicks seed points, runs KMeans initialised at those seeds (on fused
positional + CLIP features), propagates cluster labels to the full cloud, then
offers save / retry / quit.

The output .npz extends the input with a `cluster_labels` array — use
tools/visualize.py --mode cluster to inspect it.

Usage:
    uv run python -m src.scene_search.interactive_segment \\
        --input  path/to/dense_pointcloud.npz \\
        --output path/to/segmented.npz
"""

import argparse

import numpy as np
import open3d as o3d
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ── Feature helpers ───────────────────────────────────────────────────────────

def _positional_encoding(coords: np.ndarray, num_freqs: int = 10) -> np.ndarray:
    freq_bands = np.logspace(0.0, np.log10(10_000.0), num=num_freqs)
    pe = []
    for axis in range(3):
        for freq in freq_bands:
            pe.append(np.sin(coords[:, axis] / freq))
            pe.append(np.cos(coords[:, axis] / freq))
    return np.stack(pe, axis=1)


def _fuse_features(points: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    pos = _positional_encoding(points)
    pos /= np.linalg.norm(pos, axis=1, keepdims=True) + 1e-8
    emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    fused = np.concatenate([pos, emb], axis=1)
    fused /= np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8
    return PCA(n_components=10).fit_transform(fused)


def _downsample(points, colors, embeddings, every_k: int = 10):
    idxs = np.arange(0, len(points), every_k)
    return points[idxs], colors[idxs], embeddings[idxs], idxs


# ── Label propagation ─────────────────────────────────────────────────────────

def _propagate_labels(
    full_points: np.ndarray,
    seed_points: np.ndarray,
    seed_labels: np.ndarray,
    k: int = 5,
    batch_size: int = 5000,
    iterations: int = 3,
) -> np.ndarray:
    N = len(full_points)
    labels = np.zeros(N, dtype=int)

    nn = NearestNeighbors(n_neighbors=k).fit(seed_points)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        nbr_idxs = nn.kneighbors(full_points[start:end], return_distance=False)
        labels[start:end] = stats.mode(seed_labels[nbr_idxs], axis=1, keepdims=False).mode

    for it in range(1, iterations):
        print(f"  Label smoothing pass {it + 1}")
        nn2 = NearestNeighbors(n_neighbors=k).fit(full_points)
        new_labels = np.zeros(N, dtype=int)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            nbr_idxs = nn2.kneighbors(full_points[start:end], return_distance=False)
            new_labels[start:end] = stats.mode(labels[nbr_idxs], axis=1, keepdims=False).mode
        labels = new_labels

    return labels


# ── Open3D windows ────────────────────────────────────────────────────────────

def _pick_seeds(pcd: o3d.geometry.PointCloud) -> list[int]:
    print("\nShift + Left Click to place seed points.  Press Q when done.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("Pick seed points", width=1280, height=720)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()
    return picked


def _show_result(points: np.ndarray, colors: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window("Segmentation result — press Q to close")
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.array([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_file: str, output_file: str, every_k: int = 10) -> None:
    data = np.load(input_file)
    points, colors, embeddings = data["points"], data["colors"], data["embeddings"]
    print(f"Loaded {len(points):,} points from {input_file}")

    ds_pts, ds_cols, ds_embs, _ = _downsample(points, colors, embeddings, every_k)
    print(f"Working set: {len(ds_pts):,} points (every_k={every_k})")

    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(points)
    full_pcd.colors = o3d.utility.Vector3dVector(colors)

    fused_ds = _fuse_features(ds_pts, ds_embs)

    while True:
        picked_idxs = _pick_seeds(full_pcd)
        if len(picked_idxs) < 2:
            print("Need at least 2 seed points. Try again.")
            continue

        print(f"\n{len(picked_idxs)} seeds picked — clustering…")
        nn = NearestNeighbors(n_neighbors=1).fit(ds_pts)
        ds_seed_idxs = nn.kneighbors(points[picked_idxs], return_distance=False).flatten()
        init_centers = fused_ds[ds_seed_idxs]

        ds_labels = KMeans(
            n_clusters=len(picked_idxs), init=init_centers, n_init=1, max_iter=100
        ).fit_predict(fused_ds)

        print("Propagating labels to full cloud…")
        full_labels = _propagate_labels(points, ds_pts, ds_labels)

        cluster_colors = np.random.default_rng(42).random((full_labels.max() + 1, 3))
        blended = 0.7 * cluster_colors[full_labels] + 0.3 * colors
        _show_result(points, blended)

        choice = ""
        while choice not in ("y", "r", "n"):
            choice = input("\n[Y] Save and exit  [R] Retry  [N] Exit without saving: ").lower()

        if choice == "y":
            np.savez(output_file,
                     points=points,
                     colors=colors,
                     embeddings=embeddings,
                     cluster_labels=full_labels)
            print(f"Saved → {output_file}")
            break
        elif choice == "n":
            print("Exiting without saving.")
            break
        # else: retry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive seed-based point cloud segmentation")
    parser.add_argument("--input",   required=True, help=".npz with points / colors / embeddings")
    parser.add_argument("--output",  required=True, help="Output .npz (adds cluster_labels)")
    parser.add_argument("--every_k", type=int, default=10,
                        help="Downsample rate for clustering (default 10)")
    args = parser.parse_args()
    main(args.input, args.output, args.every_k)
