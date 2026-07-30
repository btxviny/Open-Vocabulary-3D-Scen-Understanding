import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy import stats
import open3d as o3d

def positional_encoding_3d(coords, num_freqs=10):
    freq_bands = np.logspace(0.0, np.log10(10000.0), num=num_freqs)
    pe = []
    for i in range(3):  # x, y, z
        for freq in freq_bands:
            pe.append(np.sin(coords[:, i] / freq))
            pe.append(np.cos(coords[:, i] / freq))
    pe = np.stack(pe, axis=1)
    return pe

def fuse_features(points, embeddings, w_pos=0.5, w_emb=0.5):
    pos_enc = positional_encoding_3d(points)
    pos_enc /= np.linalg.norm(pos_enc, axis=1, keepdims=True) + 1e-8
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    fused = np.concatenate([1 * pos_enc, 1 * embeddings], axis=1)
    fused  = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)
    reduced = PCA(n_components=10).fit_transform(fused)
    return reduced

def uniform_downsample(points, colors, embeddings, every_k=10):
    idxs = np.arange(0, len(points), every_k)
    return points[idxs], colors[idxs], embeddings[idxs], idxs

def propagate_labels(full_points, down_points, down_labels, k=100, batch_size=5000, num_iterations=3):
    N = full_points.shape[0]
    labels = np.zeros(N, dtype=int)

    # Initial propagation from downsampled points
    nn = NearestNeighbors(n_neighbors=k).fit(down_points)
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_points = full_points[batch_start:batch_end]
        neighbor_idxs = nn.kneighbors(batch_points, return_distance=False)
        neighbor_clusters = down_labels[neighbor_idxs]
        modes = stats.mode(neighbor_clusters, axis=1, keepdims=False)
        labels[batch_start:batch_end] = modes.mode

    # Iterative smoothing
    for iter in range(1, num_iterations):
        print(f"🔁 Label propagation iteration {iter+1}")
        nn = NearestNeighbors(n_neighbors=k).fit(full_points)
        new_labels = np.zeros(N, dtype=int)

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_points = full_points[batch_start:batch_end]
            neighbor_idxs = nn.kneighbors(batch_points, return_distance=False)
            neighbor_clusters = labels[neighbor_idxs]
            modes = stats.mode(neighbor_clusters, axis=1, keepdims=False)
            new_labels[batch_start:batch_end] = modes.mode

        labels = new_labels

    return labels


def get_picked_points(vis):
    """Get picked points from Open3D visualization."""
    print("")
    print("1) Please pick at least two points using [Shift + Left Click]")
    print("   Press [Y] when done picking points")
    print("2) After pressing [Y], press [Q] to close the window")
    picked_points = vis.get_picked_points()
    return picked_points

def main(input_file, output_file, every_k=10, alpha=0.5, beta=0.5):
    # Load data
    data = np.load(input_file)
    points, colors, embeddings = data["points"], data["colors"], data["embeddings"]

    # Downsample
    ds_points, ds_colors, ds_embeddings, ds_idxs = uniform_downsample(points, colors, embeddings, every_k)
    print(f"Downsampled colors range: min={ds_colors.min():.3f}, max={ds_colors.max():.3f}")
    print(f"🔻 Downsampled from {len(points)} to {len(ds_points)}")

    # Create Open3D point cloud for visualization of full resolution cloud
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(points)
    full_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Set up visualization with point picking
    while True:
        print("\nPoint Picking Instructions:")
        print("1) Hold down [SHIFT] and click points to select cluster centers")
        print("2) Press [Q] to close the window when done")
        
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window("Interactive Point Cloud Segmentation", width=1024, height=768)
        vis.add_geometry(full_pcd)  # Show full resolution cloud
        
        # Improve visualization settings
        opt = vis.get_render_option()
        opt.point_size = 2.0  # Slightly smaller point size for full res
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        
        # Set up camera view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        # Run visualization and get picked points
        vis.run()
        picked_idxs = vis.get_picked_points()
        vis.destroy_window()

        if len(picked_idxs) < 2:
            print("Error: Please pick at least 2 points for clustering")
            return

        print(f"Selected {len(picked_idxs)} points as cluster centers")
        
        # Map picked points from full resolution to downsampled cloud
        picked_points = points[picked_idxs]
        nn = NearestNeighbors(n_neighbors=1).fit(ds_points)
        ds_picked_idxs = nn.kneighbors(picked_points, return_distance=False).flatten()
        
        # Feature fusion
        fused_ds = fuse_features(ds_points, ds_embeddings, w_pos=alpha, w_emb=beta)
        init_centers = fused_ds[ds_picked_idxs]  # Use mapped indices

        # Clustering
        kmeans = KMeans(n_clusters=len(picked_idxs), init=init_centers, n_init=1, max_iter=100)
        ds_labels = kmeans.fit_predict(fused_ds)
        print(f"🔎 Clustered downsampled cloud into {np.unique(ds_labels).size} clusters")

        # Propagate
        full_labels = propagate_labels(points, ds_points, ds_labels, k=5)
        print("✅ Labels propagated to full cloud")

        # Visualize results
        cluster_colors = np.random.rand(np.max(full_labels)+1, 3)
        point_cluster_colors = cluster_colors[full_labels]
        
        # Blend cluster colors with original colors using alpha=0.3
        alpha = 0.3  # Use lower alpha to make cluster colors more transparent
        blended_colors = (1 - alpha) * colors + alpha * point_cluster_colors
        print(f"Blended colors range: min={blended_colors.min():.3f}, max={blended_colors.max():.3f}")

        colored_cloud = o3d.geometry.PointCloud()
        colored_cloud.points = o3d.utility.Vector3dVector(points)
        colored_cloud.colors = o3d.utility.Vector3dVector(blended_colors)

        # Show final result
        vis = o3d.visualization.Visualizer()
        vis.create_window("Segmentation Result")
        vis.add_geometry(colored_cloud)
        
        # Improve visualization settings
        opt = vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        
        vis.run()
        vis.destroy_window()

        # Ask user for feedback
        while True:
            choice = input("\nAre you satisfied with the result?\n[Y] Save and exit\n[R] Try again\n[N] Exit without saving\nYour choice: ").upper()
            if choice in ['Y', 'R', 'N']:
                break
            print("Invalid choice. Please enter Y, R, or N.")

        if choice == 'Y':
            np.savez(
                output_file, 
                points=points, 
                colors= colors,  # Save original colors
                embeddings=embeddings, 
                cluster_labels=full_labels
            )
            print(f"📦 Saved clustered result to {output_file}")
            break
        elif choice == 'N':
            print("Exiting without saving.")
            break
        # If choice is 'R', the while loop continues for another iteration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=".npz file with points/colors/embeddings")
    parser.add_argument("--output", required=True, help="Output path for result .npz")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for positional encoding")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for clip embedding")
    args = parser.parse_args()

    main(args.input, args.output, alpha=args.alpha, beta=args.beta)
