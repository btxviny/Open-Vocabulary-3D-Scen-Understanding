import open3d as o3d
import numpy as np
import argparse
import os

def load_point_cloud_from_npz(npz_file):
    data = np.load(npz_file)
    points = data["points"]
    colors = data.get("colors", None)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0 if colors.max() > 1 else colors)

    return pcd

def main(npz_file):
    # Load point cloud
    pcd = load_point_cloud_from_npz(npz_file)
    print(f"[INFO] Loaded point cloud with {len(pcd.points)} points.")

    # Downsample (uniform)
    pcd = pcd.uniform_down_sample(every_k_points=10)
    print(f"[INFO] After downsampling: {len(pcd.points)} points.")

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson reconstruction
    print("[INFO] Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    
    # Optionally crop mesh to remove low-density regions (e.g., floating ceiling)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.02)
    vertices_to_keep = densities > density_threshold
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

    # Save mesh
    out_path = os.path.splitext(npz_file)[0] + "_poisson.ply"
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"[INFO] Mesh saved to {out_path}")

    # Visualize
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct surface from point cloud")
    parser.add_argument("--npz_file", type=str, help="Path to .npz file with point cloud data")
    args = parser.parse_args()
    
    main(args.npz_file)
