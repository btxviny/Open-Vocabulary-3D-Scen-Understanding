import open3d as o3d
import numpy as np
import argparse


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))


def refine_registration(source, target, initial_transformation, voxel_size):
    return o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=voxel_size * 0.4,
        init=initial_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())


def align_pointclouds(source_path, target_path, voxel_size=0.02):
    # Load point clouds
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    # Preprocessing
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Global registration
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("[INFO] RANSAC result:\n", result_ransac.transformation)

    # Local refinement
    result_icp = refine_registration(source, target, result_ransac.transformation, voxel_size)
    print("[INFO] ICP refined result:\n", result_icp.transformation)

    # Apply transform and visualize
    source.transform(result_icp.transformation)
    source.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 1, 0])  # Green

    o3d.visualization.draw_geometries([source, target])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align two point clouds using Open3D.")
    parser.add_argument("--source", type=str, help="Path to the source .ply point cloud")
    parser.add_argument("--target", type=str, help="Path to the target .ply point cloud")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size for downsampling")

    args = parser.parse_args()
    align_pointclouds(args.source, args.target, args.voxel_size)
