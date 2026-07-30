import open3d as o3d
import numpy as np
import os
import argparse
from tqdm import tqdm


# Camera intrinsics
fx, fy = 386.99176, 386.99176
cx, cy = 320.96591, 241.47070
width, height = 640, 480
depth_scale = 1000.0
depth_trunc = 3.5

# Create Open3D intrinsics
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)


def parse_timestamp(dirname):
    try:
        dirname = dirname.strip("()")
        main_ts, sub_ts = map(str.strip, dirname.split(','))
        return int(main_ts), int(sub_ts)
    except Exception:
        return (0, 0)


def load_rgbd(color_path, depth_path):
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)
    arr = np.asarray(depth)
    if arr.dtype != np.uint16:
        print(f"[WARN] Depth image not 16-bit: {depth_path}")
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )


def estimate_camera_poses(frame_dirs):
    print("[INFO] Estimating camera poses...")
    poses = [np.eye(4)]
    prev_rgbd = load_rgbd(
        os.path.join(frame_dirs[0], "rgb_image.png"),
        os.path.join(frame_dirs[0], "depth_image.png")
    )

    for i in tqdm(range(1, len(frame_dirs)), desc="Odometry"):
        curr_rgbd = load_rgbd(
            os.path.join(frame_dirs[i], "rgb_image.png"),
            os.path.join(frame_dirs[i], "depth_image.png")
        )

        odo_init = np.eye(4)
        option = o3d.pipelines.odometry.OdometryOption()
        success, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            prev_rgbd, curr_rgbd,
            intrinsic,
            odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option
        )

        if not success:
            print(f"[WARN] Hybrid odometry failed at frame {i}, trying depth-only...")
            success, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd, curr_rgbd,
                intrinsic,
                odo_init,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromDepth(),
                option
            )

        if success:
            poses.append(poses[-1] @ trans)
        else:
            print(f"[ERROR] Odometry failed at frame {i}, reusing last pose.")
            poses.append(poses[-1])

        prev_rgbd = curr_rgbd

    return poses


def integrate_frames(frame_dirs, poses, output_path):
    print("[INFO] Integrating frames into TSDF volume...")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,  # Higher resolution
        sdf_trunc=0.03,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i, (frame_dir, pose) in tqdm(enumerate(zip(frame_dirs, poses)), desc="Integrating", total = len(frame_dirs)):
        color_path = os.path.join(frame_dir, "rgb_image.png")
        depth_path = os.path.join(frame_dir, "depth_image.png")

        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            print(f"[WARN] Missing files at {frame_dir}, skipping.")
            continue

        rgbd = load_rgbd(color_path, depth_path)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[INFO] Saved fused point cloud to: {output_path}")


def visualize_poses(poses):
    frames = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(p)
        for p in poses
    ]
    o3d.visualization.draw_geometries(frames)


def main(input_path, output_path, subsample=1):
    frame_dirs = [
        os.path.join(input_path, d)
        for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ]
    frame_dirs = sorted(frame_dirs, key=lambda x: parse_timestamp(os.path.basename(x)))

    valid_frame_dirs = []
    for d in frame_dirs:
        rgb_path = os.path.join(d, "rgb_image.png")
        depth_path = os.path.join(d, "depth_image.png")
        if os.path.isfile(rgb_path) and os.path.isfile(depth_path):
            valid_frame_dirs.append(d)

    if len(valid_frame_dirs) < 2:
        print("[ERROR] Not enough valid frames.")
        return

    if subsample > 1:
        valid_frame_dirs = valid_frame_dirs[::subsample]

    poses = estimate_camera_poses(valid_frame_dirs)

    # Optional pose visualization
    visualize_poses(poses)

    integrate_frames(valid_frame_dirs, poses, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to frames folder")
    parser.add_argument("--output_path", type=str, required=True, help="Output .ply point cloud")
    parser.add_argument("--subsample", type=int, default=1, help="Use every nth frame")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.subsample)
