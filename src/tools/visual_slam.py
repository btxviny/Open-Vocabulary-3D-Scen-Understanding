import os
import numpy as np
import cv2
import open3d as o3d
import argparse
from tqdm import tqdm

# Camera intrinsics
fx, fy = 386.99176025390625, 386.99176025390625
cx, cy = 320.9659118652344, 241.470703125
width, height = 640, 480
depth_scale = 1000.0  # mm to meters
depth_trunc = 3.5     # meters

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)


def parse_timestamp(dirname):
    try:
        dirname = dirname.strip("()")
        main_ts, sub_ts = map(str.strip, dirname.split(','))
        return int(main_ts), int(sub_ts)
    except Exception:
        return (0, 0)


def load_rgbd(rgb_path, depth_path):
    color = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
    return color, depth


def depth_to_3d(u, v, depth, K):
    z = depth[v, u] / depth_scale
    if z == 0 or z > depth_trunc:
        return None
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.array([x, y, z])


def get_matched_3d_points(img1, img2, depth1, depth2, K, idx=None, debug_dir=None):
    sift = cv2.SIFT_create(1000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    matches = sorted(good_matches, key=lambda x: x.distance)

    pts3d_1, pts3d_2 = [], []

    for m in matches:
        u1, v1 = map(int, kp1[m.queryIdx].pt)
        u2, v2 = map(int, kp2[m.trainIdx].pt)

        p1 = depth_to_3d(u1, v1, depth1, K)
        p2 = depth_to_3d(u2, v2, depth2, K)

        if p1 is not None and p2 is not None:
            pts3d_1.append(p1)
            pts3d_2.append(p2)

    if len(pts3d_1) < 6:
        if debug_dir is not None and idx is not None:
            os.makedirs(debug_dir, exist_ok=True)
            img1_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img2_kp = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            concat = np.hstack((img1_kp, img2_kp))
            debug_path = os.path.join(debug_dir, f"bad_match_{idx:04d}.png")
            cv2.imwrite(debug_path, concat)

        return None, None

    return np.array(pts3d_1), np.array(pts3d_2)




def estimate_pose(pts3d_src, pts3d_dst):
    retval, M, _ = cv2.estimateAffine3D(pts3d_src, pts3d_dst)
    if retval == 0:
        return None
    T = np.eye(4)
    T[:3, :4] = M  # 3x4 matrix (R | t)
    return T


def estimate_camera_poses(frame_dirs):
    poses = [np.eye(4)]
    debug_dir = '/home/viny/Desktop/debug'

    rgb_prev, depth_prev = load_rgbd(
        os.path.join(frame_dirs[0], "rgb_image.png"),
        os.path.join(frame_dirs[0], "depth_image.png"),
    )

    for i in tqdm(range(1, len(frame_dirs)), desc="Estimating Camera poses: ", total=len(frame_dirs) - 1):
        rgb_curr, depth_curr = load_rgbd(
            os.path.join(frame_dirs[i], "rgb_image.png"),
            os.path.join(frame_dirs[i], "depth_image.png"),
        )

        pts1, pts2 = get_matched_3d_points(
            rgb_prev, rgb_curr, depth_prev, depth_curr, K,
            idx=i, debug_dir=debug_dir
        )

        if pts1 is None:
            print(f"[WARN] Frame {i}: Not enough matches, copying last pose.")
            poses.append(poses[-1])
            rgb_prev, depth_prev = rgb_curr, depth_curr
            continue

        T = estimate_pose(pts1, pts2)
        if T is None:
            print(f"[WARN] Frame {i}: Pose estimation failed, copying last pose.")
            poses.append(poses[-1])
            rgb_prev, depth_prev = rgb_curr, depth_curr
            continue

        poses.append(poses[-1] @ T)
        rgb_prev, depth_prev = rgb_curr, depth_curr

    return poses



def integrate_frames(frame_dirs, poses, output_path):
    print("[INFO] Starting manual point cloud integration...")

    all_points = []
    all_colors = []
    stride = 20  # pixel stride

    for i, (frame_dir, pose) in tqdm(enumerate(zip(frame_dirs, poses)), desc="Backprojecting points to 3D: ", total=len(frame_dirs)):
        rgb_path = os.path.join(frame_dir, "rgb_image.png")
        depth_path = os.path.join(frame_dir, "depth_image.png")

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)

        h, w = depth.shape

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create meshgrid with stride
        u = np.arange(0, w, stride)
        v = np.arange(0, h, stride)
        uu, vv = np.meshgrid(u, v)

        z = depth[vv, uu] / depth_scale  # sample only at stride pixels
        valid = (z > 0) & (z < depth_trunc)

        uu = uu[valid]
        vv = vv[valid]
        z = z[valid]

        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy

        ones = np.ones_like(x)
        points_camera = np.stack([x, y, z, ones], axis=1)  # Nx4

        points_world = (pose @ points_camera.T).T[:, :3]

        colors = rgb[vv, uu, :] / 255.0

        all_points.append(points_world)
        all_colors.append(colors)

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[INFO] Saved point cloud to: {output_path}")


def main(input_path, output_path):
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
    poses = estimate_camera_poses(valid_frame_dirs)
    integrate_frames(valid_frame_dirs, poses, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()
    main(args.input_path, args.output_path)
