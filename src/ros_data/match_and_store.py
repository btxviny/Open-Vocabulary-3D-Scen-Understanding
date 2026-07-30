import os
import time
import numpy as np
import cv2
import open3d as o3d

def match_and_store_process(conn_from_lidar, conn_from_image):
    print("[MATCH & STORE] Listening for LiDAR and image data...")

    lidar_dict = {}
    image_dict = {}
    latest_matched_ts = None
    first_timestamp = None

    TIME_WINDOW_NS = 10_000_000  # 10ms

    def to_ns(ts):
        return ts[0] * 1_000_000_000 + ts[1]

    def save_pcd(points, file_path):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
        o3d.io.write_point_cloud(file_path, cloud)

    while True:
        # Receive and store LiDAR messages
        if conn_from_lidar.poll():
            lidar_msg = conn_from_lidar.recv()
            ts = (int(lidar_msg['timestamp']['secs']), int(lidar_msg['timestamp']['nsecs']))
            lidar_dict[ts] = lidar_msg

            if first_timestamp is None:
                first_timestamp = ts
                folder_name = f"data_{ts[0]}_{ts[1]}"
                folder_path = os.path.join("./data_logs", folder_name)
                os.makedirs(folder_path, exist_ok=True)
                print(f"[MATCH & STORE] Created directory: {folder_path}")

        # Receive and store image messages
        if conn_from_image.poll():
            image_msg = conn_from_image.recv()
            ts = (int(image_msg['timestamp']['secs']), int(image_msg['timestamp']['nsecs']))
            image_dict[ts] = image_msg

        # Try to match timestamps
        for ts_img in list(image_dict.keys()):
            ts_img_ns = to_ns(ts_img)
            closest_ts_lidar = None
            min_diff = TIME_WINDOW_NS + 1

            for ts_lidar in lidar_dict:
                diff = abs(to_ns(ts_lidar) - ts_img_ns)
                if diff < min_diff:
                    min_diff = diff
                    closest_ts_lidar = ts_lidar

            if closest_ts_lidar:
                img_data = image_dict.pop(ts_img)
                lidar_data = lidar_dict.pop(closest_ts_lidar)
                latest_matched_ts = ts_img

                # Clean old unmatched data
                lidar_dict = {ts: val for ts, val in lidar_dict.items() if ts > latest_matched_ts}
                image_dict = {ts: val for ts, val in image_dict.items() if ts > latest_matched_ts}

                # Create subdirectory for matched timestamp
                matched_timestamp_dir = os.path.join(folder_path, f"{ts_img}")
                os.makedirs(matched_timestamp_dir, exist_ok=True)

                # Save RGB image
                rgb_path = os.path.join(matched_timestamp_dir, "rgb_image.png")
                cv2.imwrite(rgb_path, img_data["color_image"])

                # Save depth image
                if img_data["depth_image"] is not None:
                    depth_path = os.path.join(matched_timestamp_dir, "depth_image.png")
                    cv2.imwrite(depth_path, img_data["depth_image"])

                # Save LiDAR point cloud
                pcd_path = os.path.join(matched_timestamp_dir, "lidar.pcd")
                save_pcd(lidar_data["pointcloud"], pcd_path)

                # Save extrinsic matrix
                extrinsic = np.array(lidar_data["extrinsic_matrix"], dtype=np.float32)
                extrinsic_path = os.path.join(matched_timestamp_dir, "extrinsic_matrix.npy")
                np.save(extrinsic_path, extrinsic)

                print(f"[MATCH & STORE] Saved RGB, depth, LiDAR, extrinsic for timestamp {ts_img}")

        time.sleep(0.01)
