import numpy as np
import open3d as o3d
import time
import random

def vizualization_process(conn_from_reconstruct):

    print("[VIZ] Waiting for first frame...")

    object_colormap = {}
    bbox_geometries = {}

    def get_color_for_object(obj_id):
        if obj_id not in object_colormap:
            object_colormap[obj_id] = [random.random(), random.random(), random.random()]
        return object_colormap[obj_id]

    while not conn_from_reconstruct.poll():
        time.sleep(0.01)

    msg = conn_from_reconstruct.recv()
    ts = msg["timestamp"]
    lidar_pts = np.array(msg["lidar_points"], dtype=np.float64)

    print(f"[VIZ] First frame {ts} with {len(msg['objects'])} segmented objects")

    # LiDAR point cloud
    lidar_pcd = o3d.geometry.PointCloud()
    if len(lidar_pcd.points) == 0:
        lidar_pcd.points = o3d.utility.Vector3dVector(lidar_pts)
        lidar_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.5, 0.5, 0.5], (len(lidar_pts), 1))
        )
    else:
        new_points = np.vstack((np.asarray(lidar_pcd.points), lidar_pts))
        new_colors = np.vstack((np.asarray(lidar_pcd.colors), np.tile([0.5, 0.5, 0.5], (len(lidar_pts), 1))))
        lidar_pcd.points = o3d.utility.Vector3dVector(new_points)
        lidar_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    red_cam_pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Fused LiDAR + Camera Segments")
    vis.add_geometry(lidar_pcd)
    vis.add_geometry(red_cam_pcd)

    # Set initial view
    view_ctl = vis.get_view_control()
    view_ctl.set_lookat([0, 0, 0])           # center of the scene
    view_ctl.set_front([0, 0, -1])           # camera direction
    view_ctl.set_up([0, -1, 0])              # upward direction

    active_colored_objects = {}

    try:
        while True:
            if conn_from_reconstruct.poll():
                msg = conn_from_reconstruct.recv()
                ts = msg["timestamp"]
                lidar_pts = np.array(msg["lidar_points"], dtype=np.float64)

                # Accumulate LiDAR points
                if len(lidar_pcd.points) == 0:
                    lidar_pcd.points = o3d.utility.Vector3dVector(lidar_pts)
                    lidar_pcd.colors = o3d.utility.Vector3dVector(
                        np.tile([0.5, 0.5, 0.5], (len(lidar_pts), 1))
                    )
                else:
                    new_points = np.vstack((np.asarray(lidar_pcd.points), lidar_pts))
                    new_colors = np.vstack((np.asarray(lidar_pcd.colors), np.tile([0.5, 0.5, 0.5], (len(lidar_pts), 1))))
                    lidar_pcd.points = o3d.utility.Vector3dVector(new_points)
                    lidar_pcd.colors = o3d.utility.Vector3dVector(new_colors)

                vis.update_geometry(lidar_pcd)

                red_points = []

                for obj in msg["objects"]:
                    obj_id = obj["object_id"]
                    obj_pts = np.array(obj["points"], dtype=np.float64)

                    if obj["bbox"]:
                        color = get_color_for_object(obj_id)

                        # Update or add the object's point cloud
                        if obj_id in active_colored_objects:
                            pcd = active_colored_objects[obj_id]
                            old_pts = np.asarray(pcd.points)
                            new_pts = np.vstack((old_pts, obj_pts))
                            pcd.points = o3d.utility.Vector3dVector(new_pts)

                            old_colors = np.asarray(pcd.colors)
                            new_colors = np.vstack((old_colors, np.tile(color, (len(obj_pts), 1))))
                            pcd.colors = o3d.utility.Vector3dVector(new_colors)

                            vis.update_geometry(pcd)
                        else:
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(obj_pts)
                            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(obj_pts), 1)))
                            active_colored_objects[obj_id] = pcd
                            vis.add_geometry(pcd)

                        # -----------------------
                        # Bounding box logic commented out:
                        #
                        # if len(obj_pts) < 4:
                        #     continue
                        #
                        # obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                        #     o3d.utility.Vector3dVector(obj_pts)
                        # )
                        # volume = np.prod(obb.extent)
                        # if volume < 0.0016:
                        #     continue
                        #
                        # obb.color = [0.8, 0.0, 1.0]
                        #
                        # if obj_id in bbox_geometries:
                        #     existing_obb = bbox_geometries[obj_id]
                        #     existing_obb.center = obb.center
                        #     existing_obb.R = obb.R
                        #     existing_obb.extent = obb.extent
                        #     vis.update_geometry(existing_obb)
                        # else:
                        #     bbox_geometries[obj_id] = obb
                        #     vis.add_geometry(obb)
                        # -----------------------

                    else:
                        red_points.append(obj_pts)

                # Merge and display red (non-bbox) segments
                if red_points:
                    red_all = np.vstack(red_points)
                else:
                    red_all = np.zeros((0, 3))

                red_cam_pcd.points = o3d.utility.Vector3dVector(red_all)
                red_cam_pcd.colors = o3d.utility.Vector3dVector(
                    np.tile([1.0, 0.0, 0.0], (len(red_all), 1))
                )
                vis.update_geometry(red_cam_pcd)

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[VIZ] Visualization stopped.")
    finally:
        vis.destroy_window()
