import open3d as o3d
import os

def collect_all_pcds(base_path):
    all_pcds = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pcd") and file != "lidar.pcd":
                pcd_path = os.path.join(root, file)
                pcd = o3d.io.read_point_cloud(pcd_path)
                if not pcd.is_empty():
                    # Only paint gray if the point cloud doesn't already have color
                    if not pcd.has_colors():
                        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # gray fallback
                    all_pcds.append(pcd)
    return all_pcds

if __name__ == "__main__":
    # Change this to your actual directory
    base_dir = "/home/elias/code/data_logs/data_1744645809_90342760"
    pcds = collect_all_pcds(base_dir)

    if pcds:
        print(f"Visualizing {len(pcds)} point clouds together.")
        o3d.visualization.draw_geometries(pcds)
    else:
        print("No point clouds found.")
