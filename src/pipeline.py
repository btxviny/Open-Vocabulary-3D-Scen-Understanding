"""
Example Usage:
python pipeline.py --run_path /path/to/data --scene_name "Living Room" --emoji "🏠" --camera_type auto
"""
import subprocess
import argparse
import yaml
from pathlib import Path
from time import time

def run_step(msg, cmd):
    print(f"\n{msg}")
    try:    
        start_time = time()
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Done in {time() - start_time:.2f} seconds")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return False
    return True


def update_scene_config(scene_name: str, run_path: str):
    """Update scenes_config.yaml with the new scene."""
    config_path = Path("../scenes_config.yaml")
    
    # Load existing config or create new
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {"scenes": {}}
    else:
        config = {"scenes": {}}
    
    # Define collection names and scene directory
    points_collection = f"{scene_name}_points"
    clusters_collection = f"{scene_name}_clusters"
    scene_dir = Path(run_path) / scene_name
    
    # Add or update scene configuration
    config["scenes"][scene_name] = {
        "points": {
            "npz_file": str(scene_dir / "dense_pointcloud.npz"),
            "chromadb_path": str(scene_dir / "vector_store"),
            "collection": points_collection
        },
        "clusters": {
            "npz_file": str(scene_dir / "dense_pointcloud.npz"),
            "chromadb_path": str(scene_dir / "vector_store"),
            "collection": clusters_collection
        }
    }
    
    # Save updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def detect_camera_type(run_path: str):
    """
    Detect camera type based on data structure.
    Returns 'realsense' or 'scannet'.
    """
    # Look for characteristic files
    frames_dir = [x for x in Path(run_path).iterdir() if x.is_dir()]
    
    if not frames_dir:
        return 'realsense'  # default
    
    first_frame = frames_dir[0]
    
    # Check for RealSense format (extrinsic_matrix.npy)
    if (first_frame / 'extrinsic_matrix.npy').exists():
        return 'realsense'
    # Check for ScanNet format (pose.txt or pose.npy)
    elif any((first_frame / f).exists() for f in ['pose.txt', 'pose.npy']):
        return 'scannet'
    
    return 'realsense'  # default


def main(run_path: str, scene_name: str, camera_type: str = None, force: bool = False):
    # Check if scene already exists in config
    start_time = time()
    config_path = Path("../scenes_config.yaml")
    print(f"📋 Using default config: {config_path}")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {"scenes": {}}
            if scene_name in config["scenes"]:
                if force:
                    print(f"🔄 Force overwriting existing scene: {scene_name}")
                else:
                    response = input(f"Scene '{scene_name}' already exists. Do you want to overwrite it? (y/N): ")
                    if response.lower() != 'y':
                        print("Operation cancelled.")
                        return

    # Create scene directory
    scene_dir = Path(run_path) / scene_name
    scene_dir.mkdir(exist_ok=True)
    print(f"\n📁 Created scene directory: {scene_dir}")

    # Auto-detect camera type if not specified
    if camera_type == 'auto' or camera_type is None:
        camera_type = detect_camera_type(run_path)
        print(f"🔍 Auto-detected camera type: {camera_type}")

    # Common steps run regardless of method
    steps = {
        "0: Segmenting and clustering with SAM3D:": f'python -m scene_search.sam3d --base_dir {run_path} --save_path {scene_dir}/segmented_pointcloud.npz --camera_type {camera_type}',
        "1: Creating CLIP pointcloud": f'python -m scene_search.clip --base_dir {run_path} --save_path {scene_dir}/clip_pointcloud.npz --logic patch',
        "2: Creating dense pointcloud": f'python -m scene_search.dense_pointcloud --base_dir {run_path} --save_path {scene_dir}/dense_pointcloud.npz --camera_type {camera_type}',
        "3: Interpolating embeddings and cluster IDs": f'python -m scene_search.interpolate --dense_pc_path {scene_dir}/dense_pointcloud.npz --sparse_pc_path {scene_dir}/clip_pointcloud.npz  --segmented_pc_path {scene_dir}/segmented_pointcloud.npz',
        "4: Creating vectorstore for dense clip pointcloud": f'python -m scene_search.vectorstore --npz_file {scene_dir}/dense_pointcloud.npz --logic points --collection_name {scene_name}_points',
    }

    print(f"\n🟢 Running steps for scene: {scene_name} with {camera_type} camera")
    for msg, cmd in steps.items():
        if not run_step(msg, cmd):
            return

    # Handle interactive segmentation based on flag
    if args.interactive:
        print("\n🟢 Running interactive segmentation...")
        if not run_step("Interactive segmentation", 
            f'python -m scene_search.interactive_segmentation --input {scene_dir}/dense_pointcloud.npz --output {scene_dir}/dense_pointcloud.npz'):
            return
    else:
        print("\n🟢 Using SAM3D segmentated objects...")
        if not run_step("5: Creating vectorstore for clustered pointcloud",
            f'python -m scene_search.vectorstore --npz_file {scene_dir}/dense_pointcloud.npz --logic clusters --collection_name {scene_name}_clusters'):
            return

    # Update the configuration file with scene-specific paths
    config = {"scenes": {}}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {"scenes": {}}

    points_collection = f"{scene_name}_points"
    clusters_collection = f"{scene_name}_clusters"
    
    config["scenes"][scene_name] = {
        "points": {
            "npz_file": str(scene_dir / "dense_pointcloud.npz"),
            "chromadb_path": str(scene_dir / "vector_store"),
            "collection": points_collection
        },
        "clusters": {
            "npz_file": str(scene_dir / "dense_pointcloud.npz"),
            "chromadb_path": str(scene_dir / "vector_store"),
            "collection": clusters_collection
        }
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n🟢 Scene '{scene_name}' has been processed and added to scenes_config.yaml")
    print(f"Camera type used: {camera_type}")
    print(f"Time taken: {(time() - start_time) / 60:.2f} minutes")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Scene Segmentation Pipeline")
    parser.add_argument('--run_path', type=str, required=True, help="Path to data from a run")
    parser.add_argument('--scene_name', type=str, required=True, help="Name of the scene (will be displayed in the UI)")
    parser.add_argument('--emoji', type=str, default="", help="Optional emoji to append to the scene name")
    parser.add_argument('--camera_type', type=str, choices=['auto', 'realsense', 'scannet'], default='auto',
                       help="Camera type to use. 'auto' will attempt to detect automatically.")
    parser.add_argument('--interactive', action='store_true',
                       help="Enable interactive segmentation. If not specified, uses SAM3D segmented objects.")
    parser.add_argument('--force', action='store_true',
                       help="Force overwrite existing scenes without prompting")
    args = parser.parse_args()
    
    # Combine scene name and emoji if provided
    display_name = f"{args.scene_name} {args.emoji}".strip()
    main(args.run_path, display_name, args.camera_type, args.force)