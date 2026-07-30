"""
End-to-end scene processing pipeline.

Usage (from repo root):
    uv run python -m src.pipeline \
        --run_path /path/to/scene_frames \
        --scene_name "Living Room" \
        --camera_type scannet
"""

import argparse
import subprocess
import yaml
from pathlib import Path
from time import time

# scenes_config.yaml lives at repo root (one level above src/)
_REPO_ROOT = Path(__file__).parent.parent
_SCENES_CONFIG = _REPO_ROOT / "scenes_config.yaml"


def run_step(msg: str, cmd: str) -> bool:
    print(f"\n{msg}")
    t0 = time()
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=Path(__file__).parent)
        print(f"Done in {time() - t0:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed: {e}")
        return False


def detect_camera_type(run_path: str) -> str:
    dirs = [x for x in Path(run_path).iterdir() if x.is_dir()]
    if not dirs:
        return "realsense"
    first = sorted(dirs)[0]
    if (first / "extrinsic_matrix.npy").exists():
        return "realsense"
    if (first / "pose.npy").exists() or (first / "pose.txt").exists():
        return "scannet"
    return "realsense"


def load_scenes_config() -> dict:
    if _SCENES_CONFIG.exists():
        with open(_SCENES_CONFIG) as f:
            return yaml.safe_load(f) or {"scenes": {}}
    return {"scenes": {}}


def save_scenes_config(config: dict) -> None:
    with open(_SCENES_CONFIG, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main(run_path: str, scene_name: str, camera_type: str | None = None,
         interactive: bool = False, force: bool = False) -> None:
    t0 = time()

    config = load_scenes_config()
    if scene_name in config.get("scenes", {}):
        if force:
            print(f"Overwriting existing scene: {scene_name}")
        else:
            ans = input(f"Scene '{scene_name}' already exists. Overwrite? (y/N): ")
            if ans.lower() != "y":
                print("Cancelled.")
                return

    scene_dir = Path(run_path) / scene_name
    scene_dir.mkdir(exist_ok=True)
    print(f"Scene directory: {scene_dir}")

    if camera_type is None or camera_type == "auto":
        camera_type = detect_camera_type(run_path)
        print(f"Auto-detected camera type: {camera_type}")

    steps = {
        "Step 0 – SAM3D segmentation": (
            f"python -m scene_search.sam3d"
            f" --base_dir {run_path}"
            f" --save_path {scene_dir}/segmented_pointcloud.npz"
            f" --camera_type {camera_type}"
        ),
        "Step 1 – CLIP point cloud": (
            f"python -m scene_search.clip"
            f" --base_dir {run_path}"
            f" --save_path {scene_dir}/clip_pointcloud.npz"
            f" --logic patch"
        ),
        "Step 2 – Dense point cloud": (
            f"python -m scene_search.dense_pointcloud"
            f" --base_dir {run_path}"
            f" --save_path {scene_dir}/dense_pointcloud.npz"
            f" --camera_type {camera_type}"
        ),
        "Step 3 – Interpolate embeddings + cluster IDs": (
            f"python -m scene_search.interpolate"
            f" --dense_pc_path {scene_dir}/dense_pointcloud.npz"
            f" --sparse_pc_path {scene_dir}/clip_pointcloud.npz"
            f" --segmented_pc_path {scene_dir}/segmented_pointcloud.npz"
        ),
        "Step 4 – Vector store (points)": (
            f"python -m scene_search.vectorstore"
            f" --npz_file {scene_dir}/dense_pointcloud.npz"
            f" --logic points"
            f" --collection_name {scene_name}_points"
        ),
    }

    for msg, cmd in steps.items():
        if not run_step(msg, cmd):
            return

    if interactive:
        if not run_step(
            "Step 5 – Interactive segmentation",
            f"python -m scene_search.interactive_segmentation"
            f" --input {scene_dir}/dense_pointcloud.npz"
            f" --output {scene_dir}/dense_pointcloud.npz",
        ):
            return
    else:
        if not run_step(
            "Step 5 – Vector store (clusters)",
            f"python -m scene_search.vectorstore"
            f" --npz_file {scene_dir}/dense_pointcloud.npz"
            f" --logic clusters"
            f" --collection_name {scene_name}_clusters",
        ):
            return

    config = load_scenes_config()
    config.setdefault("scenes", {})[scene_name] = {
        "points":   {"npz_file": str(scene_dir / "dense_pointcloud.npz"),
                     "chromadb_path": str(scene_dir / "vector_store"),
                     "collection": f"{scene_name}_points"},
        "clusters": {"npz_file": str(scene_dir / "dense_pointcloud.npz"),
                     "chromadb_path": str(scene_dir / "vector_store"),
                     "collection": f"{scene_name}_clusters"},
    }
    save_scenes_config(config)

    print(f"\nScene '{scene_name}' added to {_SCENES_CONFIG}")
    print(f"Camera: {camera_type} | Time: {(time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D scene processing pipeline")
    parser.add_argument("--run_path",    required=True, help="Directory of extracted frames")
    parser.add_argument("--scene_name",  required=True, help="Scene label (shown in app)")
    parser.add_argument("--camera_type", choices=["auto", "realsense", "scannet"], default="auto")
    parser.add_argument("--interactive", action="store_true",
                        help="Use interactive segmentation instead of SAM3D clusters")
    parser.add_argument("--force", action="store_true", help="Overwrite without prompting")
    args = parser.parse_args()
    main(args.run_path, args.scene_name, args.camera_type, args.interactive, args.force)
