"""
End-to-end 3D scene understanding pipeline.

Runs five steps in sequence, then registers the scene in scenes_config.yaml
so the Streamlit app can find it.

Usage (from repo root):
    uv run python -m src.pipeline \\
        --run_path ScanNet/RGBD/scene0001_00 \\
        --scene_name "Living Room"

Steps
-----
0  segment      SAM2 2D masks + per-mask PCD files written into each frame dir
1  embed        CLIP-encode masks, recursive disk-spill merge → embedded_pointcloud.npz
2  fuse         Full RGBD unproject                          → dense_pointcloud.npz
3  interpolate  Spread sparse embeddings onto dense cloud (in-place update)
4  vectorstore  ChromaDB cosine index

Individual steps (for debugging):
    export BASE="$(pwd)/ScanNet/RGBD/scene0001_00"
    export SCENE="$BASE/Living Room" && mkdir -p "$SCENE"

    uv run python -m src.scene_search.segment     --base_dir "$BASE"
    uv run python -m src.scene_search.embed       --base_dir "$BASE" --save_path "$SCENE/embedded_pointcloud.npz"
    uv run python -m src.scene_search.fuse        --base_dir "$BASE" --save_path "$SCENE/dense_pointcloud.npz"
    uv run python -m src.scene_search.interpolate --dense_pc_path "$SCENE/dense_pointcloud.npz" --sparse_pc_path "$SCENE/embedded_pointcloud.npz"
    uv run python -m src.scene_search.vectorstore --npz_file "$SCENE/dense_pointcloud.npz" --collection_name "Living_Room_points"
"""

import argparse
import re
import subprocess
import yaml
from pathlib import Path
from time import time

_REPO_ROOT     = Path(__file__).parent.parent
_SCENES_CONFIG = _REPO_ROOT / "scenes_config.yaml"
_CHECKPOINT    = _REPO_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"


def _sanitize(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    s = re.sub(r"^[^a-zA-Z0-9]+", "", s)
    s = re.sub(r"[^a-zA-Z0-9]+$", "", s)
    return s or "collection"


def _run(msg: str, cmd: str) -> bool:
    print(f"\n{msg}")
    t0 = time()
    try:
        # cwd=src/ so modules are addressed as scene_search.* (not src.scene_search.*)
        subprocess.run(cmd, shell=True, check=True, cwd=Path(__file__).parent)
        print(f"  Done in {time() - t0:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed: {e}")
        return False


def _load_config() -> dict:
    if _SCENES_CONFIG.exists():
        return yaml.safe_load(_SCENES_CONFIG.read_text()) or {"scenes": {}}
    return {"scenes": {}}


def _save_config(cfg: dict) -> None:
    _SCENES_CONFIG.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def main(run_path: str, scene_name: str, force: bool = False) -> None:
    t0 = time()

    run_path  = str(Path(run_path).resolve())
    scene_dir = Path(run_path) / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    sd = str(scene_dir)

    collection = _sanitize(f"{scene_name}_points")

    cfg = _load_config()
    if scene_name in cfg.get("scenes", {}):
        if force:
            print(f"Overwriting existing scene: {scene_name!r}")
        else:
            ans = input(f"Scene {scene_name!r} already exists. Overwrite? (y/N): ")
            if ans.lower() != "y":
                print("Cancelled.")
                return

    steps = [
        ("Step 0 – SAM2 segmentation",
         f'python -m scene_search.segment'
         f' --base_dir "{run_path}"'
         f' --checkpoint "{_CHECKPOINT}"'),

        ("Step 1 – CLIP embedding + merge",
         f'python -m scene_search.embed'
         f' --base_dir "{run_path}"'
         f' --save_path "{sd}/embedded_pointcloud.npz"'),

        ("Step 2 – Dense depth fusion",
         f'python -m scene_search.fuse'
         f' --base_dir "{run_path}"'
         f' --save_path "{sd}/dense_pointcloud.npz"'),

        ("Step 3 – Interpolate embeddings",
         f'python -m scene_search.interpolate'
         f' --dense_pc_path "{sd}/dense_pointcloud.npz"'
         f' --sparse_pc_path "{sd}/embedded_pointcloud.npz"'),

        ("Step 4 – Vector store",
         f'python -m scene_search.vectorstore'
         f' --npz_file "{sd}/dense_pointcloud.npz"'
         f' --collection_name "{collection}"'),
    ]

    for msg, cmd in steps:
        if not _run(msg, cmd):
            return

    cfg = _load_config()
    cfg.setdefault("scenes", {})[scene_name] = {
        "npz_file":      f"{sd}/dense_pointcloud.npz",
        "chromadb_path": f"{sd}/vector_store",
        "collection":    collection,
    }
    _save_config(cfg)
    print(f"\nScene {scene_name!r} ready. Total time: {(time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D scene understanding pipeline")
    parser.add_argument("--run_path",   required=True, help="Directory containing per-frame subdirs")
    parser.add_argument("--scene_name", required=True, help="Scene label shown in the app")
    parser.add_argument("--force",      action="store_true", help="Overwrite existing scene")
    args = parser.parse_args()
    main(args.run_path, args.scene_name, args.force)
