"""
End-to-end ScanNet scene preparation for the pipeline.

Extracts a ScanNet .sens file into per-frame folders that the pipeline expects,
then optionally runs the full pipeline.

Usage:
    # Prepare only (extract frames):
    python ScanNet/prepare_scene.py --sens path/to/scene.sens --out_dir data/scene0001_00

    # Prepare and run pipeline:
    python ScanNet/prepare_scene.py --sens path/to/scene.sens --out_dir data/scene0001_00 --run_pipeline --scene_name "Living Room"
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import imageio
import skimage.transform as sktf

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from scannet_sensordata import SensorData

OUTPUT_HEIGHT = 480
OUTPUT_WIDTH = 640


def extract_scene(sens_path: str, out_dir: str, frame_skip: int = 1) -> int:
    """Extract a .sens file to per-frame folders.

    Each folder contains:
        rgb_image.png   – colour frame resized to OUTPUT_HEIGHT × OUTPUT_WIDTH
        depth_image.png – 16-bit depth (millimetres)
        pose.npy        – 4×4 camera-to-world matrix (float32)

    Returns the number of frames written.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {sens_path}...")
    sd = SensorData(sens_path)
    total = len(sd.frames)
    n_out = total // frame_skip
    print(f"  {total} frames → exporting {n_out} (skip={frame_skip})")

    written = 0
    for export_idx, frame_idx in enumerate(range(0, total, frame_skip)):
        frame = sd.frames[frame_idx]
        frame_dir = out_path / f"{export_idx:06d}"
        frame_dir.mkdir(exist_ok=True)

        # --- colour ---
        color = frame.decompress_color(sd.color_compression_type)
        color_resized = sktf.resize(
            color, (OUTPUT_HEIGHT, OUTPUT_WIDTH),
            order=1, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        imageio.imwrite(str(frame_dir / "rgb_image.png"), color_resized)

        # --- depth (16-bit PNG, unit = mm) ---
        depth_bytes = frame.decompress_depth(sd.depth_compression_type)
        depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(
            sd.depth_height, sd.depth_width
        )
        imageio.imwrite(str(frame_dir / "depth_image.png"), depth)

        # --- pose saved as pose.npy so pipeline detects camera_type=scannet ---
        np.save(str(frame_dir / "pose.npy"), frame.camera_to_world)

        written += 1

    # Per-scene intrinsics from the .sens depth camera (vary across ScanNet scenes)
    intr = sd.intrinsic_depth
    intrinsics_data = {
        "fx": float(intr[0, 0]),
        "fy": float(intr[1, 1]),
        "cx": float(intr[0, 2]),
        "cy": float(intr[1, 2]),
        "width": int(sd.depth_width),
        "height": int(sd.depth_height),
        "camera_type": "scannet",
    }
    with open(out_path / "intrinsics.json", "w") as f:
        json.dump(intrinsics_data, f, indent=2)
    print(f"  Saved intrinsics: fx={intr[0,0]:.2f} fy={intr[1,1]:.2f} cx={intr[0,2]:.1f} cy={intr[1,2]:.1f}")

    print(f"Wrote {written} frames to {out_dir}")
    return written


def run_pipeline(out_dir: str, scene_name: str, force: bool = False) -> None:
    """Run the pipeline on an already-extracted scene directory."""
    repo_root = Path(__file__).parent.parent
    cmd = (
        f'python -m src.pipeline'
        f' --run_path "{out_dir}"'
        f' --scene_name "{scene_name}"'
        + (" --force" if force else "")
    )
    print(f"\nRunning pipeline:\n  {cmd}\n")
    subprocess.run(cmd, shell=True, check=True, cwd=repo_root)


def main():
    parser = argparse.ArgumentParser(description="Prepare a ScanNet scene for the pipeline")
    parser.add_argument("--sens", required=True, help="Path to .sens file")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for extracted frames")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Export every Nth frame (default: 1)")
    parser.add_argument("--run_pipeline", action="store_true",
                        help="Run the pipeline after extraction")
    parser.add_argument("--scene_name", type=str, default="",
                        help="Scene name for the pipeline (required if --run_pipeline)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing scene in config without prompting")
    args = parser.parse_args()

    if not os.path.isfile(args.sens):
        print(f"ERROR: .sens file not found: {args.sens}")
        sys.exit(1)

    extract_scene(args.sens, args.out_dir, args.frame_skip)

    if args.run_pipeline:
        if not args.scene_name:
            print("ERROR: --scene_name is required when --run_pipeline is set")
            sys.exit(1)
        run_pipeline(args.out_dir, args.scene_name, args.force)


if __name__ == "__main__":
    main()
