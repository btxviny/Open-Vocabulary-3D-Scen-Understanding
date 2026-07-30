"""
Extract ScanNet .sens files to per-frame folder trees.

Exports every Nth frame (--frame_skip) from every scene found under
--scannet_path, writing each frame into its own numbered sub-folder under
--output_path/<scene_name>/<frame_idx:06d>/:

    rgb_image.png   – colour frame, resized to 480 × 640
    depth_image.png – 16-bit PNG, unit = mm
    pose.npy        – 4×4 camera-to-world float32 matrix

Also writes <output_path>/<scene_name>/intrinsics.json with per-scene
depth-camera intrinsics.

Usage (from ScanNet/ directory):
    python unpack_images.py \\
        --scannet_path /data/ScanNet/scans \\
        --output_path  /data/ScanNet/RGBD \\
        --frame_skip   10
"""

import argparse
import json
import os
import sys

import imageio
import numpy as np
import skimage.transform as sktf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from scannet_sensordata import SensorData


def print_error(message: str) -> None:
    sys.stderr.write(f"ERROR: {message}\n")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack ScanNet .sens files to per-frame folders")
    parser.add_argument("--scannet_path", required=True, help="Root directory of ScanNet scans")
    parser.add_argument("--output_path",  required=True, help="Destination directory")
    parser.add_argument("--frame_skip",   type=int, default=1, help="Export every Nth frame")
    opt = parser.parse_args()

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    scenes = sorted(
        d for d in os.listdir(opt.scannet_path)
        if os.path.isdir(os.path.join(opt.scannet_path, d))
    )
    print(f"Found {len(scenes)} scenes")

    for i, scene in enumerate(scenes):
        sens_file = os.path.join(opt.scannet_path, scene, f"{scene}.sens")
        if not os.path.exists(sens_file):
            print_error(f"{sens_file} does not exist!")

        output_scene_path = os.path.join(opt.output_path, scene)
        os.makedirs(output_scene_path, exist_ok=True)

        print(f"[{i + 1}/{len(scenes)}] Processing {scene}…")
        sd = SensorData(sens_file)

        frame_indices = range(0, len(sd.frames), opt.frame_skip)
        for export_idx, frame_idx in tqdm(enumerate(frame_indices), total=len(frame_indices)):
            frame_folder = os.path.join(output_scene_path, f"{export_idx:06d}")
            os.makedirs(frame_folder, exist_ok=True)

            frame = sd.frames[frame_idx]

            color = frame.decompress_color(sd.color_compression_type)
            color_resized = sktf.resize(color, (480, 640), order=1, preserve_range=True).astype(np.uint8)
            imageio.imwrite(os.path.join(frame_folder, "rgb_image.png"), color_resized)

            depth_bytes = frame.decompress_depth(sd.depth_compression_type)
            depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(sd.depth_height, sd.depth_width)
            imageio.imwrite(os.path.join(frame_folder, "depth_image.png"), depth)

            np.save(os.path.join(frame_folder, "pose.npy"), frame.camera_to_world)

        intr = sd.intrinsic_depth
        with open(os.path.join(output_scene_path, "intrinsics.json"), "w") as f:
            json.dump({
                "fx": float(intr[0, 0]),
                "fy": float(intr[1, 1]),
                "cx": float(intr[0, 2]),
                "cy": float(intr[1, 2]),
                "width":  int(sd.depth_width),
                "height": int(sd.depth_height),
            }, f, indent=2)

        print(f"  Exported {len(list(frame_indices))} frames from {scene}.")


if __name__ == "__main__":
    main()
