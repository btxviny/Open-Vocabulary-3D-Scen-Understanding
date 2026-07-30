import argparse
import json
import os
import sys
import numpy as np
import skimage.transform as sktf
import imageio
from scannet_sensordata import SensorData
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--output_path', required=True, help='where to output 2d data')
parser.add_argument('--frame_skip', type=int, default=1, help='export every nth frame')
opt = parser.parse_args()
print(opt)

def print_error(message):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    sys.exit(-1)

def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    scenes = sorted([d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))])
    print('Found %d scenes' % len(scenes))

    for i, scene in enumerate(scenes):
        sens_file = os.path.join(opt.scannet_path, scene, scene + '.sens')
        if not os.path.exists(sens_file):
            print_error(f"{sens_file} does not exist!")

        output_scene_path = os.path.join(opt.output_path, scene)
        if not os.path.isdir(output_scene_path):
            os.makedirs(output_scene_path)

        print(f"[{i+1}/{len(scenes)}] Processing {scene}...")

        sd = SensorData(sens_file)

        # export each frame to its own folder
        for f_idx, frame_idx in tqdm(enumerate(range(0, len(sd.frames), opt.frame_skip)), total=len(sd.frames)//opt.frame_skip):
            frame_folder = os.path.join(output_scene_path, f"{f_idx:06d}")
            os.makedirs(frame_folder, exist_ok=True)

            frame = sd.frames[frame_idx]

            # Color (resize to 480x640)
            color_img = frame.decompress_color(sd.color_compression_type)
            color_resized = sktf.resize(color_img, (480, 640), order=1, preserve_range=True).astype(np.uint8)
            imageio.imwrite(os.path.join(frame_folder, 'rgb_image.png'), color_resized)

            # Depth (keep original resolution)
            depth_data = frame.decompress_depth(sd.depth_compression_type)
            depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape(sd.depth_height, sd.depth_width)
            imageio.imwrite(os.path.join(frame_folder, 'depth_image.png'), depth_img)

            # Pose saved as pose.npy so pipeline auto-detects camera_type=scannet
            np.save(os.path.join(frame_folder, 'pose.npy'), frame.camera_to_world)

        # Per-scene intrinsics from the .sens depth camera
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
        with open(os.path.join(output_scene_path, "intrinsics.json"), "w") as f:
            json.dump(intrinsics_data, f, indent=2)

        print(f"Finished exporting {len(sd.frames)//opt.frame_skip} frames for {scene}.")

if __name__ == "__main__":
    main()
