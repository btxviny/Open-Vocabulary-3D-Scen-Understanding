import os
import argparse

def main(base_dir):
    # Define expected files in each frame directory
    expected_files = {"rgb_image.png", "depth_image.png", "extrinsic_matrix.npy"}

    # Iterate over subdirectories
    for frame_dir in os.listdir(base_dir):
        frame_path = os.path.join(base_dir, frame_dir)
        if os.path.isdir(frame_path):
            for fname in os.listdir(frame_path):
                if fname not in expected_files:
                    fpath = os.path.join(frame_path, fname)
                    try:
                        os.remove(fpath)
                        print(f"Deleted: {fpath}")
                    except Exception as e:
                        print(f"Could not delete {fpath}: {e}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear directory of unnecessary files.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing frame subfolders.")
    args = parser.parse_args()
    main(args.base_dir)
