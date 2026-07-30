"""Quick sanity-check for camera configuration loading. Run from repo root:

    uv run python -m scene_search.test_camera_configs
"""
import numpy as np

from .camera_configs import (
    list_available_cameras,
    get_camera_config,
    get_camera_description,
    get_pose_file_patterns,
)


def main():
    cameras = list_available_cameras()
    print(f"Camera types: {cameras}\n")

    for cam in cameras:
        print(f"[{cam}]")
        print(f"  desc    : {get_camera_description(cam)}")
        print(f"  patterns: {get_pose_file_patterns(cam)}")
        intr, c2i = get_camera_config(cam)
        print(f"  fx={intr['fx']:.2f}  fy={intr['fy']:.2f}  "
              f"cx={intr['ppx']:.2f}  cy={intr['ppy']:.2f}")
        print(f"  cam_to_imu identity: {np.allclose(c2i, np.eye(4))}")
        print()

    try:
        get_camera_config("nonexistent")
    except ValueError as e:
        print(f"Expected error: {e}")


if __name__ == "__main__":
    main()
