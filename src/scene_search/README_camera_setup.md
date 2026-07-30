# Camera Configuration System

This system allows you to handle different camera setups (RealSense, ScanNet, etc.) gracefully in your 3D scene processing pipeline using a unified approach.

## Overview

The system automatically detects the camera type based on your data structure and applies the appropriate:
- Camera intrinsics (focal length, principal point)
- Camera-to-IMU transformation matrices
- Pose file loading patterns

**All configurations are now centralized in `config.yaml`** for easy management.

## Supported Camera Types

### 1. RealSense D435i (`realsense`)
- **Intrinsics**: fx=386.99, fy=386.99, ppx=320.97, ppy=241.47
- **Transformation**: Includes camera-to-IMU transformation for lidar-based localization
- **Pose files**: `extrinsic_matrix.npy`

### 2. ScanNet (`scannet`)
- **Intrinsics**: fx=577.87, fy=577.87, ppx=319.5, ppy=239.5
- **Transformation**: Identity matrix (no additional transformation needed)
- **Pose files**: `pose.txt`, `pose.npy`, `extrinsic_matrix.npy`

## Configuration File

All camera settings are now stored in `config.yaml`:

```yaml
# Camera configurations
cameras:
  realsense:
    intrinsics:
      fx: 386.99176025390625
      fy: 386.99176025390625
      ppx: 320.9659118652344
      ppy: 241.470703125
    cam_to_imu:
      - [0.894078, 0.012117, 0.447747, 0.008695]
      - [-0.446418, -0.057450, 0.892979, 0.127623]
      - [0.036543, -0.998275, -0.045955, 0.761847]
      - [0.000000, 0.000000, 0.000000, 1.000000]
    pose_file_patterns: ["extrinsic_matrix.npy"]
    description: "RealSense D435i with IMU transformation for lidar-based localization"
  
  scannet:
    intrinsics:
      fx: 577.870605
      fy: 577.870605
      ppx: 319.5
      ppy: 239.5
    cam_to_imu:
      - [1.0, 0.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0, 0.0]
      - [0.0, 0.0, 1.0, 0.0]
      - [0.0, 0.0, 0.0, 1.0]
    pose_file_patterns: ["pose.txt", "pose.npy", "extrinsic_matrix.npy"]
    description: "ScanNet dataset format with direct camera-to-world transformation"
```

## Usage

### Unified Pipeline (Recommended)
```bash
# Automatic detection
python pipeline.py --run_path /path/to/data --scene_name "Living Room" --emoji "🏠"

# Manual specification
python pipeline.py --run_path /path/to/data --scene_name "Living Room" --emoji "🏠" --camera_type realsense
python pipeline.py --run_path /path/to/data --scene_name "Living Room" --emoji "🏠" --camera_type scannet
```

### Individual Modules
```bash
# SAM3D segmentation
python -m scene_search.sam3d --base_dir /path/to/data --save_path output.npz --camera_type auto

# Dense pointcloud generation
python -m scene_search.dense_pointcloud --base_dir /path/to/data --save_path output.npz --camera_type auto
```

## Data Structure Requirements

### RealSense Format
```
base_dir/
├── frame_001/
│   ├── rgb_image.png
│   ├── depth_image.png
│   └── extrinsic_matrix.npy
├── frame_002/
│   ├── rgb_image.png
│   ├── depth_image.png
│   └── extrinsic_matrix.npy
└── ...
```

### ScanNet Format
```
base_dir/
├── frame_001/
│   ├── rgb_image.png
│   ├── depth_image.png
│   └── pose.txt (or pose.npy or extrinsic_matrix.npy)
├── frame_002/
│   ├── rgb_image.png
│   ├── depth_image.png
│   └── pose.txt (or pose.npy or extrinsic_matrix.npy)
└── ...
```

## Adding New Camera Types

To add a new camera type, edit `config.yaml`:

```yaml
cameras:
  new_camera:
    intrinsics:
      fx: 500.0
      fy: 500.0
      ppx: 320.0
      ppy: 240.0
    cam_to_imu:
      - [1.0, 0.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0, 0.0]
      - [0.0, 0.0, 1.0, 0.0]
      - [0.0, 0.0, 0.0, 1.0]
    pose_file_patterns: ["extrinsic_matrix.npy"]
    description: "New camera description"
```

## Key Benefits of the Unified System

1. **Single Pipeline**: One `pipeline.py` handles both RealSense and ScanNet
2. **Single SAM3D**: One `sam3d.py` script for all camera types
3. **Centralized Config**: All settings in `config.yaml`
4. **Automatic Detection**: No need to specify camera type for most use cases
5. **Easy Extension**: Add new camera types by editing config file
6. **Consistent Interface**: Same commands for all camera types

## Testing

Run the test script to verify your configuration:

```bash
python -m scene_search.test_camera_configs
```

This will show all available camera types and their configurations.

## Troubleshooting

### "Unknown camera type" Error
- Check that your camera type is listed in `config.yaml`
- Use `--camera_type auto` for automatic detection
- Verify your data structure matches the expected format

### Pose Loading Issues
- Ensure pose files exist in your frame directories
- Check file format (`.txt`, `.npy`, etc.)
- Verify pose matrices are 4x4 transformation matrices

### Intrinsics Mismatch
- Verify camera intrinsics in `config.yaml`
- Use camera calibration tools to get accurate values
- Check if your camera model matches the configuration

### Config File Not Found
- Ensure `config.yaml` exists in the project root
- Check file paths and working directory
- Verify YAML syntax is correct
