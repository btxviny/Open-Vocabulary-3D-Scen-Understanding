# Open 3D Scene Understanding

Open-vocabulary 3D scene understanding: load a ScanNet scene, segment objects with SAM2, embed them with CLIP, and query the 3D world by text or image.

---

## How it works

```
scene.sens
    │
    ▼
ScanNet/prepare_scene.py     ← extract RGB + depth + pose per frame, save intrinsics.json
    │
    ▼
src/pipeline.py
    ├── sam3d.py             — SAM2 → per-frame masks → 3D object clusters
    ├── clip.py              — CLIP embedding per object (patch/bin/dilated crop)
    ├── dense_pointcloud.py  — vectorised depth fusion (full-res point cloud)
    ├── interpolate.py       — propagate embeddings + cluster IDs to dense cloud
    └── vectorstore.py       — ChromaDB index
    │
    ▼
app.py  (Streamlit + Rerun)  ← text / image query → highlighted 3D regions
```

---

## Setup

**Requirements**: CUDA GPU (tested on RTX 5070 Ti, driver 610, CUDA 13.3), [uv](https://docs.astral.sh/uv/)

```bash
./setup.sh
```

Installs Python 3.11, PyTorch cu128, SAM2, CLIP, Open3D, ChromaDB, and builds the `pointops` CUDA extension from SegmentAnything3D.  If `nvcc` is not on `PATH`, the script downloads and installs CUDA 12.8 to `~/.local/cuda-12.8` (no sudo needed).

---

## ScanNet workflow

### 1. Get access & download a scene

ScanNet requires a licence — fill in the agreement at the ScanNet GitHub page.

```bash
python ScanNet/download.py -o ScanNet/scans --id scene0001_00
```

### 2. Extract frames and run the full pipeline

```bash
python ScanNet/prepare_scene.py \
    --sens ScanNet/scans/scene0001_00/scene0001_00.sens \
    --out_dir ScanNet/RGBD/scene0001_00 \
    --frame_skip 10 \
    --run_pipeline \
    --scene_name "Living Room"
```

`--frame_skip 10` keeps 1 in 10 frames — good for fast iteration.

### 2a. Extract only, then run the pipeline manually

```bash
# Extract frames
python ScanNet/prepare_scene.py \
    --sens ScanNet/scans/scene0001_00/scene0001_00.sens \
    --out_dir ScanNet/RGBD/scene0001_00 \
    --frame_skip 10

# Run pipeline
uv run python -m src.pipeline \
    --run_path ScanNet/RGBD/scene0001_00 \
    --scene_name "Living Room"
```

Per-scene camera intrinsics are read from the `.sens` file automatically and saved to `intrinsics.json` alongside the extracted frames — no manual configuration needed.

---

## App

```bash
uv run streamlit run app.py
```

- **Load scene** — loads CLIP model, ChromaDB index, and point cloud into Rerun
- **Search** — text or image query; matching points or clusters highlighted in 3D

---

## Configuration

Edit `src/config.yaml`:

| Key | Default | Description |
|---|---|---|
| `frame_stride` | 10 | Frame sampling in the CLIP step |
| `point_stride` | 10 | Point sampling per depth frame |
| `interpolation_k` | 21 | KNN neighbours for embedding propagation |
| `min_depth` / `max_depth` | 0.5 / 6.0 | Depth filtering range (metres) |
| `sam3d.voxel_size` | 0.01 | Voxel grid cell size (metres) |
| `sam3d.group_overlap_ratio` | 0.2 | Mask merge threshold across frames |

RealSense camera intrinsics and the cam-to-IMU transform are in `src/config.yaml`.  
ScanNet intrinsics are read per-scene from the `.sens` file and stored in `intrinsics.json` beside the extracted frames.

---

## Project layout

```
open-3d-scene-understanding/
├── app.py                        # Streamlit search UI + Rerun visualisation
├── pyproject.toml                # uv dependencies (PyTorch cu128, SAM2, CLIP, …)
├── setup.sh                      # One-shot install (uv + SAM2 checkpoint + pointops)
├── ScanNet/
│   ├── prepare_scene.py          # .sens → frame folders + optional pipeline launch
│   ├── unpack_images.py          # Batch .sens extraction across multiple scenes
│   ├── sens_to_pcd.py            # Quick .sens → .npz without per-frame folders
│   ├── download.py               # Official ScanNet download helper
│   └── scannet_sensordata.py     # Python 3 .sens reader (from ScanNet repo)
├── tools/
│   ├── extract_video.py          # Extract frames from a video file
│   ├── visualize.py              # Point cloud visualisation helper
│   └── triangulate.py            # Triangulation utility
└── src/
    ├── config.yaml               # Pipeline parameters + RealSense camera config
    ├── pipeline.py               # Orchestrates all steps for one scene
    └── scene_search/
        ├── sam3d.py              # SAM2 masks → 3D object clusters
        ├── clip.py               # CLIP patch embeddings per object
        ├── dense_pointcloud.py   # Vectorised depth fusion
        ├── interpolate.py        # Embedding + cluster ID propagation
        ├── vectorstore.py        # ChromaDB index creation
        ├── camera_configs.py     # Per-camera intrinsics loader
        ├── interactive_segmentation.py  # Manual cluster refinement (Open3D)
        └── utils.py              # Shared utilities
```
