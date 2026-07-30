# Open 3D Scene Understanding

Open-vocabulary 3D scene understanding and semantic search.  
Load a ScanNet scene (or any RGB-D sequence), segment objects with SAM2, embed them with CLIP, and query the 3D world in natural language.

---

## How it works

```
.sens file
    │
    ▼
ScanNet/prepare_scene.py   ← extract frames (RGB + depth + pose)
    │
    ▼
src/pipeline.py
    ├── scene_search/sam3d.py       — SAM2 masks → 3D object clusters
    ├── scene_search/clip.py        — CLIP patch embeddings per object
    ├── scene_search/dense_pointcloud.py  — full-res depth fusion
    ├── scene_search/interpolate.py — propagate embeddings to dense cloud
    └── scene_search/vectorstore.py — ChromaDB index
    │
    ▼
app.py  (Streamlit + Rerun)   ← text / image → highlighted 3D regions
```

---

## Setup

### Requirements
- CUDA-capable GPU (tested on RTX 5070 Ti, cu128)
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
./setup.sh
```

This installs Python 3.10, PyTorch with CUDA 12.8, SAM2, CLIP, Open3D, and builds the `pointops` CUDA extension from SegmentAnything3D.

---

## ScanNet workflow

### 1. Download a scene

> You need a ScanNet licence — fill in the agreement on the ScanNet GitHub page.

```bash
python ScanNet/download.py -o ScanNet/scans --id scene0001_00
```

### 2. Extract + run full pipeline

```bash
python ScanNet/prepare_scene.py \
    --sens ScanNet/scans/scene0001_00/scene0001_00.sens \
    --out_dir ScanNet/RGBD/scene0001_00 \
    --frame_skip 10 \
    --run_pipeline \
    --scene_name "scene0001_00"
```

`--frame_skip 10` keeps 1 in 10 frames — good for fast iteration.

### 2a. Extract only, then run pipeline manually

```bash
# Extract frames from .sens
python ScanNet/prepare_scene.py \
    --sens ScanNet/scans/scene0001_00/scene0001_00.sens \
    --out_dir ScanNet/RGBD/scene0001_00 \
    --frame_skip 10

# Run pipeline
uv run python -m src.pipeline \
    --run_path ScanNet/RGBD \
    --scene_name scene0001_00 \
    --camera_type scannet
```

---

## App

```bash
uv run streamlit run app.py
```

- **Load scene** — loads CLIP model, ChromaDB index, and point cloud into the Rerun viewer
- **Search** — text or image query; matching points/clusters highlighted in the Rerun 3D view

---

## Configuration

Edit `src/config.yaml`:

| Key | Default | Description |
|---|---|---|
| `frame_stride` | 10 | Frame sampling during processing |
| `point_stride` | 10 | Point sampling per frame |
| `interpolation_k` | 21 | KNN neighbours for embedding propagation |
| `sam3d.voxel_size` | 0.01 | Voxel grid size in metres |
| `sam3d.group_overlap_ratio` | 0.2 | Mask merge threshold |

Camera intrinsics for `realsense` and `scannet` are also in `src/config.yaml`.

---

## Project layout

```
open-3d-scene-understanding/
├── app.py                        # Streamlit search UI
├── pyproject.toml                # uv dependencies (PyTorch cu128, SAM2, CLIP, …)
├── setup.sh                      # One-shot install
├── ScanNet/
│   ├── prepare_scene.py          # .sens → frame folders + optional pipeline run
│   ├── download.py               # ScanNet download script
│   ├── unpack_images.py          # Batch .sens extraction
│   └── scannet_sensordata.py     # Python 3 .sens reader
└── src/
    ├── config.yaml               # Pipeline parameters + camera intrinsics
    ├── pipeline.py               # Orchestrates all steps for one scene
    └── scene_search/
        ├── sam3d.py              # SAM2 + 3D mask merging
        ├── clip.py               # CLIP embeddings per object mask
        ├── dense_pointcloud.py   # Vectorised depth fusion
        ├── interpolate.py        # Embedding + cluster ID propagation
        ├── vectorstore.py        # ChromaDB index creation
        ├── camera_configs.py     # Per-camera intrinsics loader
        └── utils.py              # Shared utilities
```
