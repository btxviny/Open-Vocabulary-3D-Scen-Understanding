# Open 3D Scene Understanding

Open-vocabulary 3D scene understanding and semantic search built on CLIP, SAM2, and SAM3D. Query a 3D scene in natural language to find objects and regions.

![SPARC screenshot](https://github.com/user-attachments/assets/4a5d4851-a2fb-4d45-a646-319acbb8cc16)

## Overview

The pipeline takes RGB-D frames + camera poses and produces a searchable 3D point cloud:

1. **SAM3D segmentation** — SAM2 masks projected into 3D, grouped into object clusters
2. **CLIP embeddings** — per-point semantic embeddings from patch-level CLIP
3. **Dense point cloud** — unprojected depth fused across all frames
4. **Interpolation** — CLIP embeddings and cluster IDs propagated to every dense point
5. **Vector store** — ChromaDB index for fast natural-language retrieval

An optional interactive segmentation step lets you click seed points to refine clusters.

---

## Supported Input Formats

| Source | Pose file | `--camera_type` |
|---|---|---|
| Intel RealSense D435i | `extrinsic_matrix.npy` | `realsense` |
| ScanNet `.sens` (extracted) | `pose.npy` | `scannet` |

---

## Installation

### Quick

```bash
./install_sparc.sh
conda activate sparc
```

The script installs conda, CUDA 12.1, PyTorch, SAM2, and SAM3D automatically.

### Manual

```bash
conda create -n sparc python=3.10
conda activate sparc
conda install -c nvidia cuda=12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install plyfile -c conda-forge -y
pip install -r requirements.txt
```

```bash
export CUDA_HOME=$CONDA_PREFIX CUDA_PATH=$CONDA_PREFIX
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

```bash
# SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .
cd checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
cd ../..

# SAM3D
git clone https://github.com/Pointcept/SegmentAnything3D.git
cd SegmentAnything3D/libs/pointops
TORCH_CUDA_ARCH_LIST="7.5 8.0" python setup.py install
cd ../../..
```

---

## Usage — RealSense data

Organise your frames as:

```
data_directory/
├── 000000/
│   ├── rgb_image.png
│   ├── depth_image.png
│   └── extrinsic_matrix.npy   # 4×4 camera-to-world
├── 000001/
│   └── ...
```

Then run:

```bash
cd src
python pipeline.py --run_path /path/to/data_directory --scene_name "My Scene"
```

---

## Usage — ScanNet data

### 1. Download a scan

```bash
python ScanNet/download.py -o ScanNet/scans --id scene0001_00
```

> You need a ScanNet licence. Follow the instructions at [ScanNet](https://github.com/ScanNet/ScanNet) to request access and obtain the download script token.

### 2. Extract + run pipeline in one step

```bash
python ScanNet/prepare_scene.py \
    --sens ScanNet/scans/scene0001_00/scene0001_00.sens \
    --out_dir ScanNet/RGBD/scene0001_00 \
    --frame_skip 10 \
    --run_pipeline \
    --scene_name "ScanNet scene0001_00"
```

`--frame_skip 10` processes every 10th frame — good for quick iteration. Remove or set to `1` for full quality.

### 2a. Extract only (then run pipeline separately)

```bash
# Extract frames
python ScanNet/prepare_scene.py \
    --sens ScanNet/scans/scene0001_00/scene0001_00.sens \
    --out_dir ScanNet/RGBD/scene0001_00 \
    --frame_skip 10

# Run pipeline
cd src
python pipeline.py \
    --run_path ../ScanNet/RGBD \
    --scene_name "scene0001_00" \
    --camera_type scannet
```

---

## Run the app

```bash
streamlit run app.py
```

---

## Configuration

`src/config.yaml` controls all pipeline parameters:

| Key | Default | Description |
|---|---|---|
| `frame_stride` | 10 | Frame sampling stride during processing |
| `point_stride` | 10 | Point sampling stride for depth unprojection |
| `interpolation_k` | 21 | KNN neighbours for embedding interpolation |
| `sam3d.voxel_size` | 0.01 | Voxel size for 3D grouping (metres) |
| `sam3d.group_overlap_ratio` | 0.2 | Minimum overlap to merge SAM masks |

Camera intrinsics for both `realsense` and `scannet` are also in `src/config.yaml`.

---

## ScanNet utilities

| Script | Purpose |
|---|---|
| `ScanNet/prepare_scene.py` | Extract `.sens` → frame folders and optionally run pipeline |
| `ScanNet/download.py` | Download individual scans or the full release |
| `ScanNet/unpack_images.py` | Batch-extract multiple scenes from `.sens` files |
| `ScanNet/sens_to_pcd.py` | Convert `.sens` directly to a merged `.npz` point cloud |
| `ScanNet/scannet_sensordata.py` | Python 3 reader for ScanNet `.sens` binary format |

---

## Output

Each processed scene produces:

- `dense_pointcloud.npz` — dense point cloud with CLIP embeddings and per-point cluster IDs
- `vector_store/` — ChromaDB collection for semantic search (points + clusters)
