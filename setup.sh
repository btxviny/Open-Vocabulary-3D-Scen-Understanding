#!/usr/bin/env bash
# Setup script for Open 3D Scene Understanding
# Uses uv for Python/package management; installs CUDA toolkit to user space if needed.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
CUDA_LOCAL="$HOME/.local/cuda-12.8"   # user-space CUDA toolkit (no sudo needed)

echo "=== Open 3D Scene Understanding – Setup ==="
echo ""

# ── 1. uv ──────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/5] uv $(uv --version) already installed."
fi

# ── 2. Python environment + dependencies ───────────────────────────────────
echo ""
echo "[2/5] Installing dependencies (PyTorch cu128)..."
cd "$ROOT"
uv sync
echo "      Done."

# ── 3. SAM2 checkpoint ─────────────────────────────────────────────────────
echo ""
echo "[3/5] Downloading SAM2 checkpoint..."
CKPT_DIR="$ROOT/checkpoints"
mkdir -p "$CKPT_DIR"
CKPT="$CKPT_DIR/sam2.1_hiera_base_plus.pt"
if [ ! -f "$CKPT" ]; then
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" \
        -O "$CKPT"
else
    echo "      Checkpoint already exists, skipping."
fi

# ── 4. SegmentAnything3D (pointops CUDA extension) ────────────────────────
echo ""
echo "[4/5] Building SegmentAnything3D pointops extension..."

SA3D_DIR="$ROOT/SegmentAnything3D"
if [ ! -d "$SA3D_DIR" ]; then
    git clone https://github.com/Pointcept/SegmentAnything3D.git "$SA3D_DIR"
fi

# ── Locate or install nvcc ─────────────────────────────────────────────────
NVCC=""

# 1) System PATH
if command -v nvcc &>/dev/null; then
    NVCC=$(command -v nvcc)
fi

# 2) User-space CUDA install from a previous run of this script
if [ -z "$NVCC" ] && [ -f "$CUDA_LOCAL/bin/nvcc" ]; then
    NVCC="$CUDA_LOCAL/bin/nvcc"
fi

# 3) Download the CUDA 12.8 runfile and install locally (no sudo needed)
if [ -z "$NVCC" ]; then
    echo "      nvcc not found — installing CUDA 12.8 toolkit to $CUDA_LOCAL ..."
    RUNFILE="/tmp/cuda_12.8_installer.run"
    if [ ! -f "$RUNFILE" ]; then
        wget -q --show-progress \
            "https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run" \
            -O "$RUNFILE"
    fi
    chmod +x "$RUNFILE"
    # --no-drm and --no-man-page avoid needing root; --toolkit-only skips driver install
    sh "$RUNFILE" --silent --toolkit --toolkitpath="$CUDA_LOCAL" \
        --no-opengl-libs --no-drm --no-man-page --override
    NVCC="$CUDA_LOCAL/bin/nvcc"
    echo "      Installed nvcc: $NVCC"
fi

CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
echo "      CUDA_HOME = $CUDA_HOME"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

# ── Build pointops ──────────────────────────────────────────────────────────
cd "$SA3D_DIR/libs/pointops"
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" \
    uv run python setup.py install
cd "$ROOT"

# ── 5. Verify ──────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying installation..."
uv run python - <<'EOF'
import torch
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {torch.version.cuda}")
print(f"  GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'}")
import clip;     print("  CLIP     : ok")
import sam2;     print("  SAM2     : ok")
import open3d;   print(f"  Open3D   : {open3d.__version__}")
import chromadb; print(f"  ChromaDB : {chromadb.__version__}")
try:
    import pointops; print("  pointops : ok")
except ImportError as e:
    print(f"  pointops : MISSING ({e})")
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Extract a ScanNet scene and run the pipeline:"
echo "  python ScanNet/prepare_scene.py --sens /path/to/scene.sens --out_dir ScanNet/RGBD/scene0001_00 --frame_skip 10 --run_pipeline --scene_name scene0001_00"
echo ""
echo "Run the search app:"
echo "  uv run streamlit run app.py"
