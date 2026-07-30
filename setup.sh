#!/usr/bin/env bash
# Setup script for Open 3D Scene Understanding
# Uses uv for Python/package management
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

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
echo "[2/5] Creating venv and installing dependencies (including PyTorch cu128)..."
cd "$ROOT"
uv sync
echo "      Done."

# ── 3. SAM2 checkpoints ────────────────────────────────────────────────────
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
cd "$SA3D_DIR/libs/pointops"
# Build with the uv-managed Python so CUDA arch flags are correct
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0" uv run python setup.py install
cd "$ROOT"

# ── 5. Verify ──────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying installation..."
uv run python - <<'EOF'
import torch
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {torch.version.cuda}")
print(f"  GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'}")
import clip; print(f"  CLIP     : ok")
import sam2;  print(f"  SAM2     : ok")
import open3d; print(f"  Open3D   : {open3d.__version__}")
import chromadb; print(f"  ChromaDB : {chromadb.__version__}")
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run the pipeline:"
echo "  cd src && uv run python pipeline.py --run_path <scene_dir> --scene_name <name>"
echo ""
echo "Run the app:"
echo "  uv run streamlit run app.py"
