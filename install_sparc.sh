#!/bin/bash

# SPARC Installation Script
# This script automates the complete installation of SPARC: 3D Scene Understanding
# Author: SPARC Team
# Version: 1.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU and CUDA
check_system() {
    print_status "Checking system requirements..."
    
    # Check if conda is installed
    if ! command_exists conda; then
        print_error "Conda is not installed. Please install Anaconda or Miniconda first."
        exit 1
    fi
    
    # Check if nvidia-smi is available
    if ! command_exists nvidia-smi; then
        print_warning "nvidia-smi not found. CUDA support may not be available."
    else
        print_status "GPU Information:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    fi
    
    # Check if git is installed
    if ! command_exists git; then
        print_error "Git is not installed. Please install git first."
        exit 1
    fi
    
    # Check if wget is installed
    if ! command_exists wget; then
        print_error "wget is not installed. Please install wget first."
        exit 1
    fi
    
    print_success "System requirements check completed."
}

# Function to create conda environment
setup_conda_env() {
    print_status "Setting up conda environment..."
    
    # Check if environment already exists
    if conda env list | grep -q "sparc"; then
        print_warning "Environment 'sparc' already exists. Removing it first..."
        conda env remove -n sparc -y
    fi
    
    # Create new environment
    print_status "Creating conda environment 'sparc' with Python 3.10..."
    conda create -n sparc python=3.10 -y
    
    print_success "Conda environment created successfully."
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Activate environment and install packages
    print_status "Activating conda environment and installing CUDA..."
    conda run -n sparc conda install -c nvidia cuda=12.1 -y
    
    print_status "Installing PyTorch with CUDA 12.1 support..."
    conda run -n sparc pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    print_status "Installing plyfile..."
    conda run -n sparc conda install plyfile -c conda-forge -y
    
    print_status "Installing ninja for faster builds..."
    conda run -n sparc conda install ninja -c conda-forge -y
    
    print_status "Installing Python requirements..."
    conda run -n sparc pip install -r requirements.txt
    
    print_success "Dependencies installed successfully."
}

# Function to setup environment variables
setup_env_vars() {
    print_status "Setting up environment variables..."
    
    # Set CUDA environment variables in the conda environment
    print_status "Configuring CUDA environment variables..."
    conda run -n sparc bash -c 'echo "export CUDA_HOME=\$CONDA_PREFIX" >> ~/.bashrc'
    conda run -n sparc bash -c 'echo "export CUDA_PATH=\$CONDA_PREFIX" >> ~/.bashrc'
    conda run -n sparc bash -c 'echo "export PATH=\"\$CUDA_HOME/bin:\$PATH\"" >> ~/.bashrc'
    conda run -n sparc bash -c 'echo "export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib:\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc'
    
    print_success "Environment variables configured for conda environment."
}

# Function to setup SAM2
setup_sam2() {
    print_status "Setting up SAM2..."
    
    # Check if SAM2 directory exists
    if [ -d "sam2" ]; then
        print_warning "SAM2 directory already exists. Removing it first..."
        rm -rf sam2
    fi
    
    print_status "Cloning SAM2 repository..."
    git clone https://github.com/facebookresearch/sam2.git
    
    print_status "Installing SAM2..."
    cd sam2
    conda run -n sparc pip install -e .
    
    print_status "Downloading SAM2 weights..."
    mkdir -p checkpoints
    cd checkpoints
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
    
    cd ../..
    print_success "SAM2 setup completed."
}

# Function to setup SAM3D
setup_sam3d() {
    print_status "Setting up SAM3D..."
    
    # Check if SegmentAnything3D directory exists
    if [ -d "SegmentAnything3D" ]; then
        print_warning "SegmentAnything3D directory already exists. Removing it first..."
        rm -rf SegmentAnything3D
    fi
    
    print_status "Cloning SegmentAnything3D repository..."
    git clone https://github.com/Pointcept/SegmentAnything3D.git
    
    print_status "Installing pointops..."
    cd SegmentAnything3D/libs/pointops
    
    # Install pointops
    conda run -n sparc python setup.py install
    
    # Install with specific CUDA architectures
    print_status "Installing pointops with CUDA architecture support..."
    conda run -n sparc bash -c 'export TORCH_CUDA_ARCH_LIST="7.5 8.0" && python setup.py install'
    
    cd ../../..
    print_success "SAM3D setup completed."
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test PyTorch CUDA availability
    print_status "Testing PyTorch CUDA support..."
    conda run -n sparc python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA not available')
"
    
    # Test SAM2 import
    print_status "Testing SAM2 import..."
    conda run -n sparc python -c "
try:
    import sam2
    print('SAM2 imported successfully')
except ImportError as e:
    print(f'SAM2 import failed: {e}')
"
    
    # Test pointops import
    print_status "Testing pointops import..."
    conda run -n sparc python -c "
try:
    import pointops
    print('pointops imported successfully')
except ImportError as e:
    print(f'pointops import failed: {e}')
"
    
    # Test Open3D import
    print_status "Testing Open3D import..."
    conda run -n sparc python -c "
try:
    import open3d as o3d
    print('Open3D imported successfully')
    print(f'Open3D version: {o3d.__version__}')
except ImportError as e:
    print(f'Open3D import failed: {e}')
except Exception as e:
    print(f'Open3D import error: {e}')
"
    
    # Test Streamlit app imports
    print_status "Testing Streamlit app imports..."
    conda run -n sparc python -c "
try:
    import streamlit as st
    import matplotlib.pyplot as plt
    import open3d as o3d
    import numpy as np
    import torch
    import clip
    print('All Streamlit app imports successful')
except ImportError as e:
    print(f'Streamlit app import failed: {e}')
except Exception as e:
    print(f'Streamlit app import error: {e}')
"
    
    print_success "Installation verification completed."
}

# Function to create usage instructions
create_usage_instructions() {
    print_status "Creating usage instructions..."
    
    cat > USAGE_INSTRUCTIONS.md << 'EOF'
# SPARC Usage Instructions

## Quick Start

1. **Activate the environment:**
   ```bash
   conda activate sparc
   ```

2. **Run the pipeline:**
   ```bash
   cd src
   python pipeline.py --run_path /path/to/data_directory --scene_name name_of_scene
   ```

3. **Run the app:**
   ```bash
   cd ..
   streamlit run app.py
   ```

## Data Format

Prepare your data directory with RGB-D frames and camera poses:
```
data_directory/
├── frame_0001/
│   ├── rgb_image.png
│   ├── depth_image.png
│   └── extrinsic_matrix.npy
├── frame_0002/
...
```

## Troubleshooting

- If you encounter CUDA issues, make sure to activate the sparc environment with `conda activate sparc`
- For GPU memory issues, try reducing batch sizes in the configuration
- Check that all dependencies are properly installed by running the verification script

## Environment Variables

The following environment variables are automatically set when you activate the sparc conda environment:
- CUDA_HOME
- CUDA_PATH
- PATH (with CUDA binaries)
- LD_LIBRARY_PATH (with CUDA libraries)
EOF
    
    print_success "Usage instructions created: USAGE_INSTRUCTIONS.md"
}

# Main installation function
main() {
    print_status "Starting SPARC installation..."
    print_status "This may take several minutes depending on your internet connection."
    
    # Check system requirements
    check_system
    
    # Setup conda environment
    setup_conda_env
    
    # Install dependencies
    install_dependencies
    
    # Setup environment variables
    setup_env_vars
    
    # Setup SAM2
    setup_sam2
    
    # Setup SAM3D
    setup_sam3d
    
    # Verify installation
    verify_installation
    
    # Create usage instructions
    create_usage_instructions
    
    print_success "SPARC installation completed successfully!"
    print_status "To start using SPARC:"
    print_status "1. Run: conda activate sparc"
    print_status "2. Follow the instructions in USAGE_INSTRUCTIONS.md"
    print_status ""
    print_status "Installation log saved to: sparc_installation.log"
}

# Run main function and log output
main 2>&1 | tee sparc_installation.log

print_success "Installation completed! Check sparc_installation.log for detailed output."
