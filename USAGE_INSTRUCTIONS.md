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
