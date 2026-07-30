# 3D Scene Segmentation - Source Code

This directory contains the source code for the 3D Scene Segmentation pipeline, organized into two main modules:

## 📁 Module Structure

### 🔍 **Evaluation Module** (`evaluation/`)
Comprehensive evaluation capabilities for assessing 3D segmentation performance.

**Components:**
- **`evaluator.py`** - Main evaluation orchestrator
- **`metrics.py`** - Segmentation performance metrics (IoU, accuracy, precision/recall)
- **`visualization.py`** - Results visualization and plotting
- **`cli.py`** - Command-line interface for evaluation
- **`preprocess_scannet_scene.py`** - ScanNet scene preprocessing
- **`segment.py`** - Core segmentation evaluation logic

**Usage:**
```bash
# Evaluate multiple scenes using mapping file
python -m evaluation.cli --mapping-file test_mapping.json --output-dir results

# Evaluate single scene
python -m evaluation.cli --single-scene scene0001_00 --pcd-path path/to/pointcloud.npz --scene-dir path/to/scene

# Skip preprocessing if files exist
python -m evaluation.cli --mapping-file test_mapping.json --skip-preprocessing
```

### 🚀 **Hyperparameter Optimization Module** (`hyperparameter_optimization/`)
Advanced hyperparameter search and optimization using MLflow.

**Components:**
- **`search.py`** - Main hyperparameter search engine
- **`analyzer.py`** - Results analysis and visualization



**Usage:**
```bash
# Run hyperparameter search
python -m hyperparameter_optimization.search search --scannet-path ScanNet/RGBD --scans-path ScanNet/scans

# Quick test (5 combinations, 2 scenes)
python -m hyperparameter_optimization.search search --scannet-path ScanNet/RGBD --scans-path ScanNet/scans --num-combinations 5 --max-scenes-per-run 2

# Analyze existing results
python -m hyperparameter_optimization.cli analyze --results-dir hyperparameter_results

# Run single experiment
python -m hyperparameter_optimization.cli experiment --scene-name scene0001_00 --preset accurate

# List available presets
python -m hyperparameter_optimization.cli list-presets
```

## 🛠️ **Installation & Setup**

### Prerequisites
```bash
# Core dependencies
pip install numpy pandas matplotlib seaborn scikit-learn pyyaml

# MLflow for hyperparameter optimization
pip install mlflow

# Optional: Enhanced visualization
pip install plotly kaleido

# Optional: Advanced optimization
pip install optuna hyperopt joblib psutil
```

### Quick Start
```bash
# 1. Install dependencies
cd src
pip install -r requirements_hyperparameter.txt

# 2. Run evaluation
python -m evaluation.cli --mapping-file ../eval/test_mapping.json

# 3. Run hyperparameter search
python -m hyperparameter_optimization.cli search --scannet-path ../ScanNet/RGBD --scans-path ../ScanNet/scans --num-combinations 10
```

## 📊 **Key Features**

### Evaluation Module
- ✅ **Multi-scene evaluation** with batch processing
- ✅ **Comprehensive metrics** (IoU, accuracy, precision/recall, F1)
- ✅ **Automatic visualization** of results
- ✅ **Ground truth comparison** with ScanNet labels
- ✅ **Flexible input** (mapping files or single scenes)
- ✅ **Progress tracking** and detailed logging

### Hyperparameter Optimization Module
- ✅ **MLflow integration** for experiment tracking
- ✅ **Parallel processing** with configurable workers
- ✅ **Parameter space exploration** across 20+ parameters
- ✅ **Automatic evaluation** integration
- ✅ **Results analysis** with statistical insights
- ✅ **Parameter importance** analysis using Random Forest
- ✅ **Predefined presets** for common configurations

## 🔧 **Configuration**

### Parameter Search Space
The system explores parameters across:
- **SAM3D**: voxel size, points per side, IoU thresholds, mask areas
- **CLIP**: model weights, processing logic, thresholds, downsampling
- **Interpolation**: neighbor counts, frame/point strides, depth thresholds

### Example Configuration
```yaml
# See example_config.yaml for complete parameter structure
sam3d:
  voxel_size: 0.05
  points_per_side: 16
  pred_iou_thresh: 0.7

clip:
  weights: "ViT-B/32"
  logic: "patch"
  threshold: 0.01
```

## 📈 **Output Structure**

### Evaluation Results
```
evaluation_output/
├── temp/                    # Temporary preprocessing files
├── results/                 # Processing results
├── individual_results/      # Per-scene results
├── summary/                 # Summary metrics
├── predictions/             # Predicted class IDs
└── plots/                   # Visualization outputs
    ├── scene0001_00/       # Per-scene plots
    └── summary/            # Summary plots
```

### Hyperparameter Search Results
```
hyperparameter_results/
├── logs/                    # Temporary log files
├── configs/                 # Configuration files for each run
├── results/                 # Results for each run
└── results_run_*.json      # Summary results
```

## 🚀 **Advanced Usage**

### Custom Parameter Ranges
Modify the parameter search space in `hyperparameter_optimization/search.py`:
```python
def _define_parameter_space(self) -> Dict[str, List[Any]]:
    return {
        "voxel_size": [0.02, 0.04, 0.06, 0.08],  # Custom range
        "points_per_side": [8, 16, 32, 64],        # Custom range
        # ... other parameters
    }
```

### Parallel Processing
```bash
# Adjust workers based on system capabilities
python -m hyperparameter_optimization.cli search \
    --max-workers 4 \
    --num-combinations 100
```

### Scene Selection
```bash
# Limit scenes for faster testing
python -m hyperparameter_optimization.cli search \
    --max-scenes-per-run 2 \
    --num-combinations 10
```

## 📊 **MLflow Integration**

### View Results
```bash
# Start MLflow UI
mlflow ui

# Navigate to http://localhost:5000
# View experiments, compare runs, and analyze results
```

### Tracked Information
- **Parameters**: All hyperparameters automatically logged
- **Metrics**: mIoU, success rates, run times across experiments
- **Artifacts**: Configuration files, results, and analysis outputs
- **Experiment History**: Complete parameter sets and results

## 🐛 **Troubleshooting**

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Path Errors**: Verify ScanNet directory structure
3. **Memory Issues**: Reduce `max_workers` or `max_scenes_per_run`
4. **Timeout Errors**: Increase timeout values in scripts

### Getting Help
1. Check logs in output directories
2. Verify data paths and permissions
3. Test with single scene first
4. Check MLflow for detailed experiment logs

## 📚 **Documentation**

- **`README_HYPERPARAMETER_SEARCH.md`** - Detailed hyperparameter search guide
- **`example_config.yaml`** - Example configuration file
- **`requirements_hyperparameter.txt`** - Dependencies list

## 🤝 **Contributing**

To extend the system:
1. Add new parameters to parameter search space
2. Implement new evaluation metrics in `evaluation/metrics.py`
3. Create custom visualizations in `evaluation/visualization.py`
4. Add new parameter combinations in `hyperparameter_optimization/search.py`

## 📄 **License**

This source code is part of the 3D Scene Segmentation project. Please refer to the main project license for usage terms.
