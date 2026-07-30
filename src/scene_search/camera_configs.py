"""
Camera configuration file for different sensor setups.
This file reads camera configurations from config.yaml.
"""

import numpy as np
import yaml
from pathlib import Path

def load_camera_configs():
    """Load camera configurations from config.yaml."""
    config_path = Path('config.yaml')
    if not config_path.exists():
        # Try to find config.yaml in parent directories
        config_path = Path(__file__).parent.parent / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found. Looked in: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'cameras' not in config:
        raise ValueError("No 'cameras' section found in config.yaml")
    
    return config['cameras']


def get_camera_config(camera_type):
    """Get the appropriate camera configuration based on camera type."""
    camera_configs = load_camera_configs()
    
    if camera_type not in camera_configs:
        available_types = list(camera_configs.keys())
        raise ValueError(f"Unknown camera type: {camera_type}. Available types: {available_types}")
    
    config = camera_configs[camera_type]
    
    # Convert YAML list to numpy array for cam_to_imu
    cam_to_imu = np.array(config['cam_to_imu'])
    
    return config['intrinsics'], cam_to_imu


def list_available_cameras():
    """List all available camera configurations."""
    try:
        camera_configs = load_camera_configs()
        return list(camera_configs.keys())
    except Exception as e:
        print(f"Warning: Could not load camera configs: {e}")
        return []


def get_camera_description(camera_type):
    """Get description of a camera type."""
    try:
        camera_configs = load_camera_configs()
        if camera_type not in camera_configs:
            return f"Unknown camera type: {camera_type}"
        
        return camera_configs[camera_type].get('description', 'No description available')
    except Exception as e:
        return f"Error loading camera description: {e}"


def get_pose_file_patterns(camera_type):
    """Get expected pose file patterns for a camera type."""
    try:
        camera_configs = load_camera_configs()
        if camera_type not in camera_configs:
            return []
        
        return camera_configs[camera_type].get('pose_file_patterns', [])
    except Exception as e:
        print(f"Warning: Could not load pose file patterns: {e}")
        return []
