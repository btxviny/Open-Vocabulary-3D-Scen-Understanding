#!/usr/bin/env python3
"""
Test script to demonstrate the camera configuration system.
Run this to see available camera types and their configurations.
"""

from camera_configs import (
    list_available_cameras, 
    get_camera_config, 
    get_camera_description,
    get_pose_file_patterns
)

def main():
    print("=== Camera Configuration System Test ===\n")
    
    # List all available camera types
    available_cameras = list_available_cameras()
    print(f"Available camera types: {available_cameras}\n")
    
    # Show details for each camera type
    for camera_type in available_cameras:
        print(f"--- {camera_type.upper()} ---")
        print(f"Description: {get_camera_description(camera_type)}")
        
        try:
            intrinsics, cam_to_imu = get_camera_config(camera_type)
            print(f"Intrinsics:")
            print(f"  fx: {intrinsics['fx']:.6f}")
            print(f"  fy: {intrinsics['fy']:.6f}")
            print(f"  ppx: {intrinsics['ppx']:.6f}")
            print(f"  ppy: {intrinsics['ppy']:.6f}")
            
            print(f"Camera-to-IMU transformation:")
            print(f"  Shape: {cam_to_imu.shape}")
            print(f"  Is identity: {np.array_equal(cam_to_imu, np.eye(4))}")
            
            print(f"Expected pose file patterns: {get_pose_file_patterns(camera_type)}")
            
        except Exception as e:
            print(f"Error getting config: {e}")
        
        print()
    
    # Test error handling
    print("--- Error Handling Test ---")
    try:
        get_camera_config("nonexistent_camera")
    except ValueError as e:
        print(f"Expected error caught: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    import numpy as np
    main()
