"""
Script to create a video from RGB frames in the same directory structure as dense_pointcloud.py

Example usage:
python create_video_from_frames.py --base_dir ../data_from_runs/data_1744645809_90342760/ --output_video output_video.mp4 --fps 30
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import tqdm


def create_video_from_frames(base_dir, output_video, fps=30, frame_size=None):
    """
    Create a video from RGB frames in subdirectories.
    
    Args:
        base_dir: Base directory containing frame subdirectories
        output_video: Output video file path
        fps: Frames per second for the output video
        frame_size: Tuple (width, height) to resize frames. If None, uses original size.
    """
    
    # Get all frame directories
    frame_names = [x for x in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, x))]
    
    if not frame_names:
        raise ValueError(f"No frame directories found in {base_dir}")
    
    # Sort frame directories numerically to ensure proper order
    # This handles cases like frame_1, frame_2, ..., frame_10, frame_11, etc.
    def natural_sort_key(name):
        """Extract numbers from directory name for natural sorting."""
        import re
        # Split the name into text and number parts
        parts = re.split(r'(\d+)', name)
        result = []
        for part in parts:
            if part.isdigit():
                result.append(int(part))
            else:
                result.append(part)
        return result
    
    frame_names.sort(key=natural_sort_key)
    frames_dir = [os.path.join(base_dir, name) for name in frame_names]
    
    print(f"Found {len(frames_dir)} frame directories")
    print(f"First 10 frame directories: {frame_names[:10]}")
    if len(frame_names) > 10:
        print(f"Last 10 frame directories: {frame_names[-10:]}")
    
    # Initialize video writer
    video_writer = None
    frame_count = 0
    
    # Process frames
    for frame_dir in tqdm.tqdm(frames_dir, desc="Processing frames for video"):
        # Load RGB image
        image_path = os.path.join(frame_dir, 'rgb_image.png')
        if not os.path.exists(image_path):
            print(f"Warning: RGB image not found in {frame_dir}, skipping...")
            continue
            
        # Load and convert image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Initialize video writer on first frame
        if video_writer is None:
            height, width = image_bgr.shape[:2]
            
            # Resize if frame_size is specified
            if frame_size is not None:
                width, height = frame_size
                image_bgr = cv2.resize(image_bgr, (width, height))
            
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            print(f"Video writer initialized: {width}x{height} at {fps} FPS")
        
        # Resize frame if needed
        if frame_size is not None:
            image_bgr = cv2.resize(image_bgr, frame_size)
        
        # Write frame to video
        video_writer.write(image_bgr)
        frame_count += 1
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {output_video} with {frame_count} frames")
    else:
        print("No valid frames found to create video")


def main():
    parser = argparse.ArgumentParser(description="Create a video from RGB frames in subdirectories.")
    parser.add_argument('--base_dir', type=str, required=True, 
                       help="Path to base directory with frame subfolders.")
    parser.add_argument('--output_video', type=str, required=True, 
                       help="Path to save the output video file.")
    parser.add_argument('--fps', type=int, default=30, 
                       help="Frames per second for the output video (default: 30)")
    parser.add_argument('--width', type=int, default=None, 
                       help="Width to resize frames (default: original width)")
    parser.add_argument('--height', type=int, default=None, 
                       help="Height to resize frames (default: original height)")
    
    args = parser.parse_args()
    
    # Validate base directory
    if not os.path.exists(args.base_dir):
        raise ValueError(f"Base directory {args.base_dir} does not exist")
    
    # Set frame size if both width and height are provided
    frame_size = None
    if args.width is not None and args.height is not None:
        frame_size = (args.width, args.height)
    elif args.width is not None or args.height is not None:
        print("Warning: Both width and height must be specified to resize frames. Using original size.")
    
    # Create video
    create_video_from_frames(
        base_dir=args.base_dir,
        output_video=args.output_video,
        fps=args.fps,
        frame_size=frame_size
    )


if __name__ == "__main__":
    main()
