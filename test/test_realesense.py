import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()

# Create a configuration object
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    # Start streaming
    pipeline.start(config)

    # Wait for frames
    frames = pipeline.wait_for_frames()

    # Get the depth frame
    depth_frame = frames.get_depth_frame()

    if not depth_frame:
        raise RuntimeError("Could not get depth frame.")

    # Get the intrinsics of the depth stream
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    # Extract and print intrinsic parameters
    print("Depth Stream Intrinsic Parameters:")
    print(f"  Width: {intrinsics.width}")
    print(f"  Height: {intrinsics.height}")
    print(f"  fx: {intrinsics.fx}")
    print(f"  fy: {intrinsics.fy}")
    print(f"  ppx (cx): {intrinsics.ppx}")
    print(f"  ppy (cy): {intrinsics.ppy}")
    print(f"  Distortion Model: {intrinsics.model}")
    print(f"  Distortion Coefficients: {intrinsics.coeffs}")

finally:
    # Stop the pipeline
    pipeline.stop()
