import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Check if the camera is a D400 series camera
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# align = rs.align(rs.stream.color)

# Getting the depth sensor's intrinsic properties
depth_profile = profile.get_stream(rs.stream.depth)
depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

# Getting the color sensor's intrinsic properties
color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

pipeline.stop()

# Print the intrinsic parameters
print("## Depth Intrinsics:")
print(f"  Width: {depth_intrinsics.width}")
print(f"  Height: {depth_intrinsics.height}")
print(f"  PPX: {depth_intrinsics.ppx}")
print(f"  PPY: {depth_intrinsics.ppy}")
print(f"  FX: {depth_intrinsics.fx}")
print(f"  FY: {depth_intrinsics.fy}")
print(f"  Distortion Model: {depth_intrinsics.model}")
print(f"  Coefficients: {depth_intrinsics.coeffs}\n")

print("## Color Intrinsics:")
print(f"  Width: {color_intrinsics.width}")
print(f"  Height: {color_intrinsics.height}")
print(f"  PPX: {color_intrinsics.ppx}")
print(f"  PPY: {color_intrinsics.ppy}")
print(f"  FX: {color_intrinsics.fx}")
print(f"  FY: {color_intrinsics.fy}")
print(f"  Distortion Model: {color_intrinsics.model}")
print(f"  Coefficients: {color_intrinsics.coeffs}")