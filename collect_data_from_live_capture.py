
import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import os

# === CONFIGURATION ===
output_dir = "icp_data"
os.makedirs(output_dir, exist_ok=True)
scene_ply_path = os.path.join(output_dir, "realsense_scene_tmp.ply")

# === 1. Setup RealSense Pipeline ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting RealSense camera...")
pipeline.start(config)

# Allow auto-exposure to stabilize
for _ in range(30):
    pipeline.wait_for_frames()

# === 2. Capture One Frame ===
print("Capturing frame...")
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

pipeline.stop()
print("Capture complete.")

# === 3. Convert to Open3D RGBD ===
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

color_o3d = o3d.geometry.Image(color_image)
depth_o3d = o3d.geometry.Image(depth_image)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d,
    depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

# Use D435 intrinsics (or adjust if using different camera)
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)

# === 4. Generate Point Cloud ===
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])  # Flip orientation for Open3D

# === 5. Save and Visualize ===
o3d.io.write_point_cloud(scene_ply_path, pcd)
print(f"Point cloud saved to: {scene_ply_path}")

o3d.visualization.draw_geometries([pcd], window_name="Captured Scene Point Cloud")
