import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import os

# === CONFIGURATION ===
output_dir = "icp_data"
os.makedirs(output_dir, exist_ok=True)
fused_pcd_path = os.path.join(output_dir, "realsense_fused_scene.ply")
num_frames = 30  # Number of frames to accumulate

# === RealSense Setup ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting RealSense pipeline...")
pipeline.start(config)

# Warm-up
for _ in range(30):
    pipeline.wait_for_frames()

# Camera intrinsics
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)

# === Accumulate Point Clouds ===
fused_pcd = o3d.geometry.PointCloud()

print(f"Capturing and fusing {num_frames} frames...")
for i in range(num_frames):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    color_o3d = o3d.geometry.Image(color_image)
    depth_o3d = o3d.geometry.Image(depth_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])  # Flip for Open3D

    fused_pcd += pcd

print("Stopping RealSense pipeline...")
pipeline.stop()

# === Downsample and Denoise ===
fused_pcd = fused_pcd.voxel_down_sample(voxel_size=0.002)
fused_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# === Save and Visualize ===
o3d.io.write_point_cloud(fused_pcd_path, fused_pcd)
print(f"Saved fused point cloud to: {fused_pcd_path}")

o3d.visualization.draw_geometries([fused_pcd], window_name="Fused Scene Point Cloud")
