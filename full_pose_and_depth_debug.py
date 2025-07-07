import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import os

# === Setup RealSense pipeline ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Warm-up
print("[INFO] Warming up RealSense sensor...")
for _ in range(30):
    pipeline.wait_for_frames()

# === Retrieve Camera Intrinsics ===
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
depth_intr = depth_stream.get_intrinsics()
color_intr = color_stream.get_intrinsics()

print("Color Intrinsics:")
print(f"  width={color_intr.width}, height={color_intr.height}")
print(f"  fx={color_intr.fx}, fy={color_intr.fy}, cx={color_intr.ppx}, cy={color_intr.ppy}")

# Build Open3D PinholeCameraIntrinsic from color camera
color_o3d_intr = o3d.camera.PinholeCameraIntrinsic(
    width=color_intr.width,
    height=color_intr.height,
    fx=color_intr.fx,
    fy=color_intr.fy,
    cx=color_intr.ppx,
    cy=color_intr.ppy
)

print("Streaming started. Press 'q' in OpenCV window to quit.")
while True:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # === Build point cloud from RGBD ===
    color_raw = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_raw = o3d.geometry.Image(depth_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,
        convert_rgb_to_intensity=False,
        depth_scale=1000.0,
        depth_trunc=3.0
    )

    scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, color_o3d_intr)
    scene_pcd.transform([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])

    # === Show extracted point cloud ===
    o3d.visualization.draw_geometries([scene_pcd], window_name="Scene Point Cloud")

    output_dir = "icp_data"
    os.makedirs(output_dir, exist_ok=True)
    scene_ply_path = os.path.join(output_dir, "realsense_scene_3.ply")
    o3d.io.write_point_cloud(scene_ply_path, scene_pcd)
    print(f"Point cloud saved to: {scene_ply_path}")

    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
