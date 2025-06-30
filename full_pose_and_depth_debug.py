
import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import json
import os
from datetime import datetime

def project_points(points_3d, intrinsic):
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()
    points_2d = []
    for x, y, z in points_3d:
        if z <= 1e-6:
            points_2d.append((0, 0))
            continue
        u = int((x * fx) / z + cx)
        v = int((y * fy) / z + cy)
        points_2d.append((u, v))
    return points_2d

def save_pose_matrix(matrix, out_dir="poses"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"pose_{ts}.json")
    with open(path, "w") as f:
        json.dump(matrix.tolist(), f, indent=4)
    print(f"Saved pose to {path}")

# === Setup RealSense pipeline ===
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# === Camera Intrinsics ===
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=617, fy=617, cx=320, cy=240
)

# === Load and sample mesh ===
mesh_original = o3d.io.read_triangle_mesh("data/3D-models/Plate2.stl")
mesh_original.compute_vertex_normals()
model_pcd = mesh_original.sample_points_uniformly(number_of_points=2000)

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

    # === Display depth image ===
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imshow("Depth Image", depth_colormap)

    # === Build point cloud from RGBD ===
    color_raw = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_raw = o3d.geometry.Image(depth_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,
        convert_rgb_to_intensity=False,
        depth_scale=1000.0,
        depth_trunc=3.0
    )

    scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    scene_pcd.transform([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])

    # === Show extracted point cloud ===
    o3d.visualization.draw_geometries([scene_pcd], window_name="Scene Point Cloud")

    output_dir = "icp_data"
    os.makedirs(output_dir, exist_ok=True)
    scene_ply_path = os.path.join(output_dir, "realsense_scene_2.ply")
    o3d.io.write_point_cloud(scene_ply_path, scene_pcd)
    print(f"Point cloud saved to: {scene_ply_path}")

    # === Automatic detection via alignment (RANSAC/ICP placeholder) ===
    # Future improvement: add segmentation or bounding box detection
    reg_result = o3d.pipelines.registration.registration_icp(
        source=model_pcd,
        target=scene_pcd,
        max_correspondence_distance=0.01,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    transformation = reg_result.transformation
    R = transformation[:3, :3]
    t = transformation[:3, 3]

    # === Save pose to file ===
    save_pose_matrix(transformation)

    # === Overlay projection of 3D model on RGB ===
    scale = 0.05
    axes = np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, scale]], dtype=np.float32)
    axes_transformed = np.dot(R, axes.T).T + t
    points_2d = project_points(axes_transformed, intrinsics)
    origin_2d = project_points([t], intrinsics)[0]

    overlay = color_image.copy()
    try:
        cv2.arrowedLine(overlay, origin_2d, points_2d[0], (0, 0, 255), 3)
        cv2.arrowedLine(overlay, origin_2d, points_2d[1], (0, 255, 0), 3)
        cv2.arrowedLine(overlay, origin_2d, points_2d[2], (255, 0, 0), 3)
    except:
        pass

    mesh = mesh_original.transform(transformation.copy())
    mesh.compute_vertex_normals()
    mesh_pcd = mesh.sample_points_uniformly(3000)
    pcd_points = np.asarray(mesh_pcd.points)
    pcd_2d = project_points(pcd_points, intrinsics)

    for pt in pcd_2d:
        u, v = pt
        if 0 <= u < 640 and 0 <= v < 480:
            cv2.circle(overlay, (u, v), 2, (0, 255, 255), -1)

    cv2.imshow("RGB + Pose Overlay", overlay)

    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
