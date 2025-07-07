import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d


def generate_dummy_mask(image_shape, radius=100):
    """Creates a circular binary mask centered in the image."""
    h, w = image_shape[:2]
    center = (w // 2, h // 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, color=255, thickness=-1)
    return mask


def main():
    # ---------- RealSense Setup ----------
    pipeline = rs.pipeline()
    config = rs.config()

    # High-res color for segmentation, depth max res for accuracy
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    print("[INFO] Warming up sensor...")
    for _ in range(30):
        pipeline.wait_for_frames()

    # ---------- Get Intrinsics ----------
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_profile.get_intrinsics()

    print(f"[INFO] Using Color Intrinsics ({color_intr.width}x{color_intr.height})")
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        width=color_intr.width,
        height=color_intr.height,
        fx=color_intr.fx,
        fy=color_intr.fy,
        cx=color_intr.ppx,
        cy=color_intr.ppy
    )

    # ---------- Capture One Frame ----------
    print("[INFO] Capturing frame...")
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not depth_frame or not color_frame:
        raise RuntimeError("Failed to get frames from RealSense.")

    color_image = np.asanyarray(color_frame.get_data())       # 1920x1080
    depth_image = np.asanyarray(depth_frame.get_data())       # aligned 1920x1080

    print(f"[INFO] color_image.shape = {color_image.shape}, depth_image.shape = {depth_image.shape}")

    # ---------- Generate Segmentation Mask ----------
    mask = generate_dummy_mask(color_image.shape, radius=100)

    # ---------- Show Mask Overlay ----------
    overlay = cv2.addWeighted(color_image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow("RGB + Mask", overlay)
    cv2.waitKey(1)

    # ---------- Create RGBD Image for Open3D ----------
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,
        convert_rgb_to_intensity=False,
        depth_trunc=3.0
    )

    full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    full_pcd.transform([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])

    print(f"[INFO] Full scene point cloud has {np.asarray(full_pcd.points).shape[0]} points.")
    o3d.io.write_point_cloud("scene_full.ply", full_pcd)
    o3d.visualization.draw_geometries([full_pcd], window_name="Full Scene Point Cloud")

    # ---------- Apply Mask to Extract 3D Points ----------
    points = []
    colors = []

    for v in range(mask.shape[0]):
        for u in range(mask.shape[1]):
            if mask[v, u] > 0:
                depth = depth_frame.get_distance(u, v)
                if depth == 0:
                    continue
                xyz = rs.rs2_deproject_pixel_to_point(color_intr, [u, v], depth)
                points.append(xyz)
                colors.append(color_image[v, u] / 255.0)

    points = np.array(points)
    colors = np.array(colors)

    if len(points) == 0:
        raise RuntimeError("No valid 3D points extracted from mask and depth.")

    print(f"[INFO] Extracted {len(points)} masked 3D points.")

    # ---------- Create Cropped Point Cloud ----------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    output_path = "masked_pointcloud.ply"
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[INFO] Saved masked point cloud to: {output_path}")

    pcd.transform([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd], window_name="Masked Point Cloud")

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
