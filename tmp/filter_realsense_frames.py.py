import argparse
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Capture and save denoised RealSense point cloud.")
    parser.add_argument('--output', type=str, default='scene.ply',
                        help='Output filename for the point cloud (PLY format)')
    parser.add_argument('--save_depth_image', action='store_true',
                        help='Save filtered depth image as PNG')
    return parser.parse_args()


def setup_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align


def capture_filtered(pipeline, align):
    # Filters
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)  # <-- keep depth resolution unchanged

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    colorizer = rs.colorizer()

    print("[INFO] Warming up sensor...")
    for _ in range(30):
        pipeline.wait_for_frames()

    # Capture and align
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    for _ in range(10):
        depth = decimation.process(depth)
        depth = depth_to_disparity.process(depth)
        depth = spatial.process(depth)
        depth = temporal.process(depth)
        depth = disparity_to_depth.process(depth)
        depth = hole_filling.process(depth)

    colorized_depth = np.asanyarray(colorizer.colorize(depth).get_data())
    plt.imshow(colorized_depth)
    plt.title("Filtered Depth")
    plt.axis('off')
    plt.show()

    return np.asanyarray(depth.get_data()), np.asanyarray(color_frame.get_data()), colorized_depth



def save_to_ply(depth_image, color_image, output_file):
    depth_o3d = o3d.geometry.Image(depth_image)
    color_o3d = o3d.geometry.Image(color_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=1000.0,
        depth_trunc=3.0
    )

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    o3d.io.write_point_cloud(output_file, pcd)
    print(f"[INFO] Point cloud saved to: {output_file}")
    o3d.visualization.draw_geometries([pcd], window_name="Filtered Point Cloud")


def main():
    args = parse_args()
    pipeline, align = setup_pipeline()

    try:
        print("[INFO] Capturing and filtering frame...")
        depth, color, depth_vis = capture_filtered(pipeline, align)
        save_to_ply(depth, color, args.output)

        if args.save_depth_image:
            from PIL import Image
            Image.fromarray(depth_vis).save("filtered_depth.png")
            print("[INFO] Saved colorized depth to filtered_depth.png")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
