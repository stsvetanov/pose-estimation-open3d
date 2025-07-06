import argparse
import numpy as np
import pyrealsense2 as rs
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description="Capture and save denoised RealSense point cloud.")
    parser.add_argument('--mode', type=str, choices=['average', 'filtered'], default='filtered',
                        help='Select noise reduction mode: "average" (manual) or "filtered" (SDK filters)')
    parser.add_argument('--frames', type=int, default=30,
                        help='Number of frames to capture for averaging or filtering')
    parser.add_argument('--output', type=str, default='scene.ply',
                        help='Output filename for the point cloud (PLY format)')
    return parser.parse_args()


def setup_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def get_filters():
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    return [spatial, temporal, hole_filling]


def capture_average(pipeline, frame_count):
    depth_frames = []
    color_frame = None

    for _ in range(frame_count):
        frames = pipeline.wait_for_frames()
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        depth_frames.append(depth)

        # Save the last color frame for color mapping
        color_frame = frames.get_color_frame()

    # Average while ignoring zeros (invalid depth)
    stack = np.stack(depth_frames).astype(np.float32)
    stack[stack == 0] = np.nan
    avg_depth = np.nanmean(stack, axis=0).astype(np.uint16)

    return avg_depth, np.asanyarray(color_frame.get_data())


def capture_filtered(pipeline, frame_count, filters):
    depth_frame = None
    color_frame = None

    for _ in range(frame_count):
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        for f in filters:
            depth = f.process(depth)

    return np.asanyarray(depth.get_data()), np.asanyarray(color_frame.get_data())


def save_to_ply(depth_image, color_image, output_file):
    # Create Open3D images
    depth_o3d = o3d.geometry.Image(depth_image)
    color_o3d = o3d.geometry.Image(color_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=1000.0,
        depth_trunc=3.0
    )

    # Create point cloud
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    # Flip it for proper orientation
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    o3d.io.write_point_cloud(output_file, pcd)
    print(f"[INFO] Point cloud saved to: {output_file}")


def main():
    args = parse_args()
    pipeline = setup_pipeline()
    filters = get_filters() if args.mode == 'filtered' else None

    try:
        print(f"[INFO] Capturing {args.frames} frames using mode: {args.mode}")
        if args.mode == 'average':
            depth, color = capture_average(pipeline, args.frames)
        else:
            depth, color = capture_filtered(pipeline, args.frames, filters)

        save_to_ply(depth, color, args.output)

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
