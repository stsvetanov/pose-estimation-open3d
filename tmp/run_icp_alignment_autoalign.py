import open3d as o3d
import numpy as np
import argparse
import os

def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

def auto_center_and_scale(model, scene):
    model_bounds = model.get_axis_aligned_bounding_box()
    scene_bounds = scene.get_axis_aligned_bounding_box()

    model_center = model_bounds.get_center()
    scene_center = scene_bounds.get_center()

    model_extent = model_bounds.get_extent().max()
    scene_extent = scene_bounds.get_extent().max()

    scale_ratio = scene_extent / model_extent
    print(f"[INFO] Auto-scaling model by factor: {scale_ratio:.3f}")
    model.scale(scale_ratio, center=model_center)

    shift = scene_center - model_center
    print(f"[INFO] Auto-translating model by: {shift}")
    model.translate(shift)

    return model

def run_alignment(model_path, scene_path, voxel_size, auto_align):
    print(f"[INFO] Loading model: {model_path}")
    model = o3d.io.read_point_cloud(model_path)
    print(f"[INFO] Loading scene: {scene_path}")
    scene = o3d.io.read_point_cloud(scene_path)

    if auto_align:
        print("[INFO] Auto-aligning model to scene bounding box...")
        model = auto_center_and_scale(model, scene)

    print("[INFO] Visualizing initial scene and model...")
    o3d.visualization.draw_geometries(
        [scene.paint_uniform_color([0.6, 0.6, 0.6]),
         model.paint_uniform_color([0.2, 0.8, 0.2])],
        window_name="Initial Position: Model + Scene"
    )

    print("[INFO] Downsampling...")
    model_down = model.voxel_down_sample(voxel_size)
    scene_down = scene.voxel_down_sample(voxel_size)

    print("[INFO] Visualizing downsampled clouds...")
    o3d.visualization.draw_geometries(
        [scene_down.paint_uniform_color([0.6, 0.6, 0.6]),
         model_down.paint_uniform_color([0.2, 0.8, 0.2])],
        window_name="Downsampled: Model + Scene"
    )

    print("[INFO] Estimating normals and computing FPFH...")
    model_fpfh = compute_fpfh(model_down, voxel_size)
    scene_fpfh = compute_fpfh(scene_down, voxel_size)

    distance_threshold = voxel_size * 1.5
    print("[INFO] Running RANSAC...")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down, scene_down, model_fpfh, scene_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    print("RANSAC transformation:\n", result_ransac.transformation)

    print("[INFO] Visualizing after RANSAC...")
    model.transform(result_ransac.transformation)
    o3d.visualization.draw_geometries(
        [scene.paint_uniform_color([0.6, 0.6, 0.6]),
         model.paint_uniform_color([0.1, 0.5, 1.0])],
        window_name="After RANSAC: Model + Scene"
    )

    print("[INFO] Refining alignment with ICP...")
    scene.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    result_icp = o3d.pipelines.registration.registration_icp(
        model, scene, distance_threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print("ICP transformation:\n", result_icp.transformation)
    model.transform(result_icp.transformation)

    print("[INFO] Final visualization...")
    o3d.visualization.draw_geometries(
        [scene.paint_uniform_color([0.6, 0.6, 0.6]),
         model.paint_uniform_color([0.9, 0.3, 0.1])],
        window_name="Final Alignment: Model + Scene"
    )

def main():
    parser = argparse.ArgumentParser(description="ICP Alignment using Open3D")
    parser.add_argument('--model', type=str, default="icp_data/sample_object_model.ply", help='Path to model PLY file')
    parser.add_argument('--scene', type=str, default="icp_data/realsense_scene_2.ply", help='Path to scene PLY file')
    parser.add_argument('--voxel', type=float, default=0.005, help='Voxel size for downsampling')
    parser.add_argument('--auto_align', action='store_true', help='Auto-center and scale model to scene bounds')
    args = parser.parse_args()

    run_alignment(args.model, args.scene, args.voxel, args.auto_align)

if __name__ == "__main__":
    main()
