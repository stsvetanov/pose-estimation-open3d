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

def run_alignment(model_path, scene_path, voxel_size, init_translation, visualize):
    print(f"[INFO] Loading model: {model_path}")
    model = o3d.io.read_point_cloud(model_path)
    print(f"[INFO] Loading scene: {scene_path}")
    scene = o3d.io.read_point_cloud(scene_path)

    if visualize:
        print("[INFO] Visualizing initial scene and model...")
        o3d.visualization.draw_geometries([scene, model], window_name="Initial Scene and Model")

    if init_translation:
        print(f"[INFO] Applying initial translation: {init_translation}")
        model.translate(init_translation)

    if visualize:
        print("[INFO] Visualizing after model translation...")
        o3d.visualization.draw_geometries([scene, model], window_name="After Model Translation")

    print("[INFO] Downsampling...")
    model_down = model.voxel_down_sample(voxel_size)
    scene_down = scene.voxel_down_sample(voxel_size)

    if visualize:
        print("[INFO] Visualizing downsampled clouds...")
        o3d.visualization.draw_geometries([scene_down, model_down], window_name="After Downsampling")

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
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000)
    )

    print("RANSAC transformation:\n", result_ransac.transformation)
    # model.transform(result_ransac.transformation)

    if visualize:
        print("[INFO] Visualizing after RANSAC...")
        o3d.visualization.draw_geometries([scene, model], window_name="After RANSAC: Model + Scene")

    print("[INFO] Refining alignment with ICP...")
    scene.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    result_icp = o3d.pipelines.registration.registration_icp(
        model, scene, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print("ICP transformation:\n", result_icp.transformation)
    model.transform(result_icp.transformation)

    print("[INFO] Visualizing final alignment...")
    o3d.visualization.draw_geometries([scene, model], window_name="Final Alignment: Model + Scene")

def main():
    parser = argparse.ArgumentParser(description="ICP Alignment using Open3D")
    parser.add_argument('--model', type=str, default="icp_data/sample_object_model.ply", help='Path to model PLY file')
    parser.add_argument('--scene', type=str, default="icp_data/realsense_scene_2.ply", help='Path to scene PLY file')
    parser.add_argument('--voxel', type=float, default=0.002, help='Voxel size for downsampling')
    parser.add_argument('--init_x', type=float, default=0.0, help='Initial X translation of model')
    parser.add_argument('--init_y', type=float, default=0.0, help='Initial Y translation of model')
    parser.add_argument('--init_z', type=float, default=0.0, help='Initial Z translation of model')
    parser.add_argument('--visualize', action='store_true', help='Enable intermediate visualizations')
    args = parser.parse_args()

    init_translation = [args.init_x, args.init_y, args.init_z]
    run_alignment(args.model, args.scene, args.voxel, init_translation, args.visualize)

if __name__ == "__main__":
    main()
