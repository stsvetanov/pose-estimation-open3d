import open3d as o3d
import numpy as np

# === FILE PATHS ===
scene_path = "icp_data/realsense_scene_2.ply"
model_path = "icp_data/sample_object_model.ply"
# model_path = "icp_data/Plate4_model.ply"

# === 1. Load Scene and Model ===
scene = o3d.io.read_point_cloud(scene_path)
model = o3d.io.read_point_cloud(model_path)

# Optional: visualize inputs
o3d.visualization.draw_geometries([scene], window_name="Captured Scene")
o3d.visualization.draw_geometries([model], window_name="3D Model")

# === 2. Downsample and Estimate Normals ===
voxel_size = 0.005
scene_down = scene.voxel_down_sample(voxel_size)
model_down = model.voxel_down_sample(voxel_size)
scene_down.estimate_normals()
model_down.estimate_normals()

# === 3. Compute FPFH Features ===
def compute_fpfh(pcd, voxel):
    radius_normal = voxel * 2
    radius_feature = voxel * 5
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

scene_fpfh = compute_fpfh(scene_down, voxel_size)
model_fpfh = compute_fpfh(model_down, voxel_size)

# === 4. Global Registration with RANSAC ===
distance_threshold = voxel_size * 1.5
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

# === 5. ICP Refinement ===
# Before ICP, compute normals on the full-resolution target (scene)
scene.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
)

o3d.visualization.draw_geometries([scene, model], window_name="After RANSAC: Model + Scene")

result_icp = o3d.pipelines.registration.registration_icp(
    model, scene,
    distance_threshold,
    result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print("ICP transformation:\n", result_icp.transformation)

# === 6. Visualize Aligned Model ===
model.transform(result_icp.transformation)
o3d.visualization.draw_geometries([scene, model], window_name="Final Alignment: Model + Scene")
