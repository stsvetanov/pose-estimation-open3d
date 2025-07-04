# https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html

import copy
import open3d as o3d
import numpy as np


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    # source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4459,
    #                                   front=[0.9288, -0.2951, -0.2242],
    #                                   lookat=[1.6784, 2.0612, 1.4451],
    #                                   up=[-0.3402, -0.9189, -0.1996])
    o3d.visualization.draw_geometries([source_temp, target_temp])


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


# demo_icp_pcds = o3d.data.DemoICPPointClouds()
# source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
source = o3d.io.read_point_cloud("icp_data/sample_model.ply")
# target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
target = o3d.io.read_point_cloud("icp_data/sample_model.ply")
target.translate([0.02, 0.02, 0])
target.paint_uniform_color([1, 0, 0])

voxel_size = 0.005
print("[INFO] Downsampling...")
source = source.voxel_down_sample(voxel_size)
target = target.voxel_down_sample(voxel_size)

threshold = 0.02
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
draw_registration_result(source, target, trans_init)

print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
print(evaluation)

print("[INFO] Estimating normals and computing FPFH...")

source_fpfh = compute_fpfh(source, voxel_size)
target_fpfh = compute_fpfh(target, voxel_size)

distance_threshold = voxel_size * 1.5
print("[INFO] Running RANSAC...")
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    target, source, target_fpfh, source_fpfh, True,
    distance_threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
    [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ],
    o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000)
)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)

reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)


print("Apply point-to-plane ICP")
reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
draw_registration_result(source, target, reg_p2l.transformation)



