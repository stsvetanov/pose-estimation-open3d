#%% 1. Libraries import

import numpy as np
import open3d as o3d

#%% 2. Generate a synthetic point cloud

num_points = 1000
plane_points = np.random.randn(num_points // 2, 3)
plane_points[:, 2] = 0.1 * np.random.randn(num_points // 2) + 1  # Add some noise to z-coordinate

sphere_center = np.array([2, 2, 2])
sphere_radius = 1
theta = np.random.uniform(0, 2*np.pi, num_points // 2)
phi = np.random.uniform(0, np.pi, num_points // 2)

x = sphere_center[0] + sphere_radius * np.sin(phi) * np.cos(theta)
y = sphere_center[1] + sphere_radius * np.sin(phi) * np.sin(theta)
z = sphere_center[2] + sphere_radius * np.cos(phi)

sphere_points = np.column_stack((x, y, z))
sphere_points += 0.05 * np.random.randn(*sphere_points.shape)  # Add some noise

points = np.vstack((plane_points, sphere_points))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])