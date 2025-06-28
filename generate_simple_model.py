import open3d as o3d
import numpy as np
import os

# === CONFIGURATION ===
output_dir = "icp_data"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "sample_object_model.ply")

# === 1. Create Rectangular Parallelepiped (60x40x8 mm) ===
width = 0.06   # 60 mm
height = 0.04  # 40 mm
depth = 0.008  # 8 mm

box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
box.compute_vertex_normals()
box.paint_uniform_color([0.2, 0.6, 0.8])  # Light blue

# === 2. Convert to Point Cloud ===
model_pcd = box.sample_points_uniformly(number_of_points=1500)

# === 3. Save to .ply file ===
o3d.io.write_point_cloud(model_path, model_pcd)
print(f"Model saved at: {model_path}")

# === 4. Visualize the result ===
o3d.visualization.draw_geometries([model_pcd], window_name="Rectangular Object (60x40x8 mm)")
