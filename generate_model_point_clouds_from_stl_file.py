import open3d as o3d
import numpy as np
import os

# === CONFIGURATION ===
output_dir = "icp_data"
os.makedirs(output_dir, exist_ok=True)
model_file = "data/3D-models/Plate4.stl"
output_ply = os.path.join(output_dir, "Plate4_model.ply")

# === 1. Load STL Mesh ===
mesh = o3d.io.read_triangle_mesh(model_file)

if mesh.is_empty():
    print(f"Error: Could not load mesh from {model_file}")
    exit()

# Optional: Normalize the scale or center if needed
# mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
# mesh.translate(-mesh.get_center())

# === 2. Compute Normals and Color (optional) ===
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.2, 0.6, 0.8])  # Light blue

# === 3. Convert to Point Cloud ===
model_pcd = mesh.sample_points_uniformly(number_of_points=1500)

# === 4. Save to .ply file ===
o3d.io.write_point_cloud(output_ply, model_pcd)
print(f"STL-based model saved at: {output_ply}")

# === 5. Visualize the result ===
o3d.visualization.draw_geometries([model_pcd], window_name="Sampled STL Model Point Cloud")
