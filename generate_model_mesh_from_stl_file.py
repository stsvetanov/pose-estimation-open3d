import open3d as o3d
import os

# Input STL file path
model_file = "data/3D-models/Plate4.stl"

# Output directory
output_dir = "icp_data"
os.makedirs(output_dir, exist_ok=True)  # Create if not exists

# Load the mesh
mesh = o3d.io.read_triangle_mesh(model_file)

# Check if mesh is valid
if mesh.is_empty():
    print(f"Error: Could not load mesh from {model_file}")
else:
    # Optional: compute normals
    mesh.compute_vertex_normals()

    # Generate output filename
    base_name = os.path.splitext(os.path.basename(model_file))[0]
    ply_file = os.path.join(output_dir, base_name + ".ply")

    # Save as .ply
    success = o3d.io.write_triangle_mesh(ply_file, mesh)
    if success:
        print(f"Saved PLY file to: {ply_file}")
    else:
        print("Error: Failed to save PLY file.")
