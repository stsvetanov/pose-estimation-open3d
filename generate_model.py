import open3d as o3d
import os
import argparse

def generate_simple_box(output_dir, filename="sample_object_model.ply"):
    print("[INFO] Generating Rectangular Object model...")
    width, height, depth = 0.06, 0.04, 0.008  # 60x40x8 mm
    box = o3d.geometry.TriangleMesh.create_box(width, height, depth)
    box.compute_vertex_normals()
    box.paint_uniform_color([0.2, 0.6, 0.8])

    model_pcd = box.sample_points_uniformly(number_of_points=5000)

    output_path = os.path.join(output_dir, filename)
    o3d.io.write_point_cloud(output_path, model_pcd)
    print(f"[INFO] Box point cloud saved to: {output_path}")
    o3d.visualization.draw_geometries([model_pcd], window_name="Box Point Cloud")


def generate_pcd_from_stl(input_file, output_dir):
    print(f"[INFO] Loading STL and generating point cloud: {input_file}")
    mesh = o3d.io.read_triangle_mesh(input_file)
    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh from {input_file}")

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.2, 0.6, 0.8])
    pcd = mesh.sample_points_uniformly(number_of_points=5000)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, base_name + "_pcd.ply")
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[INFO] Point cloud saved to: {output_path}")
    o3d.visualization.draw_geometries([pcd], window_name="STL Sampled Point Cloud")


def generate_mesh_from_stl(input_file, output_dir):
    print(f"[INFO] Loading STL and saving as mesh: {input_file}")
    mesh = o3d.io.read_triangle_mesh(input_file)
    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh from {input_file}")

    mesh.compute_vertex_normals()
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, base_name + ".ply")

    if o3d.io.write_triangle_mesh(output_path, mesh):
        print(f"[INFO] Mesh saved to: {output_path}")
    else:
        print("[ERROR] Failed to save mesh.")


def main():
    parser = argparse.ArgumentParser(description="Generate 3D model representations using Open3D.")
    parser.add_argument(
        "--mode", choices=["box", "stl_pcd", "stl_mesh"], default="box",
        help="Type of model to generate: 'box', 'stl_pcd', or 'stl_mesh'"
    )
    parser.add_argument(
        "--input", type=str, default="data/3D-models/Plate4.stl",
        help="Path to STL file (used in 'stl_pcd' or 'stl_mesh' mode)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="icp_data",
        help="Directory to save the output files"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "box":
        generate_simple_box(args.output_dir)
    elif args.mode == "stl_pcd":
        generate_pcd_from_stl(args.input, args.output_dir)
    elif args.mode == "stl_mesh":
        generate_mesh_from_stl(args.input, args.output_dir)
    else:
        print("[ERROR] Unknown mode selected.")


if __name__ == "__main__":
    main()
