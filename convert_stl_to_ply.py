import open3d as o3d
import argparse
import os
import numpy as np

def convert_stl_to_ply(input_stl, output_ply, mode, scale_to_meters=True,
                       align_to_scene=None, flatten=False, num_points=2000, z_noise=0.001):
    print(f"[INFO] Loading STL mesh from: {input_stl}")
    mesh = o3d.io.read_triangle_mesh(input_stl)
    if mesh.is_empty():
        raise ValueError("Failed to load STL mesh.")

    if scale_to_meters:
        print("[INFO] Scaling mesh from millimeters to meters.")
        mesh.scale(0.001, center=mesh.get_center())

    if align_to_scene:
        print(f"[INFO] Aligning mesh to scene from: {align_to_scene}")
        scene = o3d.io.read_point_cloud(align_to_scene)
        if scene.is_empty():
            raise ValueError("Failed to load scene point cloud.")
        mesh.translate(scene.get_center() - mesh.get_center())

    if mode == "pcd":
        print(f"[INFO] Sampling full mesh surface with {num_points} points...")
        mesh = mesh.sample_points_uniformly(number_of_points=num_points)

    elif mode == "topdown":
        print("[INFO] Extracting top-facing triangles and sampling from them...")

        mesh.compute_triangle_normals()
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.triangle_normals)

        upward_mask = normals[:, 2] > 0.9
        top_triangles = triangles[upward_mask]

        if len(top_triangles) == 0:
            raise ValueError("No top-facing triangles found. Check mesh orientation.")

        top_mesh = o3d.geometry.TriangleMesh()
        top_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        top_mesh.triangles = o3d.utility.Vector3iVector(top_triangles)
        top_mesh.compute_vertex_normals()

        print(f"[INFO] Sampling {num_points} points from top-facing surface...")
        pcd = top_mesh.sample_points_uniformly(number_of_points=num_points)

        if flatten:
            print(f"[INFO] Flattening Z coordinates with noise σ={z_noise} m...")
            points = np.asarray(pcd.points)
            points[:, 2] = np.random.normal(loc=0.0, scale=z_noise, size=points.shape[0])
            pcd.points = o3d.utility.Vector3dVector(points)

        mesh = pcd

    print(f"[INFO] Saving point cloud to: {output_ply}")
    o3d.io.write_point_cloud(output_ply, mesh)
    print("[DONE]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert STL to aligned PLY point cloud for RealSense scenes.")
    parser.add_argument("--input", required=True, help="Path to input STL file")
    parser.add_argument("--output", required=True, help="Path to output PLY file")
    parser.add_argument("--scene", help="Optional RealSense scene PLY file for ground alignment")
    parser.add_argument("--no_scale", action="store_true", help="Disable scaling from mm to meters")
    parser.add_argument("--mode", choices=["pcd", "mesh", "topdown"], default="pcd", help="Conversion mode")
    parser.add_argument("--flat", action="store_true", help="Flatten points to XY plane with Z noise")
    parser.add_argument("--num_points", type=int, default=2000, help="Number of points to sample from surface")
    parser.add_argument("--z_noise", type=float, default=0.0003, help="Stddev of noise added to Z when flattening")

    args = parser.parse_args()

    convert_stl_to_ply(
        input_stl=args.input,
        output_ply=args.output,
        scale_to_meters=not args.no_scale,
        align_to_scene=args.scene,
        mode=args.mode,
        flatten=args.flat,
        num_points=args.num_points,
        z_noise=args.z_noise
    )
