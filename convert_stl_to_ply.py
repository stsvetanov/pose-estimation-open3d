import open3d as o3d
import argparse
import os

def convert_stl_to_ply(input_stl, output_ply, mode, scale_to_meters=True, align_to_scene=None):
    print(f"[INFO] Loading STL mesh from: {input_stl}")
    model = o3d.io.read_triangle_mesh(input_stl)
    if model.is_empty():
        raise ValueError("Failed to load STL mesh.")

    # Optional: scale from mm to meters
    if scale_to_meters:
        print("[INFO] Scaling mesh from millimeters to meters.")
        model.scale(0.001, center=model.get_center())

    # Optional: align model base to scene ground level
    if align_to_scene:
        print(f"[INFO] Aligning mesh to scene from: {align_to_scene}")
        scene = o3d.io.read_point_cloud(align_to_scene)
        if scene.is_empty():
            raise ValueError("Failed to load scene point cloud.")
        # scene_ground_z = scene.get_min_bound()[2]
        # model_bottom_z = mesh.get_min_bound()[2]
        # dz = scene_ground_z - model_bottom_z
        # print(f"[INFO] Translating model by dz = {dz:.4f} to match scene table level.")
        # mesh.translate([0, 0, dz])
        model.translate(scene.get_center() - model.get_center())
        # o3d.visualization.draw_geometries([scene, model], window_name="Box + Plate4 Aligned")

    if mode == "pcd":
        # Convert to point cloud
        print("[INFO] Sampling mesh surface points...")
        model = model.sample_points_uniformly(number_of_points=2000)

    print(f"[INFO] Saving aligned point cloud to: {output_ply}")
    o3d.io.write_point_cloud(output_ply, model)
    print("[DONE]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert STL to aligned PLY point cloud for RealSense scenes.")
    parser.add_argument("--input", required=True, help="Path to input STL file")
    parser.add_argument("--output", required=True, help="Path to output PLY file")
    parser.add_argument("--scene", help="Optional RealSense scene PLY file for ground alignment")
    parser.add_argument("--no_scale", action="store_true", help="Disable scaling from mm to meters")
    parser.add_argument("--mode", choices=["pcd", "mesh"], default="pcd")

    args = parser.parse_args()

    convert_stl_to_ply(
        input_stl=args.input,
        output_ply=args.output,
        scale_to_meters=not args.no_scale,
        align_to_scene=args.scene,
        mode=args.mode
    )
