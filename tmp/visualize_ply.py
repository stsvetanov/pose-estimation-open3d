import open3d as o3d
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Visualize a PLY file using Open3D.")
    parser.add_argument("file", help="Path to the .ply file to visualize")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"[ERROR] File not found: {args.file}")
        sys.exit(1)

    try:
        pcd = o3d.io.read_point_cloud(args.file)
        print(f"[INFO] Loaded point cloud with {len(pcd.points)} points.")
        o3d.visualization.draw_geometries([pcd], window_name="PLY Viewer")
    except Exception as e:
        print(f"[ERROR] Failed to read or visualize PLY: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
