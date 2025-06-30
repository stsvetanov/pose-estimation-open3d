import os
import sys
import trimesh
import numpy as np
import cv2
import argparse
from scipy.ndimage import distance_transform_edt
from collections import Counter


def discover_cad_files(cad_dir, ext='.stl'):
    """
    Discover CAD files in cad_dir with given extension.
    Returns a sorted list of file paths.
    """
    if not os.path.isdir(cad_dir):
        raise FileNotFoundError(f"CAD directory not found: {cad_dir}")
    cad_paths = [os.path.join(cad_dir, f) for f in os.listdir(cad_dir)
                 if f.lower().endswith(ext.lower())]
    cad_paths.sort()
    return cad_paths


def extract_3d_points(mesh, target_radius_mm=10.0):
    """
    Extract 3D feature points from mesh:
      - hole center via boundary loop circle fitting
      - arc endpoints and straight-line endpoints
    Debug prints included.
    """
    # Build edge usage count to find boundary edges
    edge_counts = Counter()
    for face in mesh.faces:
        for i in range(3):
            u, v = sorted((int(face[i]), int(face[(i+1)%3])))
            edge_counts[(u, v)] += 1
    # Boundary edges used only once
    edges = np.array([e for e, cnt in edge_counts.items() if cnt == 1], dtype=int)
    print(f"[DEBUG] boundary edges count: {len(edges)}")
    if len(edges) == 0:
        raise RuntimeError("No boundary edges found; mesh is closed or malformed.")
    # Build adjacency for loops
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    # Traverse loops
    loops = []
    visited = set()
    for u, v in edges:
        if (u, v) in visited:
            continue
        loop = [u, v]
        visited.add((u, v)); visited.add((v, u))
        prev, curr = u, v
        while True:
            nbrs = [n for n in adj.get(curr, []) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            visited.add((curr, nxt)); visited.add((nxt, curr))
            if nxt == loop[0]:
                loop.append(nxt)
                break
            loop.append(nxt)
            prev, curr = curr, nxt
        loops.append(loop)
    print(f"[DEBUG] found {len(loops)} boundary loops")

    # Fit circle to each loop to detect hole
    hole_center = None
    for idx, loop in enumerate(loops):
        pts = mesh.vertices[loop]
        xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
        A = np.column_stack([2*xs, 2*ys, np.ones_like(xs)])
        b = xs**2 + ys**2
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        x0, y0, c0 = coef
        r0 = np.sqrt(max(0, c0 + x0*x0 + y0*y0))
        print(f"[DEBUG] loop {idx} size={len(loop)} radius={r0:.2f}")
        if abs(r0 - target_radius_mm) < 0.2 * target_radius_mm:
            hole_center = np.array([x0, y0, zs.mean()])
            print(f"[DEBUG] selected hole loop {idx} center={hole_center}")
            break
    if hole_center is None:
        raise RuntimeError("Hole not detected; adjust target_radius_mm or check mesh.")

    # Classify edges by length for straight vs curved
    edge_vecs = mesh.vertices[edges[:,0]] - mesh.vertices[edges[:,1]]
    lengths = np.linalg.norm(edge_vecs, axis=1)
    straight_idxs = np.argsort(lengths)[-4:]
    straight_verts = np.unique(edges[straight_idxs].ravel())
    all_b = np.unique(edges.ravel())
    curved_verts = np.setdiff1d(all_b, straight_verts)

    # Assemble 3D points
    object_pts = [hole_center]
    for v in curved_verts:
        object_pts.append(mesh.vertices[v])
    for v in straight_verts:
        object_pts.append(mesh.vertices[v])
    object_pts = np.array(object_pts, dtype=np.float32)
    print(f"[DEBUG] extracted {len(object_pts)} object points")
    return object_pts


def detect_2d_points(image, num_pts, target_radius_px=15):
    """
    Detect hole and boundary feature points in image.
    Returns Nx2 numpy array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Hole detection by Hough
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=30,
                               minRadius=int(target_radius_px*0.5),
                               maxRadius=int(target_radius_px*1.5))
    if circles is None:
        raise RuntimeError("Hole not detected in image.")
    xh, yh, _ = np.round(circles[0,0]).astype(int)
    pts = [(xh, yh)]
    # Contour for boundary
    _, thr = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No contours in image.")
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, 0.02*peri, True).reshape(-1,2)
    if len(poly) < num_pts - 1:
        raise RuntimeError(f"Need {num_pts-1} boundary pts, got {len(poly)}")
    for i in range(num_pts-1):
        pts.append(tuple(poly[i]))
    return np.array(pts, dtype=np.float32)


def solve_pose(obj_pts, img_pts, K, dist, table_h):
    init_t = np.array([[0],[0],[table_h]], dtype=np.float32)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts, imagePoints=img_pts,
        cameraMatrix=K, distCoeffs=dist,
        rvec=None, tvec=init_t, useExtrinsicGuess=True,
        iterationsCount=200, reprojectionError=4.0,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok or len(inliers) < 4:
        raise RuntimeError("PnPRansac failed or insufficient inliers.")
    return rvec, tvec, inliers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pose estimation from CAD and image.")
    # parser.add_argument('--cad-dir', default=os.path.join(os.sep,'data','3D-models'))
    parser.add_argument('--cad-dir', default='data/3D-models')
    # parser.add_argument('--image', default=os.path.join(os.sep,'data','images','IMG_20250603_193716_763.jpg'))
    parser.add_argument('--image', default='data/images/IMG_20250603_193728_905.jpg')
    parser.add_argument('--table-height', type=float, default=400.0)
    parser.add_argument('--hole-radius', type=float, default=10.0,
                        help='Hole radius in CAD units (mm)')
    parser.add_argument('--hole-radius-px', type=float, default=15.0,
                        help='Approx hole radius in image (pixels)')
    args = parser.parse_args()

    cad_list = discover_cad_files(args.cad_dir)
    if not cad_list:
        raise RuntimeError(f"No .stl CAD files in {args.cad_dir}")
    print(f"Using CAD: {cad_list[2]}")
    mesh = trimesh.load(cad_list[2], force='mesh')
    print(f"[DEBUG] Mesh vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
    # Visualize mesh
    try:
        mesh.show()
    except:
        pass

    obj_pts = extract_3d_points(mesh, target_radius_mm=args.hole_radius)
    print(f"3D points: {len(obj_pts)}")

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Cannot load image: {args.image}")
    K = np.array([[615.34515,0,313.39371],[0,615.40436,251.59381],[0,0,1]],dtype=np.float64)
    dist = np.zeros(5)
    img_ud = cv2.undistort(img, K, dist)

    # Visualize 2D preprocessing
    gray = cv2.cvtColor(img_ud, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray); cv2.waitKey(1)

    img_pts = detect_2d_points(img_ud, len(obj_pts), target_radius_px=args.hole_radius_px)
    print(f"2D points: {len(img_pts)}")
    # Draw 2D points
    dbg = img_ud.copy()
    for i,(x,y) in enumerate(img_pts):
        color = (0,0,255) if i==0 else (0,255,0)
        cv2.circle(dbg, (int(x),int(y)), 5, color, -1)
    cv2.imshow('Features', dbg); cv2.waitKey(0); cv2.destroyAllWindows()

    rvec, tvec, inliers = solve_pose(obj_pts, img_pts, K, dist, args.table_height)
    R, _ = cv2.Rodrigues(rvec)
    print("Rotation matrix:\n", R)
    print("Translation (mm):\n", tvec.ravel())


