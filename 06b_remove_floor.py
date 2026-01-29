import open3d as o3d
import numpy as np

SCAN_IN = "scan_floor_aligned.ply"
SCAN_OUT = "scan_floor_aligned_no_floor.ply"

pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError("Could not load aligned scan.")

Z_AXIS = np.array([0, 0, 1])
ANGLE_THRESH_DEG = 15  # Toleranz
DIST_THRESH = 0.002

floor_inliers = None

# Mehrere Plane-Versuche, bis wir eine "bodenartige" Plane finden
for i in range(5):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=DIST_THRESH,
        ransac_n=3,
        num_iterations=2000
    )

    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    angle = np.degrees(np.arccos(np.clip(np.dot(normal, Z_AXIS), -1.0, 1.0)))

    print(f"Try {i}: normal={normal}, angle_to_Z={angle:.2f}°")

    if angle < ANGLE_THRESH_DEG or abs(angle - 180) < ANGLE_THRESH_DEG:
        floor_inliers = inliers
        print("✅ Floor plane detected.")
        break
    else:
        # Diese Plane entfernen und weiter suchen
        pcd = pcd.select_by_index(inliers, invert=True)

if floor_inliers is None:
    raise RuntimeError("❌ No floor plane found.")

# Boden entfernen
pcd_no_floor = pcd.select_by_index(floor_inliers, invert=True)

o3d.io.write_point_cloud(SCAN_OUT, pcd_no_floor)
print("Saved:", SCAN_OUT)

# Visual check
pcd_no_floor.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd_no_floor])
