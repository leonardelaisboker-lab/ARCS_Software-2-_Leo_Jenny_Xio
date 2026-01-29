import open3d as o3d
import numpy as np

SCAN_IN = "Earth_Cube_Scan_Test.ply"
SCAN_OUT_ALIGNED = "scan_floor_aligned.ply"
SCAN_OUT_NO_FLOOR = "scan_floor_aligned_no_floor.ply"

pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError("Could not load scan.")

# Downsample für stabile Plane Detection
pcd_ds = pcd.voxel_down_sample(0.005)

# ---- Plane segmentation (Boden) ----
plane_model, inliers = pcd_ds.segment_plane(
    distance_threshold=0.003,
    ransac_n=3,
    num_iterations=2000
)

[a, b, c, d] = plane_model
normal = np.array([a, b, c])
normal /= np.linalg.norm(normal)

print("Floor normal:", normal)

# Ziel-Normal (Z-Achse)
target = np.array([0, 0, 1])

# Rotation berechnen (Rodrigues)
v = np.cross(normal, target)
s = np.linalg.norm(v)
c = np.dot(normal, target)

if s < 1e-6:
    R = np.eye(3)
else:
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

# Scan rotieren
pcd.rotate(R, center=(0, 0, 0))

# Translation: Boden auf z = 0
points = np.asarray(pcd.points)
z_min = points[:, 2].min()
pcd.translate((0, 0, -z_min))

# Speichern (aligned)
o3d.io.write_point_cloud(SCAN_OUT_ALIGNED, pcd)
print("Saved:", SCAN_OUT_ALIGNED)

# Boden entfernen
labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=100))
# einfacher: erneut Plane finden
_, floor_idx = pcd.segment_plane(
    distance_threshold=0.002,
    ransac_n=3,
    num_iterations=1000
)
pcd_no_floor = pcd.select_by_index(floor_idx, invert=True)

o3d.io.write_point_cloud(SCAN_OUT_NO_FLOOR, pcd_no_floor)
print("Saved:", SCAN_OUT_NO_FLOOR)

# Visual check
pcd_no_floor.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd_no_floor])
