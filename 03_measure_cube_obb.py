import open3d as o3d
import numpy as np

SCAN_PATH = "Earth_Cube_Scan_Test.ply"
REAL_CUBE_MM = 70.0

# --- Params ---
VOXEL_SIZE = 0.002
PLANE_DIST = 0.004
NUM_ITERS = 2000
MAX_PLANES = 5
MIN_INLIERS = 2000

DBSCAN_EPS = 0.01
DBSCAN_MIN_POINTS = 50

# --- Load & downsample ---
pcd = o3d.io.read_point_cloud(SCAN_PATH)
pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)

# --- Extract multiple planes ---
planes = []
remaining = pcd_ds

for _ in range(MAX_PLANES):
    if len(remaining.points) < MIN_INLIERS:
        break

    model, inliers = remaining.segment_plane(
        distance_threshold=PLANE_DIST,
        ransac_n=3,
        num_iterations=NUM_ITERS
    )

    if len(inliers) < MIN_INLIERS:
        break

    plane_cloud = remaining.select_by_index(inliers)
    rest_cloud = remaining.select_by_index(inliers, invert=True)

    z_mean = np.mean(np.asarray(plane_cloud.points)[:, 2])

    planes.append((model, z_mean))
    remaining = rest_cloud

# --- pick lowest plane = floor ---
floor_model = min(planes, key=lambda p: p[1])[0]
a, b, c, d = floor_model

pts = np.asarray(pcd_ds.points)
dist = np.abs(pts @ np.array([a, b, c]) + d) / np.sqrt(a*a + b*b + c*c)

rest_cloud = pcd_ds.select_by_index(np.where(dist > PLANE_DIST)[0])

# --- cluster & take largest ---
labels = np.array(
    rest_cloud.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS)
)

valid = labels[labels >= 0]
largest_label = np.bincount(valid).argmax()
cube = rest_cloud.select_by_index(np.where(labels == largest_label)[0])

# --- OBB measurement ---
obb = cube.get_oriented_bounding_box()
extent_m = np.array(obb.extent)
extent_mm = extent_m * 1000.0

mean_mm = extent_mm.mean()
scale_factor = REAL_CUBE_MM / mean_mm

print("OBB extent (mm):", extent_mm)
print(f"Measured cube size (avg): {mean_mm:.2f} mm")
print(f"Expected cube size: {REAL_CUBE_MM:.2f} mm")
print(f"Scale factor → 70mm: {scale_factor:.4f}")

# --- visualize ---
cube.paint_uniform_color([0.9, 0.2, 0.2])
obb.color = (0, 1, 0)

o3d.visualization.draw_geometries([cube, obb])
