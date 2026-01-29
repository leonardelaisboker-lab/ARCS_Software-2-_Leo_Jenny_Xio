import open3d as o3d
import numpy as np

# ==============================
# FILES
# ==============================
SCAN_IN = "scan_floor_aligned.ply"
CUBE_OUT = "cube_only.ply"

# ==============================
# PARAMETERS
# ==============================
VOXEL_SIZE = 0.003        # 3 mm
PLANE_DIST = 0.004        # max distance to plane (4 mm)
DBSCAN_EPS = 0.018
DBSCAN_MIN_POINTS = 40

# ==============================
# LOAD
# ==============================
pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError("Could not load scan")

pcd = pcd.voxel_down_sample(VOXEL_SIZE)

# ==============================
# REMOVE FLOOR VIA RANSAC PLANE
# ==============================
plane_model, inliers = pcd.segment_plane(
    distance_threshold=PLANE_DIST,
    ransac_n=3,
    num_iterations=2000
)

[a, b, c, d] = plane_model
print("Detected plane:", plane_model)
print("Plane inliers:", len(inliers))

floor = pcd.select_by_index(inliers)
nofloor = pcd.select_by_index(inliers, invert=True)

if len(nofloor.points) < 1000:
    raise RuntimeError("Too few points after floor removal")

# ==============================
# CLUSTER (cube = largest)
# ==============================
labels = np.array(
    nofloor.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=True
    )
)

if labels.max() < 0:
    raise RuntimeError("No clusters found – increase DBSCAN_EPS")

valid = labels[labels >= 0]
largest_label = np.bincount(valid).argmax()

cube = nofloor.select_by_index(np.where(labels == largest_label)[0])
print("Cube points:", len(cube.points))

# ==============================
# SAVE + VISUALIZE
# ==============================
cube.paint_uniform_color([1, 0, 0])
o3d.io.write_point_cloud(CUBE_OUT, cube)

o3d.visualization.draw_geometries([
    cube,
])
