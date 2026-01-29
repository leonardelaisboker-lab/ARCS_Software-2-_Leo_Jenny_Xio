import open3d as o3d
import numpy as np

SCAN_PATH = "Earth_Cube_Scan_Test.ply"

# --- Params (kannst du später tunen) ---
VOXEL_SIZE = 0.002        # 2mm downsample (in "scan units"; bei dir = Meter → passt)
PLANE_DIST = 0.004        # 4mm Abstandstoleranz für Boden-Plane
RANSAC_N = 3
NUM_ITERS = 2000

DBSCAN_EPS = 0.01         # 1cm Nachbarschaft (in Meter)
DBSCAN_MIN_POINTS = 50

pcd = o3d.io.read_point_cloud(SCAN_PATH)
if pcd.is_empty():
    raise RuntimeError("Point cloud could not be loaded.")

# Downsample (stabiler + schneller)
pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)

# --- 1) Plane segmentation (remove floor) ---
plane_model, inliers = pcd_ds.segment_plane(
    distance_threshold=PLANE_DIST,
    ransac_n=RANSAC_N,
    num_iterations=NUM_ITERS
)

floor = pcd_ds.select_by_index(inliers)
rest = pcd_ds.select_by_index(inliers, invert=True)

print("Plane model (ax + by + cz + d = 0):", plane_model)
print("Floor points:", np.asarray(floor.points).shape[0])
print("Remaining points:", np.asarray(rest.points).shape[0])

# --- 2) Cluster remaining points (cube should be largest cluster) ---
labels = np.array(
    rest.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=True)
)

if labels.max() < 0:
    raise RuntimeError("DBSCAN found no clusters. Try increasing DBSCAN_EPS or lowering MIN_POINTS.")

# pick largest cluster (ignore -1 = noise)
valid = labels[labels >= 0]
largest_label = np.bincount(valid).argmax()

cube = rest.select_by_index(np.where(labels == largest_label)[0])

print("Largest cluster label:", largest_label)
print("Cube cluster points:", np.asarray(cube.points).shape[0])

# --- Visual checks ---
floor.paint_uniform_color([0.7, 0.7, 0.7])   # gray
cube.paint_uniform_color([0.9, 0.3, 0.3])    # red
rest.paint_uniform_color([0.2, 0.2, 0.8])    # blue (other stuff)

o3d.visualization.draw_geometries([floor, cube])