import open3d as o3d
import numpy as np

SCAN_IN = "scan_floor_aligned.ply"

# ---- Parameters ----
VOXEL_SIZE = 0.003          # 3mm downsample for processing
Z_SLAB = 0.004              # remove bottom 4mm as floor
DBSCAN_EPS = 0.015          # 1.5 cm
DBSCAN_MIN_POINTS = 50

# ---- Load ----
pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError("Could not load scan_floor_aligned.ply")

# ---- Downsample for stable clustering ----
pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)

# ---- Remove floor by z-slab (after floor alignment) ----
pts = np.asarray(pcd_ds.points)
z_min = float(np.min(pts[:, 2]))
mask_floor = pts[:, 2] <= (z_min + Z_SLAB)

floor = pcd_ds.select_by_index(np.where(mask_floor)[0])
nofloor = pcd_ds.select_by_index(np.where(~mask_floor)[0])

print("Floor removed points:", len(floor.points))
print("Remaining points:", len(nofloor.points))

# ---- Cluster remaining (cube should be largest cluster) ----
labels = np.array(
    nofloor.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=True)
)

if labels.max() < 0:
    raise RuntimeError("DBSCAN found no clusters. Increase DBSCAN_EPS or lower MIN_POINTS.")

valid = labels[labels >= 0]
largest_label = np.bincount(valid).argmax()
cube = nofloor.select_by_index(np.where(labels == largest_label)[0])

print("Cube points:", len(cube.points))

# ---- Compute bounding box ONLY on cube ----
obb = cube.get_oriented_bounding_box()
obb.color = (0, 1, 0)

# ---- Colorize & show ----
floor.paint_uniform_color([0.7, 0.7, 0.7])   # gray
cube.paint_uniform_color([0.9, 0.2, 0.2])    # red

o3d.visualization.draw_geometries([floor, cube, obb])
