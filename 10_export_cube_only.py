import open3d as o3d
import numpy as np

SCAN_IN = "scan_floor_aligned.ply"
CUBE_OUT = "cube_only.ply"

VOXEL_SIZE = 0.003
Z_SLAB = 0.004
DBSCAN_EPS = 0.015
DBSCAN_MIN_POINTS = 50

pcd = o3d.io.read_point_cloud(SCAN_IN)
pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)

pts = np.asarray(pcd_ds.points)
z_min = float(np.min(pts[:, 2]))
mask_floor = pts[:, 2] <= (z_min + Z_SLAB)
nofloor = pcd_ds.select_by_index(np.where(~mask_floor)[0])

labels = np.array(nofloor.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS))
valid = labels[labels >= 0]
largest_label = np.bincount(valid).argmax()
cube = nofloor.select_by_index(np.where(labels == largest_label)[0])

o3d.io.write_point_cloud(CUBE_OUT, cube)
print("Saved:", CUBE_OUT, "| points:", len(cube.points))

cube.paint_uniform_color([1,0,0])
o3d.visualization.draw_geometries([cube])
