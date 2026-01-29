import open3d as o3d
import numpy as np

SCAN_IN = "scan_floor_aligned.ply"
SCAN_OUT = "scan_floor_aligned_no_floor.ply"

# Wie dick ist die "Boden-Schicht", die wir entfernen?
# Startwert: 0.003 = 3mm (weil Units Meter)
Z_SLAB = 0.003

pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError("Could not load aligned scan.")

pts = np.asarray(pcd.points)

z_min = float(np.min(pts[:, 2]))
mask_floor = pts[:, 2] <= (z_min + Z_SLAB)

print("z_min:", z_min)
print("Removing points in z <= z_min +", Z_SLAB, "->", int(mask_floor.sum()), "points")

pcd_no_floor = pcd.select_by_index(np.where(~mask_floor)[0])

o3d.io.write_point_cloud(SCAN_OUT, pcd_no_floor)
print("Saved:", SCAN_OUT)

# Visual check
pcd_no_floor.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd_no_floor])
