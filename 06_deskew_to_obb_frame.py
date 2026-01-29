import open3d as o3d
import numpy as np

SCAN_IN = "Earth_Cube_Scan_Test.ply"
SCAN_OUT = "cube_deskewed.ply"

# (Optional) nutze deinen bereits isolierten cube-cluster, wenn du den gespeichert hast
# Hier laden wir erstmal einfach den Scan:
pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError("Could not load scan.")

# ---- Tipp: wenn du den Boden schon entfernen willst, nutze hier dein isoliertes 'cube' statt pcd ----
# Für den Anfang: OBB direkt auf pcd kann funktionieren, ist aber besser auf dem cube-cluster.
# Wenn du schon cube als separate cloud hast: pcd = o3d.io.read_point_cloud("cube_only.ply")

# Downsample für stabile OBB
pcd_ds = pcd.voxel_down_sample(0.002)

# OBB berechnen
obb = pcd_ds.get_oriented_bounding_box()
R = obb.R  # 3x3 rotation matrix
center = obb.center

# Transformation: in OBB-Frame bringen
# 1) translate to center
# 2) rotate by R^T (inverse rotation)
T = np.eye(4)
T[:3, :3] = R.T
T[:3, 3] = -R.T @ center

pcd_aligned = pcd.transform(T)  # transform modifies in-place, careful!
# Workaround: reload for non-inplace if needed
pcd = o3d.io.read_point_cloud(SCAN_IN)
pcd_aligned = pcd.transform(T)

# Visual check: Jetzt sollte AABB gut passen
aabb = pcd_aligned.get_axis_aligned_bounding_box()
aabb.color = (0, 1, 0)

o3d.visualization.draw_geometries([pcd_aligned, aabb])

# Save
o3d.io.write_point_cloud(SCAN_OUT, pcd_aligned)
print("Saved:", SCAN_OUT)
