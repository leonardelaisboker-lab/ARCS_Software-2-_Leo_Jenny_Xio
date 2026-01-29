import open3d as o3d
import numpy as np
import os

# ==============================
# INPUT / OUTPUT
# ==============================
SCAN_IN = "cube_only.ply"
SCAN_OUT = "cube_only_yaw_aligned.ply"

if not os.path.exists(SCAN_IN):
    raise FileNotFoundError(f"Missing {SCAN_IN} in folder.")

# ==============================
# PARAMS
# ==============================
VOXEL_SIZE = 0.002
NORMAL_RADIUS = 0.01

# ==============================
# LOAD
# ==============================
pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError("Input point cloud is empty.")

pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)

# ==============================
# NORMALS
# ==============================
pcd_ds.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=50)
)
pcd_ds.orient_normals_consistent_tangent_plane(50)

N = np.asarray(pcd_ds.normals)
N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)

# ==============================
# PICK A DOMINANT VERTICAL NORMAL
# Strategy:
# - take normals with |nz| small (vertical faces)
# - histogram their angles in XY and pick the dominant direction
# ==============================
vertical_mask = np.abs(N[:, 2]) < 0.3
Nv = N[vertical_mask]

if len(Nv) < 200:
    raise RuntimeError("Too few vertical normals found. Try increasing NORMAL_RADIUS or relax threshold 0.3 -> 0.5")

# Project to XY and normalize
Nv_xy = Nv[:, :2]
Nv_xy_norm = np.linalg.norm(Nv_xy, axis=1, keepdims=True) + 1e-9
Nv_xy = Nv_xy / Nv_xy_norm

angles = np.arctan2(Nv_xy[:, 1], Nv_xy[:, 0])  # [-pi, pi]
# Map to [0, pi) because opposite normals represent same face direction
angles = np.mod(angles, np.pi)

# Histogram to find dominant direction
bins = 180
hist, edges = np.histogram(angles, bins=bins, range=(0, np.pi))
peak_bin = np.argmax(hist)
angle_peak = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0

# Rotate so that dominant direction aligns with +X (angle 0)
yaw = -angle_peak

Rz = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw),  np.cos(yaw), 0],
    [0,            0,           1]
])

# Apply yaw rotation to FULL cloud around its center (prevents drift)
center = pcd.get_center()
pcd.translate(-center)
pts = np.asarray(pcd.points)
pts_rot = (Rz @ pts.T).T

pcd_aligned = o3d.geometry.PointCloud()
pcd_aligned.points = o3d.utility.Vector3dVector(pts_rot)
pcd_aligned.colors = pcd.colors
pcd_aligned.translate(center)

# Save
ok = o3d.io.write_point_cloud(SCAN_OUT, pcd_aligned)
print("Saved:", SCAN_OUT, "| ok:", ok)
print("Yaw(deg):", float(np.degrees(yaw)))

# ==============================
# MEASURE via OBB (for size + later scaling)
# ==============================
obb = pcd_aligned.get_oriented_bounding_box()
extent_mm = np.array(obb.extent) * 1000.0
print("OBB extent (mm):", extent_mm)
print("OBB mean edge (mm):", float(extent_mm.mean()))

obb.color = (0, 1, 0)
pcd_aligned.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries([pcd_aligned, obb])