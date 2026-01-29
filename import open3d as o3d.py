import open3d as o3d
import numpy as np

# ==============================
# CONFIG
# ==============================
SCAN_PATH = "Earth_Cube_Scan_Test.ply"
REAL_CUBE_MM = 70.0  # realer Würfel

# ==============================
# LOAD POINT CLOUD
# ==============================
pcd = o3d.io.read_point_cloud(SCAN_PATH)
if pcd.is_empty():
    raise RuntimeError("Point cloud could not be loaded.")

# ==============================
# BOUNDING BOX
# ==============================
bbox = pcd.get_axis_aligned_bounding_box()
extent = np.array(bbox.get_extent())

print("Raw bounding box extent:", extent)

# ==============================
# UNIT & SCALE CHECK
# ==============================
mean_extent = extent.mean()

if mean_extent > 1.0:
    detected_unit = "mm"
    measured_mm = mean_extent
elif mean_extent > 0.01:
    detected_unit = "m"
    measured_mm = mean_extent * 1000
else:
    detected_unit = "unknown"
    measured_mm = None

print(f"Detected unit: {detected_unit}")
print(f"Measured cube size ≈ {measured_mm:.2f} mm")
print(f"Expected cube size: {REAL_CUBE_MM} mm")
print(f"Deviation: {(measured_mm - REAL_CUBE_MM):.2f} mm")

# ==============================
# VISUALIZE
# ==============================
o3d.visualization.draw_geometries([pcd, bbox])
