import open3d as o3d
import numpy as np

# -------------------------
# PARAMETERS
# -------------------------
DBSCAN_EPS = 0.015        # ggf. feinjustieren
DBSCAN_MIN_POINTS = 50

SCAN_IN = "cube_only.ply"
OUTPUT_PLY = "cube_main_cluster.ply"

# -------------------------
# LOAD POINT CLOUD
# -------------------------
pcd = o3d.io.read_point_cloud(SCAN_IN)

if len(pcd.points) == 0:
    raise RuntimeError("Point cloud is empty")

# -------------------------
# DBSCAN CLUSTERING
# -------------------------
labels = np.array(
    pcd.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=True
    )
)

# -------------------------
# VALID CLUSTERS ONLY
# -------------------------
valid_labels = labels[labels >= 0]

if len(valid_labels) == 0:
    raise RuntimeError(
        "No clusters found. Try increasing DBSCAN_EPS."
    )

# -------------------------
# LARGEST CLUSTER LABEL
# -------------------------
largest_label = np.bincount(valid_labels).argmax()

# -------------------------
# EXTRACT CLUSTER
# -------------------------
mask = labels == largest_label
pcd_main = pcd.select_by_index(np.where(mask)[0])

# -------------------------
# SAVE RESULT
# -------------------------
o3d.io.write_point_cloud(OUTPUT_PLY, pcd_main)

print(f"✔ Extracted largest cluster (label={largest_label})")
print(f"✔ Points: {len(pcd_main.points)}")
print(f"✔ Saved as: {OUTPUT_PLY}")

