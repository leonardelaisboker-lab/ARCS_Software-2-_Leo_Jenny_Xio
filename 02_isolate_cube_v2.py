import open3d as o3d
import numpy as np

SCAN_PATH = "Earth_Cube_Scan_Test.ply"

# --- Params ---
VOXEL_SIZE = 0.002       # 2mm
PLANE_DIST = 0.004       # 4mm
NUM_ITERS = 2000
MAX_PLANES = 5           # how many planes to try
MIN_INLIERS = 2000       # ignore tiny planes (tune if needed)

DBSCAN_EPS = 0.01        # 1cm
DBSCAN_MIN_POINTS = 50

pcd = o3d.io.read_point_cloud(SCAN_PATH)
if pcd.is_empty():
    raise RuntimeError("Point cloud could not be loaded.")

pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)

# --- Extract multiple planes ---
planes = []
remaining = pcd_ds

for i in range(MAX_PLANES):
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

    pts = np.asarray(plane_cloud.points)
    z_mean = float(np.mean(pts[:, 2]))  # "height" in the scan's coordinate system

    planes.append({
        "model": model,
        "inliers": len(inliers),
        "z_mean": z_mean,
        "cloud": plane_cloud
    })

    remaining = rest_cloud

print("\n=== Candidate planes ===")
for idx, p in enumerate(planes):
    print(f"Plane {idx}: inliers={p['inliers']} | z_mean={p['z_mean']:.4f} | model={p['model']}")

if len(planes) == 0:
    raise RuntimeError("No planes found. Increase PLANE_DIST or reduce MIN_INLIERS.")

# --- Choose floor: lowest z_mean ---
floor_idx = int(np.argmin([p["z_mean"] for p in planes]))
floor = planes[floor_idx]["cloud"]

print(f"\nSelected floor plane: {floor_idx} (lowest z_mean)")

# --- Everything else = cube + other stuff ---
# Rebuild "rest" as: original downsampled minus selected floor indices is hard after iterative removal,
# so we do a simpler approach: segment AGAIN on original downsampled using selected model.
# This keeps it clean.

a, b, c, d = planes[floor_idx]["model"]

def plane_distance(points):
    # |ax+by+cz+d| / sqrt(a^2+b^2+c^2)
    return np.abs(points @ np.array([a, b, c]) + d) / np.sqrt(a*a + b*b + c*c)

pts_all = np.asarray(pcd_ds.points)
dist = plane_distance(pts_all)

floor_mask = dist < PLANE_DIST
floor_cloud = pcd_ds.select_by_index(np.where(floor_mask)[0])
rest_cloud = pcd_ds.select_by_index(np.where(~floor_mask)[0])

print("Floor points:", len(floor_cloud.points))
print("Remaining points:", len(rest_cloud.points))

# --- Cluster remaining points (cube should be largest cluster) ---
labels = np.array(
    rest_cloud.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=True)
)

if labels.max() < 0:
    raise RuntimeError("DBSCAN found no clusters. Try increasing DBSCAN_EPS or lowering DBSCAN_MIN_POINTS.")

valid = labels[labels >= 0]
largest_label = np.bincount(valid).argmax()
cube = rest_cloud.select_by_index(np.where(labels == largest_label)[0])

print("Largest cluster label:", largest_label)
print("Cube cluster points:", len(cube.points))
    
# --- Visualize ---
floor_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # gray
cube.paint_uniform_color([0.9, 0.2, 0.2])        # red

o3d.visualization.draw_geometries([floor_cloud, cube])
