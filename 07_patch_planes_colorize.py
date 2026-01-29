import open3d as o3d
import numpy as np
import random

SCAN_IN = "scan_floor_aligned_no_floor.ply"

VOXEL_SIZE = 0.002
PLANE_DIST = 0.003
NUM_ITERS = 3000
MAX_PLANES = 5
MIN_INLIERS = 1200  # ggf. hoch/runter je nach Punktdichte

pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError(f"Could not load: {SCAN_IN}")

pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)

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

    patch = remaining.select_by_index(inliers)
    remaining = remaining.select_by_index(inliers, invert=True)

    # random color per patch
    color = [random.random(), random.random(), random.random()]
    patch.paint_uniform_color(color)

    a, b, c, d = model
    n = np.array([a, b, c], dtype=float)
    n /= np.linalg.norm(n)

    planes.append({"patch": patch, "normal": n, "inliers": len(inliers), "color": color})

print("\n=== Extracted side planes ===")
for idx, p in enumerate(planes):
    print(f"Plane {idx}: inliers={p['inliers']} normal={p['normal']} color={p['color']}")

# leftover in gray
remaining.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries([* [p["patch"] for p in planes], remaining])
