import open3d as o3d
import numpy as np
import os

# ==============================
# INPUT (robust)
# ==============================
SCAN_IN = "cube_only.ply"
if not os.path.exists(SCAN_IN):
    SCAN_IN = "cube_no_floor.ply"

if not os.path.exists(SCAN_IN):
    raise FileNotFoundError(
        "Could not find input point cloud. Expected 'cube_only.ply' (preferred) "
        "or 'cube_no_floor.ply' in the current folder."
    )

print("Using input:", SCAN_IN)

# ==============================
# PARAMS
# ==============================
VOXEL_SIZE = 0.002          # 2mm downsample
NORMAL_RADIUS = 0.01        # 1cm neighborhood
K = 6                       # cluster normals into 6 groups, keep top 5
TOP_FACES = 5

# ==============================
# HELPERS
# ==============================
def kmeans(X, k, iters=40, seed=7):
    """Simple k-means (no sklearn). X: (N, D). Returns labels, centers."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    C = X[idx].copy()

    for _ in range(iters):
        # assign
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)

        # update
        C_new = np.zeros_like(C)
        for j in range(k):
            pts = X[labels == j]
            if len(pts) == 0:
                C_new[j] = C[j]
            else:
                C_new[j] = pts.mean(axis=0)

        # normalize centers (important for normal directions)
        C_new = C_new / (np.linalg.norm(C_new, axis=1, keepdims=True) + 1e-9)

        if np.allclose(C, C_new, atol=1e-4):
            C = C_new
            break
        C = C_new

    return labels, C

def fold_normals_to_hemisphere(N):
    """
    Fold normals so +n and -n become consistent (stabilizes clustering).
    Rule: prefer nz>=0; if nz==0, prefer ny>=0; if ny==0, prefer nx>=0.
    """
    N2 = N.copy()
    for i in range(len(N2)):
        n = N2[i]
        if n[2] < 0 or (abs(n[2]) < 1e-9 and n[1] < 0) or (abs(n[2]) < 1e-9 and abs(n[1]) < 1e-9 and n[0] < 0):
            N2[i] = -n
    return N2

# ==============================
# LOAD
# ==============================
pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError(f"Point cloud is empty: {SCAN_IN}")

# Downsample for stability
pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)
if pcd_ds.is_empty():
    raise RuntimeError("Downsample produced empty point cloud. Reduce VOXEL_SIZE.")

# ==============================
# NORMALS
# ==============================
pcd_ds.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=50)
)
pcd_ds.orient_normals_consistent_tangent_plane(50)

N = np.asarray(pcd_ds.normals)
N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
N_fold = fold_normals_to_hemisphere(N)

# ==============================
# CLUSTER NORMALS
# ==============================
labels, centers = kmeans(N_fold, K, iters=50, seed=7)
counts = np.bincount(labels, minlength=K)
order = np.argsort(counts)[::-1]  # largest first
keep = order[:TOP_FACES]

# Colors for the top faces (5)
face_colors = np.array([
    [0.90, 0.20, 0.20],  # red
    [0.20, 0.70, 0.90],  # cyan
    [0.60, 0.40, 0.90],  # purple
    [0.80, 0.80, 0.20],  # yellow
    [0.40, 0.80, 0.40],  # green
], dtype=float)

colors = np.zeros((len(labels), 3), dtype=float)
colors[:] = [0.6, 0.6, 0.6]  # default gray

cluster_to_face = {cid: i for i, cid in enumerate(keep)}
for cid in keep:
    face_i = cluster_to_face[cid]
    colors[labels == cid] = face_colors[face_i]

pcd_ds.colors = o3d.utility.Vector3dVector(colors)

# ==============================
# REPORT
# ==============================
total = len(labels)
print("\n=== FACE CLUSTER REPORT (top 5 clusters) ===")
for cid in keep:
    face_i = cluster_to_face[cid]
    pct = 100.0 * counts[cid] / total
    n = centers[cid]
    print(f"Face {face_i}: cluster_id={cid} | points={counts[cid]} ({pct:.1f}%) | normal≈ {n}")

# ==============================
# VISUALIZE
# ==============================
o3d.visualization.draw_geometries([pcd_ds])
