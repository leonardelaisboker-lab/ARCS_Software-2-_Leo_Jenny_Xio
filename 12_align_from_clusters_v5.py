import open3d as o3d
import numpy as np
import os

# ==============================
# FILES
# ==============================
SCAN_IN = "cube_only.ply"
SCAN_OUT = "cube_only_xyz_aligned_v5.ply"

if not os.path.exists(SCAN_IN):
    raise FileNotFoundError(f"Missing {SCAN_IN} in the current folder.")

# ==============================
# CLUSTER IDS (from your FACE CLUSTER REPORT)
# ==============================
CID_Z = 4  # Face 0: cluster_id=4, normal ~ [0.0135, 0.0443, 0.9989]
CID_X = 2  # Face 1: cluster_id=2, normal ~ [0.9586, -0.2743, 0.0761]
CID_Y = 5  # Face 4: cluster_id=5, normal ~ [0.2551, 0.9614, 0.1029]

# ==============================
# PARAMS
# ==============================
VOXEL_SIZE = 0.002
NORMAL_RADIUS = 0.01
K = 6

# ==============================
# HELPERS
# ==============================
def kmeans(X, k, iters=60, seed=7):
    """Simple k-means without sklearn. X shape (N,3)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    C = X[idx].copy()

    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)

        C_new = np.zeros_like(C)
        for j in range(k):
            pts = X[labels == j]
            if len(pts) == 0:
                C_new[j] = C[j]
            else:
                C_new[j] = pts.mean(axis=0)

        C_new = C_new / (np.linalg.norm(C_new, axis=1, keepdims=True) + 1e-9)

        if np.allclose(C, C_new, atol=1e-4):
            C = C_new
            break
        C = C_new

    return labels, C

def fold_normals(N):
    """
    Fold normals into a consistent hemisphere (+Z preferred) to stabilize clustering.
    """
    N2 = N.copy()
    for i in range(len(N2)):
        n = N2[i]
        if n[2] < 0 or (abs(n[2]) < 1e-9 and n[1] < 0) or (abs(n[2]) < 1e-9 and abs(n[1]) < 1e-9 and n[0] < 0):
            N2[i] = -n
    return N2

def safe_unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise RuntimeError("Zero-length vector encountered in normalization.")
    return v / n

# ==============================
# LOAD + NORMALS
# ==============================
pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError("Input point cloud is empty.")

pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)
if pcd_ds.is_empty():
    raise RuntimeError("Downsample produced an empty cloud. Reduce VOXEL_SIZE.")

pcd_ds.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=50)
)
pcd_ds.orient_normals_consistent_tangent_plane(50)

N = np.asarray(pcd_ds.normals)
N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
N_fold = fold_normals(N)

# ==============================
# CLUSTER NORMALS -> CENTERS
# ==============================
labels, centers = kmeans(N_fold, K, iters=70, seed=7)

# Validate IDs
for cid in (CID_X, CID_Y, CID_Z):
    if cid < 0 or cid >= K:
        raise ValueError(f"Cluster id {cid} out of range 0..{K-1}")

# ==============================
# BUILD ORTHONORMAL XYZ FROM 3 FACES
# ==============================
z = safe_unit(centers[CID_Z])
x_hint = safe_unit(centers[CID_X])
y_hint = safe_unit(centers[CID_Y])

# Force Z to point upward (positive Z)
if z[2] < 0:
    z = -z

# Create orthogonal basis:
# 1) Start with X (project out component along Z)
x = x_hint - np.dot(x_hint, z) * z
x = safe_unit(x)

# 2) Y from cross (right-handed)
y = np.cross(z, x)
y = safe_unit(y)

# 3) Recompute X again to be perfectly orthogonal
x = np.cross(y, z)
x = safe_unit(x)

# Deterministic axis direction:
# prefer +X and +Y as "positive-ish"
if x[0] < 0:
    x = -x
    y = -y  # keep right-handedness
# Make Y roughly align with y_hint (optional sign fix)
if np.dot(y, y_hint) < 0:
    y = -y
    x = -x  # keep right-handedness

print("=== v5 ALIGNMENT BASIS ===")
print("Z (from CID_Z):", CID_Z, "->", z)
print("X (from CID_X):", CID_X, "->", x)
print("Y (computed):        ->", y)
print("Dot checks (should be ~0): x·y=", float(np.dot(x, y)), "x·z=", float(np.dot(x, z)), "y·z=", float(np.dot(y, z)))

# Rotation matrix: columns are basis vectors in original space.
# To rotate points into this basis, use transpose.
R = np.stack([x, y, z], axis=1)   # 3x3
R_inv = R.T

# ==============================
# APPLY ROTATION ABOUT CENTER
# ==============================
center = pcd.get_center()
pcd.translate(-center)

pts = np.asarray(pcd.points)
pts_rot = (R_inv @ pts.T).T

pcd_aligned = o3d.geometry.PointCloud()
pcd_aligned.points = o3d.utility.Vector3dVector(pts_rot)
pcd_aligned.colors = pcd.colors
pcd_aligned.translate(center)

# Save
ok = o3d.io.write_point_cloud(SCAN_OUT, pcd_aligned)
print("Saved:", SCAN_OUT, "| ok:", ok)

# Show with OBB + extents
obb = pcd_aligned.get_oriented_bounding_box()
obb.color = (0, 1, 0)

extent_mm = np.array(obb.extent) * 1000.0
print("OBB extent (mm):", extent_mm)
print("OBB mean edge (mm):", float(extent_mm.mean()))

pcd_aligned.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd_aligned, obb])
