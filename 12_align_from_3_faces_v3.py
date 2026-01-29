import open3d as o3d
import numpy as np
import os

# ==============================
# INPUT / OUTPUT
# ==============================
SCAN_IN = "cube_only.ply"
SCAN_OUT = "cube_only_xyz_aligned.ply"

if not os.path.exists(SCAN_IN):
    raise FileNotFoundError(f"Missing {SCAN_IN}")

# ==============================
# PARAMS
# ==============================
VOXEL_SIZE = 0.002
NORMAL_RADIUS = 0.01
K = 6  # normal clusters

# ==============================
# HELPERS
# ==============================
def kmeans(X, k, iters=50, seed=7):
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
    # fold to consistent hemisphere to stabilize clustering
    N2 = N.copy()
    for i in range(len(N2)):
        n = N2[i]
        if n[2] < 0 or (abs(n[2]) < 1e-9 and n[1] < 0) or (abs(n[2]) < 1e-9 and abs(n[1]) < 1e-9 and n[0] < 0):
            N2[i] = -n
    return N2

def make_orthonormal_basis(z_axis, x_hint):
    z = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    # remove any component of x along z
    x = x_hint - np.dot(x_hint, z) * z
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    # re-orthogonalize x (for numerical safety)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    return x, y, z

# ==============================
# LOAD + NORMALS
# ==============================
pcd = o3d.io.read_point_cloud(SCAN_IN)
if pcd.is_empty():
    raise RuntimeError("Input point cloud is empty.")

pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)
pcd_ds.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=50)
)
pcd_ds.orient_normals_consistent_tangent_plane(50)

N = np.asarray(pcd_ds.normals)
N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
N_fold = fold_normals(N)

# ==============================
# CLUSTER NORMALS
# ==============================
labels, centers = kmeans(N_fold, K, iters=60, seed=7)
counts = np.bincount(labels, minlength=K)

# rank clusters by size (ignore tiny junk)
order = np.argsort(counts)[::-1]

# ==============================
# PICK TOP FACE (Z) + SIDE FACE (X)
# ==============================
# We pick among the largest clusters to avoid noise:
candidates = order[:min(6, K)]

# Top face: |nz| max
top_cid = max(candidates, key=lambda cid: abs(centers[cid][2]))
z_axis = centers[top_cid]
if z_axis[2] < 0:
    z_axis = -z_axis

# Side face: among candidates, choose minimal |nz| but still large enough
side_candidates = [cid for cid in candidates if cid != top_cid]
side_cid = min(side_candidates, key=lambda cid: abs(centers[cid][2]))
x_axis = centers[side_cid]
# Make X point "positive-ish" for determinism
if x_axis[0] < 0:
    x_axis = -x_axis

x, y, z = make_orthonormal_basis(z_axis, x_axis)

print("Chosen top cluster:", top_cid, "normal(Z)≈", z_axis, "points:", int(counts[top_cid]))
print("Chosen side cluster:", side_cid, "normal(X hint)≈", x_axis, "points:", int(counts[side_cid]))
print("Basis X,Y,Z:", x, y, z)

# Rotation matrix that maps (X,Y,Z) basis to world axes:
# If columns are basis vectors in world coords, then R_world_from_local = [x y z]
# We want to rotate points so that basis becomes identity -> use R_local_from_world = (R_world_from_local)^T
R = np.stack([x, y, z], axis=1)          # 3x3
R_inv = R.T                               # rotate points into this basis

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

# Save + show bbox
o3d.io.write_point_cloud(SCAN_OUT, pcd_aligned)
print("Saved:", SCAN_OUT)

obb = pcd_aligned.get_oriented_bounding_box()
obb.color = (0, 1, 0)

pcd_aligned.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd_aligned, obb])
