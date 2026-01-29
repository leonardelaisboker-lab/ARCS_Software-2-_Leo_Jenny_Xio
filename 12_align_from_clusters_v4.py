import open3d as o3d
import numpy as np
import os

SCAN_IN = "cube_only.ply"
SCAN_OUT = "cube_only_xyz_aligned.ply"

if not os.path.exists(SCAN_IN):
    raise FileNotFoundError(f"Missing {SCAN_IN}")

# ====== FIXED FROM YOUR REPORT ======
TOP_CID = 4   # Face 0: cluster_id=4 normal≈ [0.0135 0.0443 0.9989]
SIDE_CID = 2  # Face 1: cluster_id=2 normal≈ [0.9586 -0.2743 0.0761]

VOXEL_SIZE = 0.002
NORMAL_RADIUS = 0.01
K = 6

def kmeans(X, k, iters=60, seed=7):
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
    N2 = N.copy()
    for i in range(len(N2)):
        n = N2[i]
        if n[2] < 0 or (abs(n[2]) < 1e-9 and n[1] < 0) or (abs(n[2]) < 1e-9 and abs(n[1]) < 1e-9 and n[0] < 0):
            N2[i] = -n
    return N2

def make_orthonormal_basis(z_axis, x_hint):
    z = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    x = x_hint - np.dot(x_hint, z) * z
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    return x, y, z

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

labels, centers = kmeans(N_fold, K, iters=60, seed=7)

if TOP_CID < 0 or TOP_CID >= K or SIDE_CID < 0 or SIDE_CID >= K:
    raise ValueError(f"Cluster IDs must be within 0..{K-1}. You set TOP_CID={TOP_CID}, SIDE_CID={SIDE_CID}")

z_axis = centers[TOP_CID]
x_axis = centers[SIDE_CID]

# deterministic sign
if z_axis[2] < 0:
    z_axis = -z_axis
if x_axis[0] < 0:
    x_axis = -x_axis

x, y, z = make_orthonormal_basis(z_axis, x_axis)

print("Using TOP_CID:", TOP_CID, "Z≈", z_axis)
print("Using SIDE_CID:", SIDE_CID, "X hint≈", x_axis)
print("Basis X,Y,Z:", x, y, z)

R = np.stack([x, y, z], axis=1)
R_inv = R.T

center = pcd.get_center()
pcd.translate(-center)

pts = np.asarray(pcd.points)
pts_rot = (R_inv @ pts.T).T

pcd_aligned = o3d.geometry.PointCloud()
pcd_aligned.points = o3d.utility.Vector3dVector(pts_rot)
pcd_aligned.colors = pcd.colors
pcd_aligned.translate(center)

o3d.io.write_point_cloud(SCAN_OUT, pcd_aligned)
print("Saved:", SCAN_OUT)

obb = pcd_aligned.get_oriented_bounding_box()
obb.color = (0, 1, 0)
pcd_aligned.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd_aligned, obb])
    