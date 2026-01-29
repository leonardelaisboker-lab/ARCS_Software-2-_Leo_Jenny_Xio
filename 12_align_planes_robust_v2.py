"""
MRAC Cube Demo - Step 12 V2: Robust Alignment with Yaw Correction
===================================================================
Input:  cube_only.ply (floor already removed)
Output: cube_only_xyz_aligned.ply (perfectly aligned to XYZ axes)

Improvements over V1:
- Added histogram-based yaw correction for perfect X/Y alignment
- More robust vertical plane selection
- Better handling of edge cases

Author: MRAC Team
Date: 2025-01-26
"""

import open3d as o3d
import numpy as np
import sys

# ============================================================================
# CONFIG
# ============================================================================

INPUT_FILE = "cube_only.ply"
OUTPUT_FILE = "cube_only_xyz_aligned.ply"

# RANSAC Parameters
RANSAC_DISTANCE_THRESHOLD = 0.003  # 3mm tolerance for plane fitting
RANSAC_N_ITERATIONS = 1000
MIN_PLANE_POINTS = 500  # Minimum points to consider a valid plane

# Plane extraction
MAX_PLANES_TO_EXTRACT = 6  # Extract up to 6 planes (cube has 6 faces)

# Angle thresholds (in degrees)
VERTICAL_ANGLE_THRESHOLD = 20  # Max deviation from vertical (90° to Z-axis)
HORIZONTAL_ANGLE_THRESHOLD = 20  # Max deviation from horizontal (0° to Z-axis)

# Yaw alignment parameters
YAW_HISTOGRAM_BINS = 360  # 1 degree resolution
YAW_SEARCH_RANGE = 45  # Search ±45° around initial alignment


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def angle_between_vectors(v1, v2):
    """Calculate angle in degrees between two vectors"""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.degrees(np.arccos(np.abs(cos_angle)))


def is_horizontal_plane(normal, threshold_deg=HORIZONTAL_ANGLE_THRESHOLD):
    """Check if plane normal is approximately vertical (pointing up/down)"""
    z_axis = np.array([0, 0, 1])
    angle = angle_between_vectors(normal, z_axis)
    return angle < threshold_deg


def is_vertical_plane(normal, threshold_deg=VERTICAL_ANGLE_THRESHOLD):
    """Check if plane normal is approximately horizontal (perpendicular to Z)"""
    z_axis = np.array([0, 0, 1])
    angle = angle_between_vectors(normal, z_axis)
    return abs(angle - 90) < threshold_deg


def extract_multiple_planes(pcd, max_planes=6):
    """
    Extract multiple planes using iterative RANSAC
    Returns list of plane info dictionaries
    """
    planes = []
    remaining_pcd = pcd
    remaining_indices = np.arange(len(pcd.points))
    
    print(f"\n{'='*60}")
    print("EXTRACTING PLANES (Multi-RANSAC)")
    print(f"{'='*60}")
    
    for i in range(max_planes):
        if len(remaining_pcd.points) < MIN_PLANE_POINTS:
            print(f"\nStopping: only {len(remaining_pcd.points)} points remaining")
            break
        
        print(f"\n--- Plane {i+1} ---")
        print(f"Points remaining: {len(remaining_pcd.points)}")
        
        # Fit plane using RANSAC
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=RANSAC_N_ITERATIONS
        )
        
        if len(inliers) < MIN_PLANE_POINTS:
            print(f"  ❌ Too few inliers ({len(inliers)}), stopping")
            break
        
        # Extract plane normal and distance
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # Ensure normal points "outward" (positive Z for top, etc.)
        if is_horizontal_plane(normal):
            if normal[2] < 0:
                normal = -normal
                d = -d
        
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        global_inlier_indices = remaining_indices[inliers]
        
        # Store plane info
        plane_info = {
            'id': i,
            'model': plane_model,
            'normal': normal,
            'distance': d,
            'inlier_cloud': inlier_cloud,
            'inlier_indices': global_inlier_indices,
            'num_points': len(inliers),
            'is_horizontal': is_horizontal_plane(normal),
            'is_vertical': is_vertical_plane(normal)
        }
        
        planes.append(plane_info)
        
        print(f"  ✓ Plane found:")
        print(f"    Normal: [{normal[0]:+.6f}, {normal[1]:+.6f}, {normal[2]:+.6f}]")
        print(f"    Inliers: {len(inliers)} ({100*len(inliers)/len(remaining_pcd.points):.1f}%)")
        print(f"    Type: {'HORIZONTAL (top/bottom)' if plane_info['is_horizontal'] else 'VERTICAL (side)' if plane_info['is_vertical'] else 'UNKNOWN'}")
        
        # Remove inliers for next iteration
        outliers = list(set(range(len(remaining_pcd.points))) - set(inliers))
        remaining_pcd = remaining_pcd.select_by_index(outliers)
        remaining_indices = remaining_indices[outliers]
    
    print(f"\n{'='*60}")
    print(f"Total planes extracted: {len(planes)}")
    print(f"{'='*60}\n")
    
    return planes


def build_orthonormal_basis(planes):
    """
    Build orthonormal XYZ basis from extracted planes
    Returns initial rotation matrix (may need yaw correction)
    """
    print(f"\n{'='*60}")
    print("BUILDING ORTHONORMAL BASIS")
    print(f"{'='*60}")
    
    # Separate horizontal and vertical planes
    horizontal_planes = [p for p in planes if p['is_horizontal']]
    vertical_planes = [p for p in planes if p['is_vertical']]
    
    print(f"\nHorizontal planes: {len(horizontal_planes)}")
    print(f"Vertical planes: {len(vertical_planes)}")
    
    # Step 1: Get Z-axis from horizontal plane
    if len(horizontal_planes) == 0:
        print("⚠️  WARNING: No horizontal plane found! Using default Z=[0,0,1]")
        z_axis = np.array([0, 0, 1])
    else:
        top_plane = max(horizontal_planes, key=lambda p: p['num_points'])
        z_axis = top_plane['normal']
        print(f"\n✓ Z-axis from plane {top_plane['id']}: {z_axis}")
        print(f"  ({top_plane['num_points']} points)")
    
    # Step 2: Get X and Y axes from vertical planes
    if len(vertical_planes) < 2:
        print(f"\n⚠️  WARNING: Only {len(vertical_planes)} vertical plane(s) found!")
        print("   Using PCA fallback for X/Y axes")
        
        # Fallback: PCA
        if len(vertical_planes) > 0:
            all_vertical_points = np.vstack([p['inlier_cloud'].points for p in vertical_planes])
        else:
            all_vertical_points = np.vstack([p['inlier_cloud'].points for p in planes])
        
        centered = all_vertical_points - np.mean(all_vertical_points, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        x_axis = eigenvectors[:, 0]
        y_axis = eigenvectors[:, 1]
        
        # Ensure orthogonal to Z
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
    else:
        # Sort vertical planes by |normal_x| to find X-dominant plane
        vertical_planes_sorted_x = sorted(vertical_planes, key=lambda p: abs(p['normal'][0]), reverse=True)
        x_plane = vertical_planes_sorted_x[0]
        x_axis_raw = x_plane['normal']
        
        print(f"\n✓ X-axis candidate from plane {x_plane['id']}: {x_axis_raw}")
        print(f"  ({x_plane['num_points']} points)")
        
        # Project to be perpendicular to Z
        x_axis = x_axis_raw - np.dot(x_axis_raw, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis: cross product
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        print(f"\n✓ X-axis (orthogonalized): {x_axis}")
        print(f"✓ Y-axis (from cross product): {y_axis}")
    
    # Build rotation matrix
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # Verify orthonormality
    print(f"\n--- Basis Verification ---")
    print(f"X·Y = {np.dot(x_axis, y_axis):.6f} (should be ~0)")
    print(f"Y·Z = {np.dot(y_axis, z_axis):.6f} (should be ~0)")
    print(f"Z·X = {np.dot(z_axis, x_axis):.6f} (should be ~0)")
    print(f"Det(R) = {np.linalg.det(rotation_matrix):.6f} (should be ~1)")
    
    return rotation_matrix, z_axis


def find_optimal_yaw_angle(points_2d):
    """
    Find optimal yaw angle using histogram of edge orientations
    
    Args:
        points_2d: Points in XY plane (Nx2 array)
    
    Returns:
        optimal_angle: Yaw correction angle in radians
    """
    print(f"\n{'='*60}")
    print("YAW CORRECTION (Histogram-based)")
    print(f"{'='*60}")
    
    # Compute edge orientations using convex hull
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
        
        # Compute edge angles
        edge_angles = []
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            edge = p2 - p1
            angle = np.arctan2(edge[1], edge[0])
            edge_angles.append(angle)
        
        edge_angles = np.array(edge_angles)
        
    except Exception as e:
        print(f"⚠️  ConvexHull failed: {e}")
        print("   Using grid-based approach instead")
        
        # Fallback: use all point pairs
        # Sample subset to avoid too many computations
        n_samples = min(1000, len(points_2d))
        indices = np.random.choice(len(points_2d), n_samples, replace=False)
        sampled_points = points_2d[indices]
        
        edge_angles = []
        for i in range(0, len(sampled_points)-1, 10):
            p1 = sampled_points[i]
            p2 = sampled_points[i+1]
            edge = p2 - p1
            angle = np.arctan2(edge[1], edge[0])
            edge_angles.append(angle)
        
        edge_angles = np.array(edge_angles)
    
    # Normalize angles to [0, π/2] (cube has 4-fold symmetry)
    edge_angles_normalized = np.mod(edge_angles, np.pi/2)
    
    # Create histogram
    hist, bin_edges = np.histogram(
        np.degrees(edge_angles_normalized), 
        bins=YAW_HISTOGRAM_BINS//4,
        range=(0, 90)
    )
    
    # Find dominant angle
    peak_bin = np.argmax(hist)
    dominant_angle_deg = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
    
    # Convert to rotation angle (align dominant angle to 0°)
    yaw_correction_deg = -dominant_angle_deg
    yaw_correction_rad = np.radians(yaw_correction_deg)
    
    print(f"\nDominant edge angle: {dominant_angle_deg:.2f}°")
    print(f"Yaw correction: {yaw_correction_deg:.2f}°")
    print(f"Histogram peak strength: {hist[peak_bin]} edges")
    
    return yaw_correction_rad


def align_cube_to_axes(pcd, rotation_matrix, z_axis):
    """
    Align point cloud using rotation matrix + yaw correction
    """
    print(f"\n{'='*60}")
    print("APPLYING ALIGNMENT TRANSFORMATION")
    print(f"{'='*60}")
    
    # Get points
    points = np.asarray(pcd.points)
    
    # Center before rotation
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    print(f"\nOriginal centroid: [{centroid[0]*1000:.2f}, {centroid[1]*1000:.2f}, {centroid[2]*1000:.2f}] mm")
    
    # Apply initial rotation
    R_inv = rotation_matrix.T
    points_aligned = points_centered @ R_inv
    
    print(f"✓ Initial rotation applied")
    
    # Step 2: Yaw correction using histogram
    points_xy = points_aligned[:, :2]  # Project to XY plane
    
    try:
        yaw_correction = find_optimal_yaw_angle(points_xy)
        
        # Create yaw rotation matrix (around Z-axis)
        cos_yaw = np.cos(yaw_correction)
        sin_yaw = np.sin(yaw_correction)
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        
        # Apply yaw correction
        points_aligned = points_aligned @ R_yaw.T
        
        print(f"✓ Yaw correction applied: {np.degrees(yaw_correction):.2f}°")
        
    except Exception as e:
        print(f"⚠️  Yaw correction failed: {e}")
        print("   Continuing without yaw correction")
    
    # Update point cloud
    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(points_aligned)
    
    # Copy colors
    if pcd.has_colors():
        pcd_aligned.colors = pcd.colors
    
    # Recompute normals
    pcd_aligned.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    
    print(f"✓ Cube centered at origin")
    
    return pcd_aligned


def verify_alignment(pcd):
    """
    Verify alignment by checking bounding box
    """
    print(f"\n{'='*60}")
    print("ALIGNMENT VERIFICATION")
    print(f"{'='*60}")
    
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    
    print(f"\nAxis-Aligned Bounding Box:")
    print(f"  X extent: {extent[0]*1000:.2f} mm")
    print(f"  Y extent: {extent[1]*1000:.2f} mm")
    print(f"  Z extent: {extent[2]*1000:.2f} mm")
    
    mean_extent = np.mean(extent)
    deviations = extent - mean_extent
    
    print(f"\nCubicity check (deviations from mean {mean_extent*1000:.2f} mm):")
    print(f"  X deviation: {deviations[0]*1000:+.2f} mm")
    print(f"  Y deviation: {deviations[1]*1000:+.2f} mm")
    print(f"  Z deviation: {deviations[2]*1000:+.2f} mm")
    
    max_deviation = np.max(np.abs(deviations))
    if max_deviation < 0.005:
        print(f"\n✓ GOOD: Max deviation {max_deviation*1000:.2f} mm < 5mm")
        return True
    else:
        print(f"\n⚠️  WARNING: Max deviation {max_deviation*1000:.2f} mm > 5mm")
        return False


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print(f"\n{'='*60}")
    print("MRAC CUBE DEMO - STEP 12 V2: ALIGNMENT WITH YAW CORRECTION")
    print(f"{'='*60}\n")
    
    # 1. Load cube
    print(f"Loading: {INPUT_FILE}")
    pcd = o3d.io.read_point_cloud(INPUT_FILE)
    print(f"✓ Loaded {len(pcd.points)} points\n")
    
    if len(pcd.points) == 0:
        print("❌ ERROR: Point cloud is empty!")
        sys.exit(1)
    
    # 2. Extract planes
    planes = extract_multiple_planes(pcd, max_planes=MAX_PLANES_TO_EXTRACT)
    
    if len(planes) < 2:
        print("❌ ERROR: Not enough planes found!")
        sys.exit(1)
    
    # 3. Build basis
    rotation_matrix, z_axis = build_orthonormal_basis(planes)
    
    # 4. Align with yaw correction
    pcd_aligned = align_cube_to_axes(pcd, rotation_matrix, z_axis)
    
    # 5. Verify
    is_good = verify_alignment(pcd_aligned)
    
    # 6. Save
    print(f"\n{'='*60}")
    print(f"Saving aligned cube to: {OUTPUT_FILE}")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_aligned)
    print(f"✓ Saved successfully!")
    print(f"{'='*60}\n")
    
    # 7. Visualize
    print("Visualizing result...")
    print("  Red box = Axis-aligned bounding box")
    print("  RGB axes = XYZ coordinate system")
    print("  → Bounding box should fit TIGHTLY on all 6 faces!\n")
    
    aabb = pcd_aligned.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=[0, 0, 0]
    )
    
    o3d.visualization.draw_geometries(
        [pcd_aligned, aabb, coord_frame],
        window_name="Step 12 V2: Aligned Cube (with Yaw Correction)",
        width=1920,
        height=1080
    )
    
    if is_good:
        print("\n✅ ALIGNMENT SUCCESSFUL!")
        print("   → Ready for Step 13 (Digital Twin comparison)")
    else:
        print("\n⚠️  Alignment needs improvement")
        print("   → Check if cube has clear faces or is too noisy")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()