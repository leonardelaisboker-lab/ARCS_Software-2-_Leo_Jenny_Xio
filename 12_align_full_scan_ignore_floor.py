"""
MRAC Cube Demo - Complete Pipeline: From Raw Scan to Aligned Colored Cube
===========================================================================
Input:  Earth_Cube_Scan_Test.ply (original scan WITH floor)
Output: cube_aligned_colored.ply (aligned cube, floor removed)
        full_scan_with_colored_cube.ply (optional: floor + colored cube)

Strategy:
1. Load original scan (with floor)
2. Detect and separate floor using RANSAC
3. Isolate cube using DBSCAN clustering
4. Align cube (ignore floor for alignment!)
5. Color cube faces
6. Optional: Combine colored cube back with floor for visualization

Author: MRAC Team
Date: 2025-01-26
"""

import open3d as o3d
import numpy as np
import sys

# ============================================================================
# CONFIG
# ============================================================================

INPUT_FILE = "Earth_Cube_Scan_Test.ply"  # Original scan WITH floor
OUTPUT_CUBE_ONLY = "cube_aligned_colored.ply"
OUTPUT_WITH_FLOOR = "full_scan_with_colored_cube.ply"

# Floor detection (RANSAC)
FLOOR_RANSAC_DISTANCE = 0.01  # 10mm tolerance
FLOOR_RANSAC_ITERATIONS = 1000
FLOOR_MIN_POINTS = 1000

# Cube isolation (DBSCAN)
DBSCAN_EPS = 0.005  # 5mm clustering radius
DBSCAN_MIN_POINTS = 10

# Cube plane extraction (RANSAC)
CUBE_RANSAC_DISTANCE = 0.003  # 3mm tolerance
CUBE_RANSAC_ITERATIONS = 1000
MIN_PLANE_POINTS = 500
MAX_PLANES_TO_EXTRACT = 6

# Alignment thresholds
VERTICAL_ANGLE_THRESHOLD = 20
HORIZONTAL_ANGLE_THRESHOLD = 20

# Face colors
FACE_COLORS = {
    'top': [1.0, 0.2, 0.2],      # Red
    'bottom': [0.2, 1.0, 0.2],   # Green
    'front': [0.2, 0.2, 1.0],    # Blue
    'back': [1.0, 1.0, 0.2],     # Yellow
    'left': [1.0, 0.2, 1.0],     # Magenta
    'right': [0.2, 1.0, 1.0]     # Cyan
}

FLOOR_COLOR = [0.6, 0.6, 0.6]  # Gray for floor


# ============================================================================
# STEP 1: FLOOR DETECTION & REMOVAL
# ============================================================================

def detect_and_remove_floor(pcd):
    """
    Detect floor plane using RANSAC and separate cube from floor
    Returns: cube_pcd, floor_pcd
    """
    print(f"\n{'='*60}")
    print("STEP 1: FLOOR DETECTION")
    print(f"{'='*60}")
    
    print(f"\nTotal points: {len(pcd.points)}")
    
    # Detect floor plane
    print("Detecting floor plane (RANSAC)...")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=FLOOR_RANSAC_DISTANCE,
        ransac_n=3,
        num_iterations=FLOOR_RANSAC_ITERATIONS
    )
    
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    print(f"\n✓ Floor plane detected:")
    print(f"  Normal: [{normal[0]:+.4f}, {normal[1]:+.4f}, {normal[2]:+.4f}]")
    print(f"  Inliers: {len(inliers)} ({100*len(inliers)/len(pcd.points):.1f}%)")
    
    # Separate floor and non-floor points
    floor_pcd = pcd.select_by_index(inliers)
    non_floor_pcd = pcd.select_by_index(inliers, invert=True)
    
    print(f"\nPoints after floor removal: {len(non_floor_pcd.points)}")
    
    return non_floor_pcd, floor_pcd, plane_model


def isolate_cube_cluster(pcd):
    """
    Use DBSCAN to find largest cluster (= cube)
    """
    print(f"\n{'='*60}")
    print("STEP 2: CUBE ISOLATION (DBSCAN)")
    print(f"{'='*60}")
    
    print(f"\nClustering with DBSCAN (eps={DBSCAN_EPS*1000}mm)...")
    
    labels = np.array(pcd.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=False
    ))
    
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters")
    
    if max_label < 0:
        print("⚠️  WARNING: No clusters found! Using all points")
        return pcd
    
    # Find largest cluster
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    largest_cluster_size = np.max(counts)
    
    print(f"\n✓ Largest cluster (cube):")
    print(f"  Label: {largest_cluster_label}")
    print(f"  Points: {largest_cluster_size}")
    
    # Extract largest cluster
    cube_indices = np.where(labels == largest_cluster_label)[0]
    cube_pcd = pcd.select_by_index(cube_indices)
    
    return cube_pcd


# ============================================================================
# STEP 3: PLANE EXTRACTION (same as before)
# ============================================================================

def angle_between_vectors(v1, v2):
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.degrees(np.arccos(np.abs(cos_angle)))


def is_horizontal_plane(normal):
    z_axis = np.array([0, 0, 1])
    angle = angle_between_vectors(normal, z_axis)
    return angle < HORIZONTAL_ANGLE_THRESHOLD


def is_vertical_plane(normal):
    z_axis = np.array([0, 0, 1])
    angle = angle_between_vectors(normal, z_axis)
    return abs(angle - 90) < VERTICAL_ANGLE_THRESHOLD


def extract_cube_planes(pcd):
    """Extract planes from cube only (floor already removed)"""
    print(f"\n{'='*60}")
    print("STEP 3: CUBE PLANE EXTRACTION")
    print(f"{'='*60}")
    
    planes = []
    remaining_pcd = pcd
    remaining_indices = np.arange(len(pcd.points))
    
    for i in range(MAX_PLANES_TO_EXTRACT):
        if len(remaining_pcd.points) < MIN_PLANE_POINTS:
            break
        
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=CUBE_RANSAC_DISTANCE,
            ransac_n=3,
            num_iterations=CUBE_RANSAC_ITERATIONS
        )
        
        if len(inliers) < MIN_PLANE_POINTS:
            break
        
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        if is_horizontal_plane(normal) and normal[2] < 0:
            normal = -normal
            d = -d
        
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        
        planes.append({
            'id': i,
            'normal': normal,
            'distance': d,
            'num_points': len(inliers),
            'is_horizontal': is_horizontal_plane(normal),
            'is_vertical': is_vertical_plane(normal)
        })
        
        print(f"Plane {i+1}: {len(inliers)} points | " + 
              f"Normal=[{normal[0]:+.3f}, {normal[1]:+.3f}, {normal[2]:+.3f}]")
        
        outliers = list(set(range(len(remaining_pcd.points))) - set(inliers))
        remaining_pcd = remaining_pcd.select_by_index(outliers)
        remaining_indices = remaining_indices[outliers]
    
    print(f"\n✓ Extracted {len(planes)} planes")
    return planes


# ============================================================================
# STEP 4: ALIGNMENT (same as before)
# ============================================================================

def build_orthonormal_basis(planes):
    """Build XYZ basis from planes"""
    print(f"\n{'='*60}")
    print("STEP 4: BUILDING ORTHONORMAL BASIS")
    print(f"{'='*60}")
    
    horizontal_planes = [p for p in planes if p['is_horizontal']]
    vertical_planes = [p for p in planes if p['is_vertical']]
    
    # Z-axis from horizontal plane
    if len(horizontal_planes) > 0:
        top_plane = max(horizontal_planes, key=lambda p: p['num_points'])
        z_axis = top_plane['normal']
    else:
        z_axis = np.array([0, 0, 1])
    
    # X and Y from vertical planes
    if len(vertical_planes) >= 2:
        vertical_planes_sorted = sorted(vertical_planes, key=lambda p: abs(p['normal'][0]), reverse=True)
        x_axis_raw = vertical_planes_sorted[0]['normal']
        
        x_axis = x_axis_raw - np.dot(x_axis_raw, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
    else:
        # Fallback to default
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
    
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    print(f"✓ Basis constructed (Det={np.linalg.det(rotation_matrix):.4f})")
    
    return rotation_matrix


def find_yaw_correction(points_2d):
    """Find yaw correction angle"""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
        
        edge_angles = []
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            edge = p2 - p1
            angle = np.arctan2(edge[1], edge[0])
            edge_angles.append(angle)
        
        edge_angles = np.array(edge_angles)
        edge_angles_norm = np.mod(edge_angles, np.pi/2)
        
        hist, bin_edges = np.histogram(np.degrees(edge_angles_norm), bins=90, range=(0, 90))
        peak_bin = np.argmax(hist)
        dominant_angle = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
        
        return np.radians(-dominant_angle)
    except:
        return 0.0


def align_cube(pcd, rotation_matrix):
    """Apply rotation and yaw correction"""
    print(f"\n{'='*60}")
    print("STEP 5: ALIGNMENT TRANSFORMATION")
    print(f"{'='*60}")
    
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # Initial rotation
    R_inv = rotation_matrix.T
    points_aligned = points_centered @ R_inv
    
    # Yaw correction
    points_xy = points_aligned[:, :2]
    yaw_correction = find_yaw_correction(points_xy)
    
    if abs(yaw_correction) > 0.01:  # Only apply if significant
        cos_yaw = np.cos(yaw_correction)
        sin_yaw = np.sin(yaw_correction)
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        points_aligned = points_aligned @ R_yaw.T
        print(f"✓ Yaw correction: {np.degrees(yaw_correction):.2f}°")
    
    # Create new point cloud
    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(points_aligned)
    
    if pcd.has_colors():
        pcd_aligned.colors = pcd.colors
    
    pcd_aligned.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    
    return pcd_aligned


# ============================================================================
# STEP 6: FACE COLORING
# ============================================================================

def color_cube_faces(pcd):
    """Color each face based on position"""
    print(f"\n{'='*60}")
    print("STEP 6: COLORING CUBE FACES")
    print(f"{'='*60}")
    
    points = np.asarray(pcd.points)
    colors = np.zeros((len(points), 3))
    
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    
    extent = max_bound - min_bound
    tolerance = 0.1 * np.min(extent)
    
    face_counts = {key: 0 for key in FACE_COLORS.keys()}
    
    for i, point in enumerate(points):
        assigned = False
        
        if point[2] > max_bound[2] - tolerance:
            colors[i] = FACE_COLORS['top']
            face_counts['top'] += 1
            assigned = True
        elif point[2] < min_bound[2] + tolerance:
            colors[i] = FACE_COLORS['bottom']
            face_counts['bottom'] += 1
            assigned = True
        elif point[1] > max_bound[1] - tolerance:
            colors[i] = FACE_COLORS['front']
            face_counts['front'] += 1
            assigned = True
        elif point[1] < min_bound[1] + tolerance:
            colors[i] = FACE_COLORS['back']
            face_counts['back'] += 1
            assigned = True
        elif point[0] > max_bound[0] - tolerance:
            colors[i] = FACE_COLORS['right']
            face_counts['right'] += 1
            assigned = True
        elif point[0] < min_bound[0] + tolerance:
            colors[i] = FACE_COLORS['left']
            face_counts['left'] += 1
            assigned = True
        
        if not assigned:
            colors[i] = [0.5, 0.5, 0.5]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print("\nFace assignment:")
    for face, count in face_counts.items():
        print(f"  {face:8s}: {count:5d} points")
    
    return pcd


# ============================================================================
# STEP 7: MEASUREMENTS
# ============================================================================

def measure_cube(pcd):
    """Measure and display cube dimensions"""
    print(f"\n{'='*60}")
    print("STEP 7: CUBE MEASUREMENTS")
    print(f"{'='*60}")
    
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    
    print(f"\nDimensions:")
    print(f"  X: {extent[0]*1000:.2f} mm")
    print(f"  Y: {extent[1]*1000:.2f} mm")
    print(f"  Z: {extent[2]*1000:.2f} mm")
    
    mean = np.mean(extent)
    print(f"\nMean edge: {mean*1000:.2f} mm")
    
    volume = extent[0] * extent[1] * extent[2] * 1e9
    print(f"Volume: {volume:.2f} mm³")
    
    return extent


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print(f"\n{'='*60}")
    print("MRAC CUBE DEMO - COMPLETE PIPELINE")
    print("From Raw Scan to Aligned Colored Cube")
    print(f"{'='*60}\n")
    
    # Load original scan (with floor)
    print(f"Loading: {INPUT_FILE}")
    pcd_full = o3d.io.read_point_cloud(INPUT_FILE)
    print(f"✓ Loaded {len(pcd_full.points)} points (including floor)")
    
    # Step 1: Remove floor
    pcd_no_floor, pcd_floor, floor_model = detect_and_remove_floor(pcd_full)
    
    # Step 2: Isolate cube
    pcd_cube = isolate_cube_cluster(pcd_no_floor)
    
    # Step 3: Extract planes (only from cube!)
    planes = extract_cube_planes(pcd_cube)
    
    # Step 4: Build basis
    rotation_matrix = build_orthonormal_basis(planes)
    
    # Step 5: Align cube
    pcd_cube_aligned = align_cube(pcd_cube, rotation_matrix)
    
    # Step 6: Color faces
    pcd_cube_colored = color_cube_faces(pcd_cube_aligned)
    
    # Step 7: Measure
    extent = measure_cube(pcd_cube_colored)
    
    # Save cube only
    print(f"\n{'='*60}")
    print(f"Saving cube: {OUTPUT_CUBE_ONLY}")
    o3d.io.write_point_cloud(OUTPUT_CUBE_ONLY, pcd_cube_colored)
    print(f"✓ Saved!")
    
    # Optional: Save with floor (floor in gray)
    print(f"\nSaving full scan with colored cube: {OUTPUT_WITH_FLOOR}")
    pcd_floor.paint_uniform_color(FLOOR_COLOR)
    pcd_combined = pcd_cube_colored + pcd_floor
    o3d.io.write_point_cloud(OUTPUT_WITH_FLOOR, pcd_combined)
    print(f"✓ Saved!")
    print(f"{'='*60}\n")
    
    # Visualize
    print("Opening visualization...")
    print("\n🎨 Color Legend:")
    print("  🔴 Red    = Top    |  🟡 Yellow  = Back")
    print("  🟢 Green  = Bottom |  🟣 Magenta = Left")
    print("  🔵 Blue   = Front  |  🔷 Cyan    = Right")
    print(f"\n📏 Dimensions:")
    print(f"  X: {extent[0]*1000:.2f} mm")
    print(f"  Y: {extent[1]*1000:.2f} mm")
    print(f"  Z: {extent[2]*1000:.2f} mm\n")
    
    aabb = pcd_cube_colored.get_axis_aligned_bounding_box()
    aabb.color = (0.3, 0.3, 0.3)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.03, origin=[0, 0, 0]
    )
    
    # Show cube only (without floor)
    o3d.visualization.draw_geometries(
        [pcd_cube_colored, aabb, coord_frame],
        window_name="Aligned Cube - Colored Faces (Floor removed for alignment)",
        width=1920,
        height=1080
    )
    
    print(f"\n{'='*60}")
    print("✅ COMPLETE PIPELINE FINISHED!")
    print(f"{'='*60}")
    print(f"\n📁 Output files:")
    print(f"  1. {OUTPUT_CUBE_ONLY} (aligned colored cube)")
    print(f"  2. {OUTPUT_WITH_FLOOR} (cube + floor)")
    print(f"\n🎯 Next: Use {OUTPUT_CUBE_ONLY} for digital twin comparison!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()