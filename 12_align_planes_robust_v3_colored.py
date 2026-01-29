"""
MRAC Cube Demo - Step 12 V3: Alignment with Face Coloring & Measurements
==========================================================================
Input:  cube_only.ply (floor already removed)
Output: cube_only_xyz_aligned_colored.ply (aligned + colored by face)

New Features:
- Each cube face gets a different color
- Dimensions displayed in visualization
- Enhanced visual feedback

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
OUTPUT_FILE = "cube_only_xyz_aligned_colored.ply"

# RANSAC Parameters
RANSAC_DISTANCE_THRESHOLD = 0.003
RANSAC_N_ITERATIONS = 1000
MIN_PLANE_POINTS = 500

# Plane extraction
MAX_PLANES_TO_EXTRACT = 6

# Angle thresholds (in degrees)
VERTICAL_ANGLE_THRESHOLD = 20
HORIZONTAL_ANGLE_THRESHOLD = 20

# Yaw alignment
YAW_HISTOGRAM_BINS = 360

# Face colors (RGB) - distinct colors for each face
FACE_COLORS = {
    'top': [1.0, 0.2, 0.2],      # Red
    'bottom': [0.2, 1.0, 0.2],   # Green
    'front': [0.2, 0.2, 1.0],    # Blue
    'back': [1.0, 1.0, 0.2],     # Yellow
    'left': [1.0, 0.2, 1.0],     # Magenta
    'right': [0.2, 1.0, 1.0]     # Cyan
}


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
    """Extract multiple planes using iterative RANSAC"""
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
        
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=RANSAC_N_ITERATIONS
        )
        
        if len(inliers) < MIN_PLANE_POINTS:
            print(f"  ❌ Too few inliers ({len(inliers)}), stopping")
            break
        
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        if is_horizontal_plane(normal):
            if normal[2] < 0:
                normal = -normal
                d = -d
        
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        global_inlier_indices = remaining_indices[inliers]
        
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
        
        outliers = list(set(range(len(remaining_pcd.points))) - set(inliers))
        remaining_pcd = remaining_pcd.select_by_index(outliers)
        remaining_indices = remaining_indices[outliers]
    
    print(f"\n{'='*60}")
    print(f"Total planes extracted: {len(planes)}")
    print(f"{'='*60}\n")
    
    return planes


def build_orthonormal_basis(planes):
    """Build orthonormal XYZ basis from extracted planes"""
    print(f"\n{'='*60}")
    print("BUILDING ORTHONORMAL BASIS")
    print(f"{'='*60}")
    
    horizontal_planes = [p for p in planes if p['is_horizontal']]
    vertical_planes = [p for p in planes if p['is_vertical']]
    
    print(f"\nHorizontal planes: {len(horizontal_planes)}")
    print(f"Vertical planes: {len(vertical_planes)}")
    
    if len(horizontal_planes) == 0:
        print("⚠️  WARNING: No horizontal plane found! Using default Z=[0,0,1]")
        z_axis = np.array([0, 0, 1])
    else:
        top_plane = max(horizontal_planes, key=lambda p: p['num_points'])
        z_axis = top_plane['normal']
        print(f"\n✓ Z-axis from plane {top_plane['id']}: {z_axis}")
    
    if len(vertical_planes) < 2:
        print(f"\n⚠️  WARNING: Only {len(vertical_planes)} vertical plane(s)!")
        
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
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
    else:
        vertical_planes_sorted_x = sorted(vertical_planes, key=lambda p: abs(p['normal'][0]), reverse=True)
        x_plane = vertical_planes_sorted_x[0]
        x_axis_raw = x_plane['normal']
        
        print(f"\n✓ X-axis candidate from plane {x_plane['id']}: {x_axis_raw}")
        
        x_axis = x_axis_raw - np.dot(x_axis_raw, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
    
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    print(f"\n--- Basis Verification ---")
    print(f"Det(R) = {np.linalg.det(rotation_matrix):.6f}")
    
    return rotation_matrix, z_axis


def find_optimal_yaw_angle(points_2d):
    """Find optimal yaw angle using histogram of edge orientations"""
    print(f"\n{'='*60}")
    print("YAW CORRECTION")
    print(f"{'='*60}")
    
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
        
    except:
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
    
    edge_angles_normalized = np.mod(edge_angles, np.pi/2)
    
    hist, bin_edges = np.histogram(
        np.degrees(edge_angles_normalized), 
        bins=YAW_HISTOGRAM_BINS//4,
        range=(0, 90)
    )
    
    peak_bin = np.argmax(hist)
    dominant_angle_deg = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
    
    yaw_correction_deg = -dominant_angle_deg
    yaw_correction_rad = np.radians(yaw_correction_deg)
    
    print(f"\nYaw correction: {yaw_correction_deg:.2f}°")
    
    return yaw_correction_rad


def align_cube_to_axes(pcd, rotation_matrix, z_axis):
    """Align point cloud using rotation matrix + yaw correction"""
    print(f"\n{'='*60}")
    print("APPLYING ALIGNMENT")
    print(f"{'='*60}")
    
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    R_inv = rotation_matrix.T
    points_aligned = points_centered @ R_inv
    
    print(f"✓ Initial rotation applied")
    
    points_xy = points_aligned[:, :2]
    
    try:
        yaw_correction = find_optimal_yaw_angle(points_xy)
        
        cos_yaw = np.cos(yaw_correction)
        sin_yaw = np.sin(yaw_correction)
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        
        points_aligned = points_aligned @ R_yaw.T
        print(f"✓ Yaw correction applied")
        
    except Exception as e:
        print(f"⚠️  Yaw correction skipped")
    
    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(points_aligned)
    
    if pcd.has_colors():
        pcd_aligned.colors = pcd.colors
    
    pcd_aligned.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    
    return pcd_aligned


def color_faces_by_position(pcd):
    """
    Color each face of the cube based on position
    Top, Bottom, Front, Back, Left, Right
    """
    print(f"\n{'='*60}")
    print("COLORING FACES BY POSITION")
    print(f"{'='*60}")
    
    points = np.asarray(pcd.points)
    colors = np.zeros((len(points), 3))
    
    # Get bounding box to determine face thresholds
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    center = (min_bound + max_bound) / 2
    
    # Tolerance for face assignment (10% of extent)
    extent = max_bound - min_bound
    tolerance = 0.1 * np.min(extent)
    
    print(f"\nBounding box:")
    print(f"  Min: [{min_bound[0]*1000:.1f}, {min_bound[1]*1000:.1f}, {min_bound[2]*1000:.1f}] mm")
    print(f"  Max: [{max_bound[0]*1000:.1f}, {max_bound[1]*1000:.1f}, {max_bound[2]*1000:.1f}] mm")
    print(f"  Tolerance: {tolerance*1000:.2f} mm")
    
    # Assign colors based on position
    face_counts = {key: 0 for key in FACE_COLORS.keys()}
    
    for i, point in enumerate(points):
        assigned = False
        
        # Top face (max Z)
        if point[2] > max_bound[2] - tolerance:
            colors[i] = FACE_COLORS['top']
            face_counts['top'] += 1
            assigned = True
        
        # Bottom face (min Z)
        elif point[2] < min_bound[2] + tolerance:
            colors[i] = FACE_COLORS['bottom']
            face_counts['bottom'] += 1
            assigned = True
        
        # Front face (max Y)
        elif point[1] > max_bound[1] - tolerance:
            colors[i] = FACE_COLORS['front']
            face_counts['front'] += 1
            assigned = True
        
        # Back face (min Y)
        elif point[1] < min_bound[1] + tolerance:
            colors[i] = FACE_COLORS['back']
            face_counts['back'] += 1
            assigned = True
        
        # Right face (max X)
        elif point[0] > max_bound[0] - tolerance:
            colors[i] = FACE_COLORS['right']
            face_counts['right'] += 1
            assigned = True
        
        # Left face (min X)
        elif point[0] < min_bound[0] + tolerance:
            colors[i] = FACE_COLORS['left']
            face_counts['left'] += 1
            assigned = True
        
        # Interior/edge points - gray
        if not assigned:
            colors[i] = [0.5, 0.5, 0.5]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"\n--- Face Color Assignment ---")
    for face_name, count in face_counts.items():
        color = FACE_COLORS[face_name]
        print(f"{face_name:8s}: {count:5d} points | RGB={color}")
    
    return pcd


def create_dimension_lines(extent, offset=0.005):
    """
    Create dimension lines showing measurements
    Returns list of LineSet objects
    """
    lines = []
    
    x_len, y_len, z_len = extent
    half_x, half_y, half_z = x_len/2, y_len/2, z_len/2
    
    # X dimension (red)
    line_x = o3d.geometry.LineSet()
    points_x = [
        [-half_x, -half_y-offset, -half_z-offset],
        [half_x, -half_y-offset, -half_z-offset]
    ]
    line_x.points = o3d.utility.Vector3dVector(points_x)
    line_x.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_x.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    lines.append(line_x)
    
    # Y dimension (green)
    line_y = o3d.geometry.LineSet()
    points_y = [
        [-half_x-offset, -half_y, -half_z-offset],
        [-half_x-offset, half_y, -half_z-offset]
    ]
    line_y.points = o3d.utility.Vector3dVector(points_y)
    line_y.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_y.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
    lines.append(line_y)
    
    # Z dimension (blue)
    line_z = o3d.geometry.LineSet()
    points_z = [
        [-half_x-offset, -half_y-offset, -half_z],
        [-half_x-offset, -half_y-offset, half_z]
    ]
    line_z.points = o3d.utility.Vector3dVector(points_z)
    line_z.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_z.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
    lines.append(line_z)
    
    return lines


def verify_and_measure(pcd):
    """Verify alignment and return measurements"""
    print(f"\n{'='*60}")
    print("MEASUREMENTS")
    print(f"{'='*60}")
    
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    
    print(f"\nCube Dimensions:")
    print(f"  X: {extent[0]*1000:.2f} mm  (Red)")
    print(f"  Y: {extent[1]*1000:.2f} mm  (Green)")
    print(f"  Z: {extent[2]*1000:.2f} mm  (Blue)")
    
    mean_extent = np.mean(extent)
    deviations = extent - mean_extent
    
    print(f"\nMean edge length: {mean_extent*1000:.2f} mm")
    print(f"\nDeviations from mean:")
    print(f"  X: {deviations[0]*1000:+.2f} mm")
    print(f"  Y: {deviations[1]*1000:+.2f} mm")
    print(f"  Z: {deviations[2]*1000:+.2f} mm")
    
    volume = extent[0] * extent[1] * extent[2] * 1e9  # mm³
    print(f"\nVolume: {volume:.2f} mm³")
    
    return extent


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print(f"\n{'='*60}")
    print("MRAC CUBE DEMO - V3: COLORED FACES & MEASUREMENTS")
    print(f"{'='*60}\n")
    
    # 1. Load
    print(f"Loading: {INPUT_FILE}")
    pcd = o3d.io.read_point_cloud(INPUT_FILE)
    print(f"✓ Loaded {len(pcd.points)} points")
    
    # 2. Extract planes
    planes = extract_multiple_planes(pcd, max_planes=MAX_PLANES_TO_EXTRACT)
    
    # 3. Build basis
    rotation_matrix, z_axis = build_orthonormal_basis(planes)
    
    # 4. Align
    pcd_aligned = align_cube_to_axes(pcd, rotation_matrix, z_axis)
    
    # 5. Color faces
    pcd_colored = color_faces_by_position(pcd_aligned)
    
    # 6. Measure
    extent = verify_and_measure(pcd_colored)
    
    # 7. Save
    print(f"\n{'='*60}")
    print(f"Saving to: {OUTPUT_FILE}")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd_colored)
    print(f"✓ Saved!")
    print(f"{'='*60}\n")
    
    # 8. Visualize with measurements
    print("Opening visualization...")
    print("\nColor Legend:")
    print("  🔴 Red    = Top")
    print("  🟢 Green  = Bottom")
    print("  🔵 Blue   = Front")
    print("  🟡 Yellow = Back")
    print("  🟣 Magenta = Left")
    print("  🔷 Cyan   = Right")
    print("\nDimension Lines:")
    print(f"  Red line   = X: {extent[0]*1000:.2f} mm")
    print(f"  Green line = Y: {extent[1]*1000:.2f} mm")
    print(f"  Blue line  = Z: {extent[2]*1000:.2f} mm\n")
    
    # Create visualization elements
    aabb = pcd_colored.get_axis_aligned_bounding_box()
    aabb.color = (0.3, 0.3, 0.3)  # Gray box
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.03, origin=[0, 0, 0]
    )
    
    dimension_lines = create_dimension_lines(extent)
    
    # Combine all geometries
    geometries = [pcd_colored, aabb, coord_frame] + dimension_lines
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Cube Analysis - Colored Faces & Measurements",
        width=1920,
        height=1080
    )
    
    print(f"\n{'='*60}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()