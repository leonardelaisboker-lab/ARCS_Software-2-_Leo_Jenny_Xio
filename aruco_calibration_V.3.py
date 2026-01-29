"""
ArUco-Based Multi-Object Calibration for Polycam Scans
=======================================================

This script processes a Polycam scan containing:
- Calibration cube (40×40×40mm with ArUco markers)
- Other objects to be measured
- Floor/table surface

Process:
1. Load Polycam scan (PLY + photos)
2. Detect ArUco markers in photos
3. Find calibration cube in point cloud (via markers)
4. Calculate coordinate system and scale from calibration cube
5. Detect floor (below calibration cube)
6. Find other objects (DBSCAN clustering)
7. Align and scale everything
8. Save results

Requirements:
- pip install opencv-python opencv-contrib-python open3d scipy numpy

Author: MRAC Team
Date: 2026-01-26
"""

import cv2
import numpy as np
import open3d as o3d
import os
import json
from pathlib import Path
from scipy.spatial import ConvexHull

# ============================================================================
# CONFIGURATION
# ============================================================================

# ArUco settings (must match marker generation!)
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 50  # Physical marker size (50×50mm)
CUBE_SIZE_MM = 60    # Physical cube size (60×60×60mm with 5mm border)

# Marker to face mapping
MARKER_TO_FACE = {
    0: 'BOTTOM',
    1: 'TOP',
    2: 'FRONT',
    3: 'BACK',
    4: 'LEFT',
    5: 'RIGHT'
}

# Detection settings
MIN_MARKERS_REQUIRED = 2  # Need at least 2 markers
FLOOR_OFFSET_MM = 2       # How far below BOTTOM marker is floor
DBSCAN_EPS = 0.01         # 10mm clustering for object separation
DBSCAN_MIN_POINTS = 100   # Minimum points per object

# Plane extraction
PLANE_DISTANCE_THRESHOLD = 0.003  # 3mm
PLANE_RANSAC_ITERATIONS = 1000
MIN_PLANE_POINTS = 500


# ============================================================================
# STEP 1: LOAD POLYCAM DATA
# ============================================================================

def load_polycam_scan(scan_folder):
    """Load Polycam export data"""
    print(f"\n{'='*70}")
    print("STEP 1: LOADING POLYCAM SCAN")
    print(f"{'='*70}\n")
    
    scan_path = Path(scan_folder)
    
    # Find PLY file
    ply_files = list(scan_path.glob("*.ply"))
    if len(ply_files) == 0:
        raise FileNotFoundError(f"❌ No PLY file found in {scan_folder}")
    
    ply_file = ply_files[0]
    print(f"Loading point cloud: {ply_file.name}")
    pcd = o3d.io.read_point_cloud(str(ply_file))
    print(f"✓ Loaded {len(pcd.points)} points")
    
    # Find images
    image_folders = ["images", "frames", "photos", "color"]
    image_paths = []
    
    for folder_name in image_folders:
        img_folder = scan_path / folder_name
        if img_folder.exists():
            extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
            for ext in extensions:
                image_paths.extend(img_folder.glob(ext))
            if image_paths:
                print(f"✓ Found {len(image_paths)} images in {folder_name}/")
                break
    
    if len(image_paths) == 0:
        print("⚠️  WARNING: No images found!")
        print("   Marker detection requires photos from Polycam export.")
        return pcd, []
    
    return pcd, sorted(image_paths)


# ============================================================================
# STEP 2: DETECT ARUCO MARKERS
# ============================================================================

def detect_aruco_markers(image_paths):
    """Detect ArUco markers in all images"""
    print(f"\n{'='*70}")
    print("STEP 2: DETECTING ARUCO MARKERS IN PHOTOS")
    print(f"{'='*70}\n")
    
    if not image_paths:
        print("❌ No images provided, skipping marker detection")
        return {}, {}
    
    # Setup detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    all_detections = {}
    marker_counts = {i: 0 for i in range(6)}
    
    print(f"Scanning {len(image_paths)} images for ArUco markers...")
    
    for img_idx, img_path in enumerate(image_paths):
        # Load and convert to grayscale
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            detections = []
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in MARKER_TO_FACE:
                    detections.append({
                        'id': int(marker_id),
                        'face': MARKER_TO_FACE[marker_id],
                        'corners': corners[i][0].tolist(),
                        'center': np.mean(corners[i][0], axis=0).tolist()
                    })
                    marker_counts[marker_id] += 1
            
            if detections:
                all_detections[str(img_path)] = {
                    'image_path': str(img_path),
                    'image_size': list(img.shape[:2]),
                    'markers': detections
                }
        
        # Progress
        if (img_idx + 1) % 10 == 0:
            print(f"  Processed {img_idx + 1}/{len(image_paths)} images...")
    
    print(f"\n✓ Detection complete!")
    print(f"\nMarker detections:")
    for marker_id, count in marker_counts.items():
        if count > 0:
            face = MARKER_TO_FACE[marker_id]
            print(f"  Marker {marker_id} ({face:8s}): {count} detections")
    
    total_detections = sum(marker_counts.values())
    detected_markers = sum(1 for c in marker_counts.values() if c > 0)
    
    print(f"\n✓ Found {detected_markers} different markers")
    print(f"✓ Total detections: {total_detections}")
    
    if total_detections == 0:
        print("\n⚠️  WARNING: No ArUco markers detected!")
        print("   Possible reasons:")
        print("   - Markers not visible in photos")
        print("   - Wrong dictionary (should be DICT_4X4_50)")
        print("   - Photos too blurry")
    
    return all_detections, marker_counts


# ============================================================================
# STEP 3: FIND CALIBRATION CUBE IN POINT CLOUD
# ============================================================================

def extract_planes_from_cloud(pcd, max_planes=10):
    """Extract major planes using RANSAC"""
    planes = []
    remaining_pcd = pcd
    
    for i in range(max_planes):
        if len(remaining_pcd.points) < MIN_PLANE_POINTS:
            break
        
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=PLANE_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=PLANE_RANSAC_ITERATIONS
        )
        
        if len(inliers) < MIN_PLANE_POINTS:
            break
        
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        # Get plane center
        plane_points = np.asarray(remaining_pcd.select_by_index(inliers).points)
        center = np.mean(plane_points, axis=0)
        
        planes.append({
            'id': i,
            'normal': normal,
            'center': center,
            'distance': d,
            'num_points': len(inliers),
            'points': plane_points
        })
        
        # Remove plane
        outliers = list(set(range(len(remaining_pcd.points))) - set(inliers))
        remaining_pcd = remaining_pcd.select_by_index(outliers)
    
    return planes


def find_calibration_cube(pcd, marker_counts):
    """
    Find calibration cube in point cloud
    Strategy: Extract planes, identify likely cube faces
    """
    print(f"\n{'='*70}")
    print("STEP 3: LOCATING CALIBRATION CUBE")
    print(f"{'='*70}\n")
    
    if sum(marker_counts.values()) == 0:
        print("⚠️  No markers detected, using heuristic to find cube")
    
    # Extract planes
    print("Extracting planes from point cloud...")
    planes = extract_planes_from_cloud(pcd, max_planes=15)
    print(f"✓ Found {len(planes)} planes")
    
    # Identify cube planes (heuristic: similar size, orthogonal normals)
    # For now, take the 6 largest planes that are roughly cube-sized
    
    # Sort by size
    planes_sorted = sorted(planes, key=lambda p: p['num_points'], reverse=True)
    
    # Find cube: look for cluster of planes with similar centers
    cube_planes = []
    
    for plane in planes_sorted[:10]:  # Check top 10 planes
        # Estimate if this could be part of a cube
        # Heuristic: plane size should be reasonable for 40mm cube
        extent = np.ptp(plane['points'], axis=0)  # Range in x, y, z
        
        # Very rough check (scan units vary)
        if np.all(extent > 0.01) and np.all(extent < 0.15):  # Between 10-150mm ish
            cube_planes.append(plane)
        
        if len(cube_planes) >= 6:
            break
    
    if len(cube_planes) < 3:
        print("⚠️  Warning: Could not confidently identify cube")
        print(f"   Found only {len(cube_planes)} candidate planes")
        # Use all planes as fallback
        cube_planes = planes_sorted[:6]
    
    print(f"\n✓ Identified {len(cube_planes)} cube planes")
    
    # Calculate cube center (average of plane centers)
    cube_center = np.mean([p['center'] for p in cube_planes], axis=0)
    print(f"✓ Cube center: [{cube_center[0]:.4f}, {cube_center[1]:.4f}, {cube_center[2]:.4f}]")
    
    # Extract cube points (all points near these planes)
    cube_point_cloud = extract_cube_points(pcd, cube_planes, cube_center)
    
    return cube_point_cloud, cube_planes, cube_center


def extract_cube_points(pcd, cube_planes, cube_center, radius=0.08):
    """Extract points belonging to cube"""
    points = np.asarray(pcd.points)
    
    # Method 1: Distance to cube center
    distances = np.linalg.norm(points - cube_center, axis=1)
    near_center = distances < radius
    
    # Method 2: Close to any cube plane
    near_plane = np.zeros(len(points), dtype=bool)
    for plane in cube_planes:
        a, b, c = plane['normal']
        d = plane['distance']
        dist_to_plane = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        near_plane |= (dist_to_plane < PLANE_DISTANCE_THRESHOLD * 3)
    
    # Combine
    cube_mask = near_center & near_plane
    cube_indices = np.where(cube_mask)[0]
    
    cube_pcd = pcd.select_by_index(cube_indices)
    
    print(f"✓ Extracted {len(cube_indices)} points for calibration cube")
    
    return cube_pcd


# ============================================================================
# STEP 4: CALCULATE COORDINATE SYSTEM & SCALE
# ============================================================================

def calculate_calibration(cube_pcd, cube_planes, marker_counts):
    """Calculate coordinate system and scale from calibration cube"""
    print(f"\n{'='*70}")
    print("STEP 4: CALCULATING COORDINATE SYSTEM & SCALE")
    print(f"{'='*70}\n")
    
    # Build coordinate system from cube planes
    # Find horizontal plane (Z-axis)
    z_axis = None
    horizontal_plane = None
    
    for plane in cube_planes:
        # Check if roughly horizontal (normal points up/down)
        if abs(plane['normal'][2]) > 0.8:  # Mostly vertical normal = horizontal plane
            z_axis = plane['normal'].copy()
            if z_axis[2] < 0:
                z_axis = -z_axis  # Point up
            horizontal_plane = plane
            break
    
    if z_axis is None:
        print("⚠️  No clear horizontal plane, using Z=[0,0,1]")
        z_axis = np.array([0, 0, 1])
    else:
        print(f"✓ Z-axis from horizontal plane: {z_axis}")
    
    # Find X and Y axes from vertical planes
    vertical_planes = [p for p in cube_planes 
                      if abs(p['normal'][2]) < 0.3]  # Mostly horizontal normal = vertical plane
    
    if len(vertical_planes) >= 2:
        # Sort by X-alignment
        vertical_planes.sort(key=lambda p: abs(p['normal'][0]), reverse=True)
        
        # X from most X-aligned plane
        x_axis_raw = vertical_planes[0]['normal']
        x_axis = x_axis_raw - np.dot(x_axis_raw, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y from cross product
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        print(f"✓ X-axis: {x_axis}")
        print(f"✓ Y-axis: {y_axis}")
    else:
        print("⚠️  Not enough vertical planes, using default axes")
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
    
    # Build rotation matrix
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    det = np.linalg.det(rotation_matrix)
    print(f"\n✓ Rotation matrix determinant: {det:.4f}")
    
    # Calculate scale
    # Measure cube dimensions
    cube_points = np.asarray(cube_pcd.points)
    cube_center = np.mean(cube_points, axis=0)
    
    # Transform to aligned coordinates
    points_centered = cube_points - cube_center
    points_aligned = points_centered @ rotation_matrix.T
    
    # Measure extents
    extents = np.ptp(points_aligned, axis=0)  # Range in each dimension
    mean_extent = np.mean(extents)
    
    # Scale factor: measured extent / expected extent
    scale_factor = mean_extent / (CUBE_SIZE_MM * 0.001)  # Convert mm to m
    
    print(f"\n✓ Cube measurements (scan units):")
    print(f"   X: {extents[0]:.4f}")
    print(f"   Y: {extents[1]:.4f}")
    print(f"   Z: {extents[2]:.4f}")
    print(f"   Mean: {mean_extent:.4f}")
    print(f"\n✓ Expected: {CUBE_SIZE_MM * 0.001:.4f} m ({CUBE_SIZE_MM}mm)")
    print(f"✓ Scale factor: {scale_factor:.6f}")
    
    return rotation_matrix, cube_center, scale_factor, horizontal_plane


# ============================================================================
# STEP 5: DETECT FLOOR
# ============================================================================

def detect_floor(pcd, cube_center, horizontal_plane):
    """Detect floor below calibration cube"""
    print(f"\n{'='*70}")
    print("STEP 5: DETECTING FLOOR")
    print(f"{'='*70}\n")
    
    points = np.asarray(pcd.points)
    
    # Floor is below the cube
    # Estimate floor Z based on cube bottom
    if horizontal_plane:
        # Use horizontal plane as reference
        cube_bottom_z = cube_center[2] - (CUBE_SIZE_MM * 0.001) / 2
    else:
        # Fallback: use cube center minus half height
        cube_bottom_z = cube_center[2] - 0.025  # Rough estimate
    
    floor_threshold = cube_bottom_z - (FLOOR_OFFSET_MM * 0.001)
    
    print(f"Cube center Z: {cube_center[2]:.4f}")
    print(f"Estimated cube bottom Z: {cube_bottom_z:.4f}")
    print(f"Floor threshold: Z < {floor_threshold:.4f}")
    
    # Separate floor and objects
    floor_mask = points[:, 2] < floor_threshold
    floor_indices = np.where(floor_mask)[0]
    object_indices = np.where(~floor_mask)[0]
    
    floor_pcd = pcd.select_by_index(floor_indices)
    objects_pcd = pcd.select_by_index(object_indices)
    
    print(f"\n✓ Floor: {len(floor_indices)} points")
    print(f"✓ Objects: {len(object_indices)} points")
    
    return objects_pcd, floor_pcd


# ============================================================================
# STEP 6: FIND OTHER OBJECTS
# ============================================================================

def find_objects(objects_pcd, cube_pcd):
    """Find all objects using DBSCAN clustering"""
    print(f"\n{'='*70}")
    print("STEP 6: FINDING OBJECTS (DBSCAN)")
    print(f"{'='*70}\n")
    
    print(f"Input: {len(objects_pcd.points)} points")
    
    # Downsample for faster clustering if too many points
    if len(objects_pcd.points) > 1000000:  # 1 million threshold
        print("⚡ Downsampling for faster clustering...")
        voxel_size = 0.005  # 5mm voxels
        objects_pcd_downsampled = objects_pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"✓ Reduced to {len(objects_pcd_downsampled.points)} points (voxel_size={voxel_size*1000}mm)")
    else:
        objects_pcd_downsampled = objects_pcd
    
    print(f"\nClustering {len(objects_pcd_downsampled.points)} points...")
    
    # DBSCAN clustering with larger epsilon for downsampled data
    clustering_eps = DBSCAN_EPS if len(objects_pcd_downsampled.points) < 1000000 else 0.02
    print(f"Using eps={clustering_eps*1000}mm, min_points={DBSCAN_MIN_POINTS}")
    
    labels = np.array(objects_pcd_downsampled.cluster_dbscan(
        eps=clustering_eps,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=True
    ))
    
    max_label = labels.max()
    n_clusters = max_label + 1
    
    print(f"✓ Found {n_clusters} clusters in downsampled data")
    
    # If we downsampled, we need to work with the downsampled clusters
    working_pcd = objects_pcd_downsampled
    
    # Extract each cluster
    objects = []
    cube_points = np.asarray(cube_pcd.points)
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_pcd = working_pcd.select_by_index(cluster_indices)
        cluster_points = np.asarray(cluster_pcd.points)
        
        # Check if this is the calibration cube
        # (by comparing centers)
        cluster_center = np.mean(cluster_points, axis=0)
        cube_center = np.mean(cube_points, axis=0)
        distance_to_cube = np.linalg.norm(cluster_center - cube_center)
        
        is_calibration_cube = distance_to_cube < 0.01  # Within 10mm
        
        object_type = "Calibration Cube" if is_calibration_cube else f"Object {len(objects) + 1}"
        
        objects.append({
            'id': cluster_id,
            'type': object_type,
            'is_calibration': is_calibration_cube,
            'pcd': cluster_pcd,
            'num_points': len(cluster_indices),
            'center': cluster_center
        })
        
        print(f"  Cluster {cluster_id}: {len(cluster_indices)} points - {object_type}")
    
    print(f"\n✓ Identified {len(objects)} objects")
    
    return objects


# ============================================================================
# STEP 7: ALIGN AND SCALE EVERYTHING
# ============================================================================

def align_and_scale(pcd, rotation_matrix, center, scale_factor):
    """Apply alignment transformation"""
    points = np.asarray(pcd.points)
    
    # Center
    points_centered = points - center
    
    # Rotate
    R_inv = rotation_matrix.T
    points_aligned = points_centered @ R_inv
    
    # Create aligned point cloud
    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(points_aligned)
    
    if pcd.has_colors():
        pcd_aligned.colors = pcd.colors
    
    pcd_aligned.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    
    return pcd_aligned


# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

def save_results(objects, floor_pcd, rotation_matrix, center, scale_factor, 
                marker_detections, output_dir):
    """Save all results"""
    print(f"\n{'='*70}")
    print("STEP 8: SAVING RESULTS")
    print(f"{'='*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each object
    for obj in objects:
        obj_name = obj['type'].lower().replace(' ', '_')
        filename = f"{output_dir}/{obj_name}.ply"
        o3d.io.write_point_cloud(filename, obj['pcd'])
        print(f"✓ Saved: {filename}")
    
    # Save floor
    if len(floor_pcd.points) > 0:
        floor_file = f"{output_dir}/floor.ply"
        o3d.io.write_point_cloud(floor_file, floor_pcd)
        print(f"✓ Saved: {floor_file}")
    
    # Save full scene (all objects + floor)
    full_pcd = o3d.geometry.PointCloud()
    for obj in objects:
        full_pcd += obj['pcd']
    full_pcd += floor_pcd
    
    full_file = f"{output_dir}/full_scene_aligned.ply"
    o3d.io.write_point_cloud(full_file, full_pcd)
    print(f"✓ Saved: {full_file}")
    
    # Save calibration data
    calib_file = f"{output_dir}/calibration_data.txt"
    with open(calib_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CALIBRATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("ROTATION MATRIX:\n")
        f.write(str(rotation_matrix) + "\n\n")
        
        f.write(f"CENTER: {center}\n\n")
        
        f.write(f"SCALE FACTOR: {scale_factor:.6f} (scan_units / m)\n")
        f.write(f"CUBE SIZE (reference): {CUBE_SIZE_MM}mm\n")
        f.write(f"MARKER SIZE: {MARKER_SIZE_MM}mm\n\n")
        
        f.write("DETECTED OBJECTS:\n")
        for obj in objects:
            f.write(f"  {obj['type']}: {obj['num_points']} points\n")
        
        f.write(f"\nMARKER DETECTIONS:\n")
        if marker_detections:
            for marker_id, count in marker_detections.items():
                if count > 0:
                    face = MARKER_TO_FACE.get(marker_id, 'UNKNOWN')
                    f.write(f"  Marker {marker_id} ({face}): {count} detections\n")
        else:
            f.write("  No markers detected\n")
    
    print(f"✓ Saved: {calib_file}\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main calibration pipeline"""
    
    print(f"\n{'='*70}")
    print("ARUCO-BASED MULTI-OBJECT CALIBRATION")
    print("From Polycam Scan to Aligned Objects")
    print(f"{'='*70}\n")
    
    # Get input
    scan_folder = input("Enter path to Polycam scan folder: ").strip()
    scan_folder = scan_folder.strip('"')  # Remove quotes if present
    
    if not os.path.exists(scan_folder):
        print(f"❌ ERROR: Folder not found: {scan_folder}")
        return
    
    output_folder = "calibration_results"
    
    try:
        # Step 1: Load data
        pcd, image_paths = load_polycam_scan(scan_folder)
        
        # Step 2: Detect markers
        marker_detections, marker_counts = detect_aruco_markers(image_paths)
        
        # Step 3: Find calibration cube
        cube_pcd, cube_planes, cube_center = find_calibration_cube(pcd, marker_counts)
        
        # Step 4: Calculate calibration
        rotation_matrix, center, scale_factor, horizontal_plane = calculate_calibration(
            cube_pcd, cube_planes, marker_counts
        )
        
        # Step 5: Detect floor
        objects_pcd, floor_pcd = detect_floor(pcd, center, horizontal_plane)
        
        # Step 6: Find other objects
        objects = find_objects(objects_pcd, cube_pcd)
        
        # Step 7: Align everything
        print(f"\n{'='*70}")
        print("STEP 7: ALIGNING ALL OBJECTS")
        print(f"{'='*70}\n")
        
        for obj in objects:
            obj['pcd'] = align_and_scale(obj['pcd'], rotation_matrix, center, scale_factor)
            print(f"✓ Aligned: {obj['type']}")
        
        floor_pcd = align_and_scale(floor_pcd, rotation_matrix, center, scale_factor)
        print("✓ Aligned: Floor")
        
        # Step 8: Save results
        save_results(objects, floor_pcd, rotation_matrix, center, scale_factor,
                    marker_counts, output_folder)
        
        # Summary
        print(f"\n{'='*70}")
        print("✅ CALIBRATION COMPLETE!")
        print(f"{'='*70}\n")
        
        print(f"📁 Output folder: {output_folder}/")
        print(f"\n📊 Summary:")
        print(f"   - Found {len(objects)} objects")
        print(f"   - Detected {sum(1 for c in marker_counts.values() if c > 0)} different markers")
        print(f"   - Scale factor: {scale_factor:.6f}")
        
        # Visualize
        print("\n🎨 Opening visualization...")
        
        vis_objects = [obj['pcd'] for obj in objects]
        vis_objects.append(floor_pcd)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0]
        )
        vis_objects.append(coord_frame)
        
        # Add bounding boxes
        for obj in objects:
            if not obj['is_calibration']:
                aabb = obj['pcd'].get_axis_aligned_bounding_box()
                aabb.color = (1, 0, 0)
                vis_objects.append(aabb)
        
        o3d.visualization.draw_geometries(
            vis_objects,
            window_name="Calibrated Scene - All Objects Aligned",
            width=1920,
            height=1080
        )
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
