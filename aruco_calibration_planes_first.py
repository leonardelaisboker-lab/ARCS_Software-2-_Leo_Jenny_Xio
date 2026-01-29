"""
ArUco-Based Calibration - PLANES FIRST Strategy
================================================

Correct workflow:
1. Load full resolution scan (8M+ points)
2. Detect ArUco markers in photos
3. Extract ALL planes using RANSAC (on full resolution!)
   → ~11 planes expected: 5 (cube A) + 5 (ArUco cube) + 1 (floor)
4. Identify ArUco cube from planes + markers
5. Calculate calibration (60mm reference)
6. Downsample ONLY for clustering (not for plane detection!)
7. Cluster objects using DBSCAN
8. Match clusters to planes
9. Align & scale everything

Requirements:
- pip install opencv-python opencv-contrib-python open3d scipy numpy

Author: MRAC Team
Date: 2026-01-27
"""

import cv2
import numpy as np
import open3d as o3d
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# ArUco settings
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 50  # Physical marker size
CUBE_SIZE_MM = 60    # Physical cube size

# Marker to face mapping
MARKER_TO_FACE = {
    0: 'BOTTOM',
    1: 'TOP',
    2: 'FRONT',
    3: 'BACK',
    4: 'LEFT',
    5: 'RIGHT'
}

# Plane extraction settings (CRITICAL - on full resolution!)
PLANE_DISTANCE_THRESHOLD = 0.003  # 3mm
PLANE_RANSAC_ITERATIONS = 1000
MIN_PLANE_POINTS = 1000
MAX_PLANES_TO_EXTRACT = 15  # Extract more than needed

# Clustering settings (on downsampled data)
DBSCAN_EPS = 0.01
DBSCAN_MIN_POINTS = 100


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_scan(scan_folder):
    """Load point cloud and images"""
    print(f"\n{'='*70}")
    print("STEP 1: LOADING SCAN")
    print(f"{'='*70}\n")
    
    scan_path = Path(scan_folder)
    
    # Find PLY
    ply_files = list(scan_path.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY file in {scan_folder}")
    
    ply_file = ply_files[0]
    print(f"Loading: {ply_file.name}")
    pcd = o3d.io.read_point_cloud(str(ply_file))
    print(f"✓ {len(pcd.points):,} points loaded")
    
    # Find images
    image_folders = ["images", "frames", "photos", "color"]
    image_paths = []
    
    for folder in image_folders:
        img_folder = scan_path / folder
        if img_folder.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                image_paths.extend(img_folder.glob(ext))
            if image_paths:
                print(f"✓ {len(image_paths)} images in {folder}/")
                break
    
    if not image_paths:
        print("⚠️  No images found - marker detection will fail")
    
    return pcd, sorted(image_paths)


# ============================================================================
# STEP 2: DETECT MARKERS
# ============================================================================

def detect_markers(image_paths):
    """Detect ArUco markers in photos"""
    print(f"\n{'='*70}")
    print("STEP 2: DETECTING ARUCO MARKERS")
    print(f"{'='*70}\n")
    
    if not image_paths:
        return {}, {}
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    marker_counts = {i: 0 for i in range(6)}
    
    print(f"Scanning {len(image_paths)} images...")
    
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None:
            for marker_id in ids.flatten():
                if marker_id in MARKER_TO_FACE:
                    marker_counts[marker_id] += 1
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)}...")
    
    print(f"\n✓ Marker detections:")
    for marker_id, count in marker_counts.items():
        if count > 0:
            face = MARKER_TO_FACE[marker_id]
            print(f"  Marker {marker_id} ({face:8s}): {count} detections")
    
    detected = sum(1 for c in marker_counts.values() if c > 0)
    print(f"\n✓ Found {detected} different markers")
    
    return marker_counts


# ============================================================================
# STEP 3: EXTRACT ALL PLANES (FULL RESOLUTION!)
# ============================================================================

def extract_all_planes(pcd):
    """
    Extract ALL major planes using RANSAC
    THIS IS THE KEY STEP - done on FULL RESOLUTION!
    """
    print(f"\n{'='*70}")
    print("STEP 3: EXTRACTING PLANES (FULL RESOLUTION)")
    print(f"{'='*70}\n")
    
    print(f"Extracting planes from {len(pcd.points):,} points...")
    print("This is done on FULL resolution for maximum accuracy!")
    print()
    
    planes = []
    remaining_pcd = pcd
    
    for i in range(MAX_PLANES_TO_EXTRACT):
        if len(remaining_pcd.points) < MIN_PLANE_POINTS:
            break
        
        # RANSAC plane fitting
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
        
        # Get plane points and center
        plane_pcd = remaining_pcd.select_by_index(inliers)
        plane_points = np.asarray(plane_pcd.points)
        center = np.mean(plane_points, axis=0)
        
        # Calculate plane size (for debugging)
        extents = np.ptp(plane_points, axis=0)
        area_estimate = np.prod(np.sort(extents)[-2:])  # Two largest dimensions
        
        planes.append({
            'id': i,
            'normal': normal,
            'center': center,
            'distance': d,
            'num_points': len(inliers),
            'points': plane_points,
            'pcd': plane_pcd,
            'extents': extents,
            'area': area_estimate
        })
        
        print(f"  Plane {i}: {len(inliers):7,} points, "
              f"normal=[{normal[0]:+.2f}, {normal[1]:+.2f}, {normal[2]:+.2f}], "
              f"area≈{area_estimate:.4f}m²")
        
        # Remove this plane
        outliers = list(set(range(len(remaining_pcd.points))) - set(inliers))
        remaining_pcd = remaining_pcd.select_by_index(outliers)
    
    print(f"\n✓ Extracted {len(planes)} planes")
    
    return planes


# ============================================================================
# STEP 4: IDENTIFY ARUCO CUBE & FLOOR
# ============================================================================

def identify_aruco_cube_and_floor(planes, marker_counts):
    """
    Identify which planes belong to:
    - ArUco calibration cube
    - Floor
    - Other objects
    """
    print(f"\n{'='*70}")
    print("STEP 4: IDENTIFYING ARUCO CUBE & FLOOR")
    print(f"{'='*70}\n")
    
    # Strategy:
    # 1. Floor = largest horizontal plane
    # 2. ArUco cube = cluster of ~6 planes near each other
    # 3. Other cube = remaining planes
    
    # Sort by size
    planes_by_size = sorted(planes, key=lambda p: p['num_points'], reverse=True)
    
    # Find floor (largest + horizontal)
    floor_plane = None
    for plane in planes_by_size:
        # Check if horizontal (normal points up/down)
        if abs(plane['normal'][2]) > 0.8:  # Within ~36° of vertical
            floor_plane = plane
            print(f"✓ Floor: Plane {plane['id']} ({plane['num_points']:,} points)")
            break
    
    if not floor_plane:
        print("⚠️  Warning: No clear floor plane found")
        floor_plane = planes_by_size[0]  # Fallback: largest plane
    
    # Remove floor from candidates
    non_floor_planes = [p for p in planes if p['id'] != floor_plane['id']]
    
    # Find ArUco cube: cluster of planes with similar Z-height
    # (both cubes are on the floor, so their centers should be at similar heights)
    
    if len(non_floor_planes) < 5:
        print(f"⚠️  Warning: Only {len(non_floor_planes)} non-floor planes found")
        print("    Expected at least 10 (5 per cube)")
    
    # For now, just separate by X-position (left vs right cube)
    # or by proximity clustering
    
    # Simple heuristic: group by distance
    from scipy.spatial.distance import cdist
    
    centers = np.array([p['center'] for p in non_floor_planes])
    
    if len(centers) >= 6:
        # Find two main clusters
        # This is a simplified approach - proper clustering would be better
        
        # For now: identify ArUco cube by marker detections
        # Planes near markers = ArUco cube
        
        # Simplified: assume first 5-6 planes = one cube, next 5-6 = other cube
        print(f"\n✓ Found {len(non_floor_planes)} object planes")
        print("  (ArUco cube identification needs marker-plane matching)")
    
    # Return simplified structure for now
    return {
        'floor': floor_plane,
        'aruco_cube_planes': non_floor_planes[:6] if len(non_floor_planes) >= 6 else non_floor_planes,
        'other_planes': non_floor_planes[6:] if len(non_floor_planes) > 6 else [],
        'all_object_planes': non_floor_planes
    }


# ============================================================================
# STEP 5: CALCULATE CALIBRATION
# ============================================================================

def calculate_calibration(aruco_planes):
    """Calculate coordinate system and scale from ArUco cube planes"""
    print(f"\n{'='*70}")
    print("STEP 5: CALCULATING CALIBRATION")
    print(f"{'='*70}\n")
    
    if len(aruco_planes) < 3:
        print("⚠️  Not enough planes for calibration")
        return None, None, None
    
    # Find horizontal and vertical planes
    horizontal_planes = []
    vertical_planes = []
    
    for plane in aruco_planes:
        if abs(plane['normal'][2]) > 0.7:  # Horizontal
            horizontal_planes.append(plane)
        else:  # Vertical
            vertical_planes.append(plane)
    
    # Build coordinate system
    if horizontal_planes:
        z_axis = horizontal_planes[0]['normal'].copy()
        if z_axis[2] < 0:
            z_axis = -z_axis
        print(f"✓ Z-axis from horizontal plane")
    else:
        z_axis = np.array([0, 0, 1])
        print("⚠️  Using default Z-axis")
    
    if len(vertical_planes) >= 2:
        # X from first vertical plane
        x_axis_raw = vertical_planes[0]['normal']
        x_axis = x_axis_raw - np.dot(x_axis_raw, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y from cross product
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        print(f"✓ X/Y axes from vertical planes")
    else:
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        print("⚠️  Using default X/Y axes")
    
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # Calculate center
    all_points = np.vstack([p['points'] for p in aruco_planes])
    center = np.mean(all_points, axis=0)
    
    # Calculate scale
    # Transform to aligned frame and measure
    points_centered = all_points - center
    points_aligned = points_centered @ rotation_matrix.T
    extents = np.ptp(points_aligned, axis=0)
    mean_extent = np.mean(extents)
    
    scale_factor = mean_extent / (CUBE_SIZE_MM * 0.001)
    
    print(f"\n✓ Cube center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"✓ Measured extents: X={extents[0]:.4f}, Y={extents[1]:.4f}, Z={extents[2]:.4f}m")
    print(f"✓ Expected: {CUBE_SIZE_MM * 0.001:.4f}m ({CUBE_SIZE_MM}mm)")
    print(f"✓ Scale factor: {scale_factor:.6f}")
    
    return rotation_matrix, center, scale_factor


# ============================================================================
# STEP 6: DOWNSAMPLE FOR CLUSTERING
# ============================================================================

def downsample_for_clustering(pcd, planes_info):
    """
    NOW we downsample - but only for clustering, not for plane geometry!
    """
    print(f"\n{'='*70}")
    print("STEP 6: DOWNSAMPLING FOR CLUSTERING")
    print(f"{'='*70}\n")
    
    # Remove floor points first
    floor_plane = planes_info['floor']
    all_points = np.asarray(pcd.points)
    
    # Calculate distance to floor plane
    a, b, c = floor_plane['normal']
    d = floor_plane['distance']
    distances = np.abs(a * all_points[:, 0] + 
                      b * all_points[:, 1] + 
                      c * all_points[:, 2] + d)
    
    # Keep points NOT on floor (>5mm away)
    object_mask = distances > 0.005
    object_indices = np.where(object_mask)[0]
    
    objects_pcd = pcd.select_by_index(object_indices)
    
    print(f"After floor removal: {len(objects_pcd.points):,} points")
    
    # NOW downsample for clustering
    TARGET_POINTS = 1500000
    
    if len(objects_pcd.points) > TARGET_POINTS:
        print(f"⚡ Downsampling to ~{TARGET_POINTS/1000000:.1f}M for clustering...")
        
        ratio = len(objects_pcd.points) / TARGET_POINTS
        voxel_size = 0.001 * (ratio ** 0.4)
        
        objects_downsampled = objects_pcd.voxel_down_sample(voxel_size=voxel_size)
        
        print(f"✓ Reduced to {len(objects_downsampled.points):,} points")
        print(f"  Voxel: {voxel_size*1000:.2f}mm, "
              f"Reduction: {100*(1-len(objects_downsampled.points)/len(objects_pcd.points)):.1f}%")
    else:
        objects_downsampled = objects_pcd
        print("✓ No downsampling needed")
    
    return objects_downsampled


# ============================================================================
# STEP 7: CLUSTER OBJECTS
# ============================================================================

def cluster_objects(objects_pcd):
    """Cluster objects using DBSCAN"""
    print(f"\n{'='*70}")
    print("STEP 7: CLUSTERING OBJECTS (DBSCAN)")
    print(f"{'='*70}\n")
    
    print(f"Clustering {len(objects_pcd.points):,} points...")
    print(f"eps={DBSCAN_EPS*1000}mm, min_points={DBSCAN_MIN_POINTS}")
    
    labels = np.array(objects_pcd.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=True
    ))
    
    n_clusters = labels.max() + 1
    print(f"\n✓ Found {n_clusters} clusters")
    
    clusters = []
    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        cluster_pcd = objects_pcd.select_by_index(indices)
        
        clusters.append({
            'id': cluster_id,
            'pcd': cluster_pcd,
            'num_points': len(indices),
            'center': np.mean(np.asarray(cluster_pcd.points), axis=0)
        })
        
        print(f"  Cluster {cluster_id}: {len(indices):,} points")
    
    return clusters


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print("ARUCO CALIBRATION - PLANES FIRST STRATEGY")
    print(f"{'='*70}\n")
    
    scan_folder = input("Scan folder: ").strip().strip('"')
    
    if not os.path.exists(scan_folder):
        print(f"❌ Not found: {scan_folder}")
        return
    
    try:
        # Step 1: Load
        pcd, image_paths = load_scan(scan_folder)
        
        # Step 2: Detect markers
        marker_counts = detect_markers(image_paths)
        
        # Step 3: Extract planes (FULL RESOLUTION!)
        planes = extract_all_planes(pcd)
        
        # Step 4: Identify cubes
        planes_info = identify_aruco_cube_and_floor(planes, marker_counts)
        
        # Step 5: Calculate calibration
        aruco_planes = planes_info['aruco_cube_planes']
        rotation, center, scale = calculate_calibration(aruco_planes)
        
        if rotation is None:
            print("\n❌ Calibration failed")
            return
        
        # Step 6: Downsample for clustering
        objects_downsampled = downsample_for_clustering(pcd, planes_info)
        
        # Step 7: Cluster
        clusters = cluster_objects(objects_downsampled)
        
        print(f"\n{'='*70}")
        print("✅ DONE!")
        print(f"{'='*70}\n")
        print(f"Planes extracted: {len(planes)}")
        print(f"Clusters found: {len(clusters)}")
        print(f"Scale factor: {scale:.6f}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
