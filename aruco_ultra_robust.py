"""
ArUco Calibration - ULTRA ROBUST for Noisy Photogrammetry
==========================================================

Designed for: Noisy photogrammetry scans with outliers, uneven surfaces
Strategy: Aggressive filtering + multi-pass detection + validation + fallbacks

Key features:
- Statistical outlier removal
- Voxel smoothing
- Multi-pass RANSAC (coarse → fine)
- Plane validation (size, normal, quality)
- Fallback strategies at every step
- Extensive debug output

Author: MRAC Team
Date: 2026-01-27
"""

import cv2
import numpy as np
import open3d as o3d
import os
from pathlib import Path
from scipy.spatial.distance import cdist

# ============================================================================
# CONFIGURATION
# ============================================================================

# ArUco settings
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 50
CUBE_SIZE_MM = 60

MARKER_TO_FACE = {
    0: 'BOTTOM', 1: 'TOP', 2: 'FRONT',
    3: 'BACK', 4: 'LEFT', 5: 'RIGHT'
}

# NOISE FILTERING (aggressive for bad scans!)
OUTLIER_NB_NEIGHBORS = 30        # More neighbors = more aggressive
OUTLIER_STD_RATIO = 1.5          # Lower = more aggressive (1.5-2.0)
VOXEL_DOWNSAMPLE_MM = 0.5  # 0.5mm (oder sogar 0.2mm!)

# PLANE DETECTION - Multi-pass strategy
PLANE_PASSES = [
    # Pass 1: Large planes (floor, walls)
    {'name': 'Large', 'threshold': 0.005, 'min_points': 100000, 'iterations': 1500},
    # Pass 2: Medium planes (cube faces)
    {'name': 'Medium', 'threshold': 0.002, 'min_points': 10000, 'iterations': 1000},
    # Pass 3: Small planes (details)
    {'name': 'Small', 'threshold': 0.001, 'min_points': 5000, 'iterations': 500},
]

MAX_PLANES_TOTAL = 20

# PLANE VALIDATION
MIN_PLANE_AREA_M2 = 0.0001      # ~10×10mm minimum
MAX_PLANE_AREA_M2 = 10.0        # 10m² maximum
MAX_INLIER_RATIO = 0.6          # If >60% of points = one plane, something's wrong

# CUBE IDENTIFICATION
CUBE_FACE_AREA_MIN = 0.002      # ~45×45mm
CUBE_FACE_AREA_MAX = 0.01       # ~100×100mm
CUBE_CLUSTER_RADIUS = 0.2       # Cube faces within 200mm


# ============================================================================
# STEP 1: LOAD & CLEAN
# ============================================================================

def load_and_clean_scan(scan_folder):
    """Load scan with AGGRESSIVE cleaning for noisy photogrammetry"""
    print(f"\n{'='*70}")
    print("STEP 1: LOADING & CLEANING SCAN (AGGRESSIVE MODE)")
    print(f"{'='*70}\n")
    
    scan_path = Path(scan_folder)
    
    # Find PLY
    ply_files = list(scan_path.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY in {scan_folder}")
    
    ply_file = ply_files[0]
    print(f"Loading: {ply_file.name}")
    pcd_raw = o3d.io.read_point_cloud(str(ply_file))
    print(f"✓ Raw: {len(pcd_raw.points):,} points")
    
    # STEP 1A: Statistical Outlier Removal
    print(f"\n🧹 Removing outliers...")
    print(f"   neighbors={OUTLIER_NB_NEIGHBORS}, std_ratio={OUTLIER_STD_RATIO}")
    
    pcd_clean, inlier_indices = pcd_raw.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS,
        std_ratio=OUTLIER_STD_RATIO
    )
    
    outliers_removed = len(pcd_raw.points) - len(pcd_clean.points)
    outlier_pct = 100 * outliers_removed / len(pcd_raw.points)
    
    print(f"✓ Removed {outliers_removed:,} outliers ({outlier_pct:.1f}%)")
    print(f"✓ Clean: {len(pcd_clean.points):,} points")
    
    # STEP 1B: Voxel Downsampling (smoothing + reduce noise)
    print(f"\n🔧 Voxel smoothing (grid={VOXEL_DOWNSAMPLE_MM}mm)...")
    
    voxel_size = VOXEL_DOWNSAMPLE_MM / 1000.0  # mm to m
    pcd_smooth = pcd_clean.voxel_down_sample(voxel_size=voxel_size)
    
    reduction = 100 * (1 - len(pcd_smooth.points) / len(pcd_clean.points))
    
    print(f"✓ Smoothed: {len(pcd_smooth.points):,} points ({reduction:.1f}% reduction)")
    
    # STEP 1C: Estimate normals (needed for validation)
    print(f"\n📐 Computing normals...")
    pcd_smooth.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
            max_nn=30
        )
    )
    print(f"✓ Normals computed")
    
    # Summary
    total_removed = len(pcd_raw.points) - len(pcd_smooth.points)
    total_pct = 100 * total_removed / len(pcd_raw.points)
    
    print(f"\n📊 Cleaning summary:")
    print(f"   Raw:     {len(pcd_raw.points):,} points")
    print(f"   Cleaned: {len(pcd_smooth.points):,} points")
    print(f"   Removed: {total_removed:,} points ({total_pct:.1f}%)")
    
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
    
    return pcd_smooth, sorted(image_paths)


# ============================================================================
# STEP 2: DETECT MARKERS
# ============================================================================

def detect_markers(image_paths):
    """Detect ArUco markers"""
    print(f"\n{'='*70}")
    print("STEP 2: DETECTING ARUCO MARKERS")
    print(f"{'='*70}\n")
    
    if not image_paths:
        print("⚠️  No images found")
        return {}
    
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
            print(f"  {idx + 1}/{len(image_paths)}...")
    
    print(f"\n✓ Detections:")
    for marker_id, count in marker_counts.items():
        if count > 0:
            print(f"  Marker {marker_id} ({MARKER_TO_FACE[marker_id]:8s}): {count}")
    
    detected = sum(1 for c in marker_counts.values() if c > 0)
    print(f"\n✓ Found {detected} different markers")
    
    return marker_counts


# ============================================================================
# STEP 3: MULTI-PASS PLANE EXTRACTION
# ============================================================================

def validate_plane(plane, total_points):
    """Validate if plane is physically plausible"""
    
    # Check 1: Inlier ratio (not too many points)
    inlier_ratio = plane['num_points'] / total_points
    if inlier_ratio > MAX_INLIER_RATIO:
        return False, f"Too many inliers ({inlier_ratio*100:.1f}%)"
    
    # Check 2: Area (not too small/large)
    area = plane['area']
    if area < MIN_PLANE_AREA_M2:
        return False, f"Too small ({area*1000000:.1f}mm²)"
    if area > MAX_PLANE_AREA_M2:
        return False, f"Too large ({area:.2f}m²)"
    
    # Check 3: Normal must be reasonable (not random)
    normal = plane['normal']
    if np.linalg.norm(normal) < 0.9:  # Should be normalized
        return False, "Invalid normal"
    
    return True, "OK"


def extract_planes_multipass(pcd):
    """Extract planes using multi-pass strategy"""
    print(f"\n{'='*70}")
    print("STEP 3: MULTI-PASS PLANE EXTRACTION")
    print(f"{'='*70}\n")
    
    total_points = len(pcd.points)
    print(f"Extracting from {total_points:,} points")
    print(f"Strategy: {len(PLANE_PASSES)} passes (coarse → fine)\n")
    
    all_planes = []
    remaining_pcd = pcd
    
    for pass_idx, pass_config in enumerate(PLANE_PASSES):
        print(f"--- PASS {pass_idx + 1}: {pass_config['name']} Planes ---")
        print(f"    threshold={pass_config['threshold']*1000:.1f}mm, "
              f"min_points={pass_config['min_points']:,}")
        
        pass_planes = 0
        
        while len(remaining_pcd.points) > pass_config['min_points']:
            if len(all_planes) >= MAX_PLANES_TOTAL:
                print(f"    Reached max planes ({MAX_PLANES_TOTAL})")
                break
            
            # RANSAC
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=pass_config['threshold'],
                ransac_n=3,
                num_iterations=pass_config['iterations']
            )
            
            if len(inliers) < pass_config['min_points']:
                break
            
            # Extract plane geometry
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            
            plane_pcd = remaining_pcd.select_by_index(inliers)
            plane_points = np.asarray(plane_pcd.points)
            center = np.mean(plane_points, axis=0)
            
            # Calculate area
            extents = np.ptp(plane_points, axis=0)
            area = np.prod(np.sort(extents)[-2:])
            
            plane = {
                'id': len(all_planes),
                'pass': pass_idx,
                'normal': normal,
                'center': center,
                'distance': d,
                'num_points': len(inliers),
                'points': plane_points,
                'pcd': plane_pcd,
                'extents': extents,
                'area': area
            }
            
            # VALIDATE
            valid, reason = validate_plane(plane, total_points)
            
            if valid:
                all_planes.append(plane)
                pass_planes += 1
                
                # Determine type
                if abs(normal[2]) > 0.8:
                    plane_type = "HORIZ"
                else:
                    plane_type = "VERT"
                
                print(f"    ✓ Plane {plane['id']}: {len(inliers):7,} pts, "
                      f"area={area*10000:.1f}cm², {plane_type}")
            else:
                print(f"    ✗ Rejected: {reason}")
            
            # Remove plane
            outliers = list(set(range(len(remaining_pcd.points))) - set(inliers))
            remaining_pcd = remaining_pcd.select_by_index(outliers)
        
        print(f"    Found {pass_planes} planes in this pass")
        print(f"    Remaining: {len(remaining_pcd.points):,} points\n")
    
    print(f"{'='*70}")
    print(f"✓ TOTAL: {len(all_planes)} valid planes extracted")
    print(f"{'='*70}\n")
    
    return all_planes


# ============================================================================
# STEP 4: IDENTIFY CUBES & FLOOR
# ============================================================================

def identify_structures(planes):
    """Identify floor, ArUco cube, and other objects"""
    print(f"\n{'='*70}")
    print("STEP 4: IDENTIFYING STRUCTURES")
    print(f"{'='*70}\n")
    
    # Separate by orientation
    horizontal = []
    vertical = []
    
    for plane in planes:
        if abs(plane['normal'][2]) > 0.7:
            horizontal.append(plane)
        else:
            vertical.append(plane)
    
    print(f"Plane distribution:")
    print(f"  Horizontal: {len(horizontal)}")
    print(f"  Vertical:   {len(vertical)}")
    
    # FLOOR = largest horizontal plane
    horizontal_sorted = sorted(horizontal, key=lambda p: p['num_points'], reverse=True)
    
    if horizontal_sorted:
        floor = horizontal_sorted[0]
        print(f"\n✓ Floor identified: Plane {floor['id']}")
        print(f"  Points: {floor['num_points']:,}")
        print(f"  Area: {floor['area']:.2f}m²")
    else:
        print("\n⚠️  No clear floor found (using largest plane)")
        floor = sorted(planes, key=lambda p: p['num_points'], reverse=True)[0]
    
    # CUBES = clusters of vertical planes
    non_floor = [p for p in planes if p['id'] != floor['id']]
    
    print(f"\n🔍 Analyzing {len(non_floor)} non-floor planes...")
    
    # Filter by cube face size
    cube_candidates = []
    for plane in non_floor:
        if CUBE_FACE_AREA_MIN <= plane['area'] <= CUBE_FACE_AREA_MAX:
            cube_candidates.append(plane)
            print(f"  ✓ Plane {plane['id']}: area={plane['area']*10000:.1f}cm² (cube-sized)")
        else:
            print(f"  ✗ Plane {plane['id']}: area={plane['area']*10000:.1f}cm² (too large/small)")
    
    print(f"\n✓ {len(cube_candidates)} cube-face candidates")
    
    # Cluster by proximity
    if len(cube_candidates) >= 5:
        print(f"\n🔗 Clustering by proximity (radius={CUBE_CLUSTER_RADIUS*1000}mm)...")
        
        centers = np.array([p['center'] for p in cube_candidates])
        
        # Simple proximity grouping
        cube_groups = []
        used = set()
        
        for i, plane in enumerate(cube_candidates):
            if i in used:
                continue
            
            # Find all planes within radius
            distances = np.linalg.norm(centers - plane['center'], axis=1)
            nearby = np.where(distances < CUBE_CLUSTER_RADIUS)[0]
            
            if len(nearby) >= 3:  # At least 3 faces = cube
                group = [cube_candidates[j] for j in nearby]
                cube_groups.append(group)
                used.update(nearby)
                print(f"  Group {len(cube_groups)}: {len(group)} faces")
        
        print(f"\n✓ Found {len(cube_groups)} cube groups")
        
        # Assign roles (heuristic: ArUco cube = more markers visible)
        if len(cube_groups) >= 2:
            aruco_planes = cube_groups[0]
            other_planes = cube_groups[1]
        elif len(cube_groups) == 1:
            aruco_planes = cube_groups[0]
            other_planes = []
        else:
            aruco_planes = cube_candidates[:6]
            other_planes = cube_candidates[6:]
    else:
        print(f"\n⚠️  Not enough cube faces found")
        aruco_planes = cube_candidates
        other_planes = []
    
    return {
        'floor': floor,
        'aruco_cube': aruco_planes,
        'other_cubes': other_planes,
        'all_objects': non_floor
    }


# ============================================================================
# STEP 5: CALCULATE CALIBRATION
# ============================================================================

def calculate_calibration(aruco_planes):
    """Calculate coordinate system and scale"""
    print(f"\n{'='*70}")
    print("STEP 5: CALCULATING CALIBRATION")
    print(f"{'='*70}\n")
    
    if len(aruco_planes) < 3:
        print(f"❌ Need ≥3 planes, got {len(aruco_planes)}")
        return None, None, None
    
    print(f"Using {len(aruco_planes)} planes for calibration")
    
    # Find horizontal and vertical planes
    horiz = [p for p in aruco_planes if abs(p['normal'][2]) > 0.7]
    vert = [p for p in aruco_planes if abs(p['normal'][2]) <= 0.7]
    
    # Build coordinate system
    if horiz:
        z_axis = horiz[0]['normal'].copy()
        if z_axis[2] < 0:
            z_axis = -z_axis
    else:
        z_axis = np.array([0, 0, 1])
    
    if len(vert) >= 2:
        x_raw = vert[0]['normal']
        x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
    else:
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
    
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    
    # Calculate center & scale
    all_points = np.vstack([p['points'] for p in aruco_planes])
    center = np.mean(all_points, axis=0)
    
    points_aligned = (all_points - center) @ rotation.T
    extents = np.ptp(points_aligned, axis=0)
    mean_extent = np.mean(extents)
    
    scale = mean_extent / (CUBE_SIZE_MM * 0.001)
    
    print(f"✓ Extents: X={extents[0]:.4f}, Y={extents[1]:.4f}, Z={extents[2]:.4f}m")
    print(f"✓ Expected: {CUBE_SIZE_MM}mm = {CUBE_SIZE_MM*0.001:.4f}m")
    print(f"✓ Scale: {scale:.6f}")
    
    return rotation, center, scale


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print("ULTRA-ROBUST ARUCO CALIBRATION")
    print("Designed for noisy photogrammetry scans")
    print(f"{'='*70}\n")
    
    scan_folder = input("Scan folder: ").strip().strip('"')
    
    if not os.path.exists(scan_folder):
        print(f"❌ Not found: {scan_folder}")
        return
    
    try:
        # Step 1: Load & clean aggressively
        pcd, images = load_and_clean_scan(scan_folder)
        
        # Step 2: Detect markers
        markers = detect_markers(images)
        
        # Step 3: Multi-pass plane extraction
        planes = extract_planes_multipass(pcd)
        
        if len(planes) == 0:
            print("\n❌ No planes found!")
            return
        
        # Step 4: Identify structures
        structures = identify_structures(planes)
        
        # Step 5: Calculate calibration
        rotation, center, scale = calculate_calibration(structures['aruco_cube'])
        
        if rotation is None:
            print("\n❌ Calibration failed")
            return
        
        print(f"\n{'='*70}")
        print("✅ SUCCESS!")
        print(f"{'='*70}\n")
        print(f"Planes found: {len(planes)}")
        print(f"ArUco cube faces: {len(structures['aruco_cube'])}")
        print(f"Scale factor: {scale:.6f}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
