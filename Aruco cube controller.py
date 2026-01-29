"""
ArUco Cube Controller - Real-Time 3D Tracking
==============================================

Use your physical ArUco calibration cube as a 3D controller!
Move it around, rotate it → the virtual cube follows in real-time.

Controls:
- ESC: Exit
- R: Reset virtual cube position
- S: Screenshot
- C: Toggle camera/cube view

Requirements:
- pip install opencv-python opencv-contrib-python numpy
- Webcam
- ArUco calibration cube (40×40×40mm with markers)

Author: MRAC Team
Date: 2026-01-27
"""

import cv2
import numpy as np
import time
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

# ArUco settings
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 32  # Physical marker size in mm
CUBE_SIZE_MM = 40    # Physical cube size in mm

# Marker to face mapping
MARKER_TO_FACE = {
    0: 'BOTTOM',
    1: 'TOP',
    2: 'FRONT',
    3: 'BACK',
    4: 'LEFT',
    5: 'RIGHT'
}

# Camera calibration (rough estimate - works for most webcams)
# For better results, calibrate your camera properly!
CAMERA_MATRIX = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

DIST_COEFFS = np.zeros((4, 1))  # Assuming no distortion

# Virtual cube settings
VIRTUAL_CUBE_SIZE = 100  # Size of virtual cube in pixels
SMOOTHING_FRAMES = 5     # Number of frames to smooth motion


# ============================================================================
# POSE ESTIMATION
# ============================================================================

def estimate_pose_from_markers(corners, ids, marker_size):
    """
    Estimate 3D pose of the cube from detected ArUco markers
    Returns: rotation_vector, translation_vector, or None if failed
    """
    
    if ids is None or len(ids) == 0:
        return None, None
    
    # Use first detected marker for pose estimation
    # In production, you'd combine multiple markers for robustness
    
    # Define 3D coordinates of marker corners (in mm)
    # Marker is centered at origin, size = marker_size
    half_size = marker_size / 2
    obj_points = np.array([
        [-half_size, -half_size, 0],
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0],
        [-half_size,  half_size, 0]
    ], dtype=np.float32)
    
    # Get 2D image points from detected marker
    img_points = corners[0][0].astype(np.float32)
    
    # Solve PnP (Perspective-n-Point)
    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points,
        CAMERA_MATRIX,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    
    if success:
        return rvec, tvec
    
    return None, None


# ============================================================================
# 3D VISUALIZATION
# ============================================================================

def draw_cube_3d(img, rvec, tvec, cube_size, camera_matrix, dist_coeffs, color=(0, 255, 0)):
    """Draw a 3D cube using the estimated pose"""
    
    if rvec is None or tvec is None:
        return img
    
    # Define 3D cube vertices (in mm)
    half = cube_size / 2
    cube_points = np.array([
        [-half, -half, 0],      # Bottom face
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0],
        [-half, -half, cube_size],  # Top face
        [ half, -half, cube_size],
        [ half,  half, cube_size],
        [-half,  half, cube_size]
    ], dtype=np.float32)
    
    # Project 3D points to 2D image plane
    img_points, _ = cv2.projectPoints(
        cube_points,
        rvec, tvec,
        camera_matrix,
        dist_coeffs
    )
    
    img_points = img_points.reshape(-1, 2).astype(int)
    
    # Draw cube edges
    # Bottom face
    for i in range(4):
        pt1 = tuple(img_points[i])
        pt2 = tuple(img_points[(i + 1) % 4])
        cv2.line(img, pt1, pt2, color, 2)
    
    # Top face
    for i in range(4):
        pt1 = tuple(img_points[i + 4])
        pt2 = tuple(img_points[((i + 1) % 4) + 4])
        cv2.line(img, pt1, pt2, color, 2)
    
    # Vertical edges
    for i in range(4):
        pt1 = tuple(img_points[i])
        pt2 = tuple(img_points[i + 4])
        cv2.line(img, pt1, pt2, color, 2)
    
    # Draw coordinate axes
    axis_length = cube_size * 1.5
    axis_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],  # X-axis (red)
        [0, axis_length, 0],  # Y-axis (green)
        [0, 0, axis_length]   # Z-axis (blue)
    ], dtype=np.float32)
    
    axis_img_points, _ = cv2.projectPoints(
        axis_points,
        rvec, tvec,
        camera_matrix,
        dist_coeffs
    )
    axis_img_points = axis_img_points.reshape(-1, 2).astype(int)
    
    origin = tuple(axis_img_points[0])
    cv2.line(img, origin, tuple(axis_img_points[1]), (0, 0, 255), 3)  # X = Red
    cv2.line(img, origin, tuple(axis_img_points[2]), (0, 255, 0), 3)  # Y = Green
    cv2.line(img, origin, tuple(axis_img_points[3]), (255, 0, 0), 3)  # Z = Blue
    
    return img


def draw_virtual_cube(img, rvec, tvec, position=(320, 240), size=100):
    """
    Draw a virtual cube that mimics the real cube's rotation
    Position is in 2D screen space
    """
    
    if rvec is None or tvec is None:
        return img
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # Define virtual cube vertices
    half = size / 2
    vertices_3d = np.array([
        [-half, -half, -half],
        [ half, -half, -half],
        [ half,  half, -half],
        [-half,  half, -half],
        [-half, -half,  half],
        [ half, -half,  half],
        [ half,  half,  half],
        [-half,  half,  half]
    ], dtype=np.float32)
    
    # Apply rotation
    rotated_vertices = vertices_3d @ rotation_matrix.T
    
    # Project to 2D (simple orthographic projection)
    vertices_2d = rotated_vertices[:, :2] + np.array(position)
    vertices_2d = vertices_2d.astype(int)
    
    # Sort vertices by Z-depth for proper drawing order
    z_depths = rotated_vertices[:, 2]
    
    # Define faces (each face is 4 vertices)
    faces = [
        [0, 1, 2, 3],  # Front
        [4, 5, 6, 7],  # Back
        [0, 1, 5, 4],  # Bottom
        [2, 3, 7, 6],  # Top
        [0, 3, 7, 4],  # Left
        [1, 2, 6, 5]   # Right
    ]
    
    face_colors = [
        (255, 100, 100),  # Front - Light red
        (255, 200, 100),  # Back - Orange
        (100, 255, 100),  # Bottom - Light green
        (100, 100, 255),  # Top - Light blue
        (255, 100, 255),  # Left - Magenta
        (100, 255, 255)   # Right - Cyan
    ]
    
    # Calculate average Z for each face (for depth sorting)
    face_depths = []
    for face in faces:
        avg_z = np.mean([z_depths[v] for v in face])
        face_depths.append(avg_z)
    
    # Sort faces by depth (back to front)
    sorted_faces = sorted(zip(faces, face_colors, face_depths), key=lambda x: x[2])
    
    # Draw faces
    for face, color, _ in sorted_faces:
        points = np.array([vertices_2d[v] for v in face])
        cv2.fillPoly(img, [points], color)
        cv2.polylines(img, [points], True, (0, 0, 0), 2)
    
    return img


# ============================================================================
# SMOOTHING
# ============================================================================

class PoseSmoothing:
    """Smooth pose estimation over multiple frames"""
    
    def __init__(self, buffer_size=5):
        self.rvec_buffer = deque(maxlen=buffer_size)
        self.tvec_buffer = deque(maxlen=buffer_size)
    
    def add(self, rvec, tvec):
        if rvec is not None and tvec is not None:
            self.rvec_buffer.append(rvec)
            self.tvec_buffer.append(tvec)
    
    def get_smoothed(self):
        if len(self.rvec_buffer) == 0:
            return None, None
        
        # Simple average smoothing
        rvec_smooth = np.mean(self.rvec_buffer, axis=0)
        tvec_smooth = np.mean(self.tvec_buffer, axis=0)
        
        return rvec_smooth, tvec_smooth


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    """Main AR tracking loop"""
    
    print(f"\n{'='*70}")
    print("ARUCO CUBE CONTROLLER - REAL-TIME 3D TRACKING")
    print(f"{'='*70}\n")
    
    print("Controls:")
    print("  ESC - Exit")
    print("  R   - Reset")
    print("  S   - Screenshot")
    print("  C   - Toggle view mode")
    print()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera!")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Setup ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Pose smoothing
    smoother = PoseSmoothing(SMOOTHING_FRAMES)
    
    # State
    show_camera = True
    fps_history = deque(maxlen=30)
    
    print("✓ Camera initialized")
    print("✓ ArUco detector ready")
    print("\n🎮 Point your ArUco cube at the camera!\n")
    
    while True:
        start_time = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(frame)
        
        # Estimate pose
        rvec, tvec = estimate_pose_from_markers(corners, ids, MARKER_SIZE_MM)
        smoother.add(rvec, tvec)
        rvec_smooth, tvec_smooth = smoother.get_smoothed()
        
        # Create output image
        if show_camera:
            output = frame.copy()
        else:
            output = np.zeros_like(frame)
        
        # Draw detected markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(output, corners, ids)
            
            # Draw marker IDs and face names
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in MARKER_TO_FACE:
                    center = corners[i][0].mean(axis=0).astype(int)
                    face_name = MARKER_TO_FACE[marker_id]
                    text = f"{marker_id}: {face_name}"
                    cv2.putText(output, text, tuple(center), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw 3D cube overlay on real cube
        if rvec_smooth is not None:
            output = draw_cube_3d(output, rvec_smooth, tvec_smooth, 
                                 CUBE_SIZE_MM, CAMERA_MATRIX, DIST_COEFFS,
                                 color=(0, 255, 255))
            
            # Draw virtual cube
            output = draw_virtual_cube(output, rvec_smooth, tvec_smooth,
                                      position=(500, 120), size=VIRTUAL_CUBE_SIZE)
            
            # Display rotation info
            rotation_matrix, _ = cv2.Rodrigues(rvec_smooth)
            euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]
            
            cv2.putText(output, f"Rotation: X={euler_angles[0]:.1f} Y={euler_angles[1]:.1f} Z={euler_angles[2]:.1f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        
        # Display info
        cv2.putText(output, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        markers_detected = len(ids) if ids is not None else 0
        status_color = (0, 255, 0) if markers_detected > 0 else (0, 0, 255)
        cv2.putText(output, f"Markers: {markers_detected}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Instructions
        cv2.putText(output, "ESC:Exit | R:Reset | S:Screenshot | C:Toggle", 
                   (10, output.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show output
        cv2.imshow('ArUco Cube Controller', output)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n👋 Exiting...")
            break
        elif key == ord('r') or key == ord('R'):
            print("🔄 Reset")
            smoother = PoseSmoothing(SMOOTHING_FRAMES)
        elif key == ord('s') or key == ord('S'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, output)
            print(f"📸 Screenshot saved: {filename}")
        elif key == ord('c') or key == ord('C'):
            show_camera = not show_camera
            mode = "Camera" if show_camera else "Virtual"
            print(f"👁️  View mode: {mode}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()q