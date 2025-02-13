# charuco_calib.py
# Troy Kaufman
# tkaufman@g.hmc.edu
# 2/5/2025

##############################################################################
# Script that calibrates camera by using OpenCV's Charuco calibration library
##############################################################################
import cv2
import numpy as np
import pyrealsense2 as rs

# ArUco dictionary and ChArUco board parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.02, aruco_dict)  # (squaresX, squaresY, squareLength, markerLength)

# Initialize RealSense pipeline with both color and depth streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Get stream profile and create alignment object
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
depth_intrinsics = depth_profile.get_intrinsics()
color_intrinsics = color_profile.get_intrinsics()

# Create alignment primitive with color as its target stream
align = rs.align(rs.stream.color)

# Parameters for calibration
all_corners = []
all_ids = []
image_size = None
frames_captured = 0
num_images = 20  # Number of frames to capture before calibrating

print("CALIBRATION PHASE")
print("Press 'c' to capture a frame for calibration. Need", num_images, "frames.")

# Calibration Phase
while frames_captured < num_images:
    # Wait for a coherent pair of frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    
    if not color_frame:
        continue
        
    frame = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
        # Refine detection using ChArUco board
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        
        # Draw detected markers and ChArUco corners
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            
        cv2.imshow("Calibration", frame)
        
        # Capture frame when 'c' is pressed...very helpful
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if charuco_corners is not None and charuco_ids is not None:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                image_size = gray.shape[::-1]
                frames_captured += 1
                print(f"✅ Captured frame {frames_captured}/{num_images}")
            else:
                print("⚠️ No valid ChArUco corners found. Frame NOT captured.")

cv2.destroyWindow("Calibration")
print("\nRunning camera calibration...")

# Perform calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print("\nCalibration complete!")
print("\nCamera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# currently very cluttered on the screen. Will need to clean up but the results are looking great!!!
def get_3d_coordinates(depth_frame, color_intrin, pixel_coords):
    """Convert pixel coordinates to 3D world coordinates using depth data."""
    depth = depth_frame.get_distance(int(pixel_coords[0]), int(pixel_coords[1]))
    return rs.rs2_deproject_pixel_to_point(color_intrin, [pixel_coords[0], pixel_coords[1]], depth)

print("\nStarting 3D tracking...")
print("Press 'q' to quit.") # lots of latency...instead press <ctrl> + <c> to exit tracking mode back into terminal

# create way to pipe this data to a log file

# find way to obtain 3d coord info without a hard connection (IoT by using esp32 board with integrated wifi...might be too slow though and it has limited range)

# 3D Tracking Phase
try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = aruco_detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Refine detection using ChArUco board
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)

            if charuco_corners is not None and charuco_ids is not None:
                # Draw detected markers and ChArUco corners
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids)

                # Estimate board pose
                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)

                if ret:
                    # Draw coordinate system
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                    
                    # Get 3D coordinates for each ChArUco corner
                    for i in range(len(charuco_corners)):
                        corner = charuco_corners[i][0]
                        corner_id = charuco_ids[i][0]
                        
                        # Get 3D coordinates using depth
                        world_point = get_3d_coordinates(depth_frame, color_intrinsics, corner)
                        
                        # Draw corner ID and 3D coordinates
                        cv2.putText(color_image, f"ID:{corner_id} ({world_point[0]:.2f}, {world_point[1]:.2f}, {world_point[2]:.2f}m)",
                                  (int(corner[0]) + 10, int(corner[1]) + 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("RealSense ChArUco 3D Tracking", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()