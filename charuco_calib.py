# charuco_calib.py
# Troy Kaufman
# tkaufman@g.hmc.edu
# 2/5/2025

import cv2
import numpy as np
import pyrealsense2 as rs

# ArUco dictionary and ChArUco board parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard_create(5, 7, 0.04, 0.02, aruco_dict)  # (squaresX, squaresY, squareLength, markerLength)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Parameters for calibration
all_corners = []
all_ids = []
image_size = None
frames_captured = 0
num_images = 20  # Number of frames to capture before calibrating

print("Press 'c' to capture a frame for calibration, and 'q' to quit.")

while True:
    # Capture frame from RealSense
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    frame = np.asanyarray(color_frame.get_data())

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if ids is not None:
        # Refine detection using ChArUco board
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

        # Draw detected markers and ChArUco corners
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if charuco_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

        # Show the frame
        cv2.imshow("RealSense ChArUco Calibration", frame)

        # Capture frame when 'c' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and charuco_corners is not None and charuco_ids is not None:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            image_size = gray.shape[::-1]
            frames_captured += 1
            print(f"Captured frame {frames_captured}/{num_images}")

        # Break loop when enough images are collected
        if frames_captured >= num_images:
            break

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop RealSense pipeline
pipeline.stop()
cv2.destroyAllWindows()

# Run ChArUco-based camera calibration
print("Running camera calibration...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_corners, all_ids, board, image_size, None, None
)

# Display calibration results
print("\nCamera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# Save calibration results to a file
np.savez("camera_calibration_realsense.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

print("Calibration complete. Results saved to 'camera_calibration_realsense.npz'.")
