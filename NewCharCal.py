# NewCharCal.py
# Modified for 4K capture
# 2/14/2025

import cv2
import numpy as np

# ArUco dictionary and ChArUco board parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Create a ChArUco board: (squaresX, squaresY, squareLength, markerLength)
board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.02, aruco_dict)

# Open the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit(1)

# Set resolution to 4K (3840 x 2160)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# Calibration parameters
all_corners = []
all_ids = []
image_size = None
frames_captured = 0
num_images = 20  # Number of frames to capture before calibrating

print("CALIBRATION PHASE")
print(f"Press 'c' to capture a frame for calibration. Need {num_images} frames.")

while frames_captured < num_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    
    # If markers were detected, try to refine with the ChArUco board
    charuco_corners = None
    charuco_ids = None
    if ids is not None and len(ids) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        # Draw detected markers and corners for visualization
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
    
    cv2.imshow("Calibration", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            image_size = gray.shape[::-1]  # (width, height)
            frames_captured += 1
            print(f"✅ Captured frame {frames_captured}/{num_images}")
        else:
            print("⚠️ No valid ChArUco corners found. Frame NOT captured.")
    elif key == ord('q'):
        print("Exiting calibration.")
        break

cv2.destroyWindow("Calibration")
cap.release()

if frames_captured == 0:
    print("No calibration data collected. Exiting.")
    exit(1)

print("\nRunning camera calibration...")

# Perform calibration using the collected data
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print("\nCalibration complete!")
print("Camera (Intrinsic) Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
