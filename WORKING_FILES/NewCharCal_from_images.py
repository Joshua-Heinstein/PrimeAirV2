"""
File: NewCharCal_with_images.py
Author: Santiago Burgos_fallon
Date: 2025-02-14
Version: 1.0

Description:
    This script performs camera calibration using a set of pre-captured images of a ChArUco board.
    Instead of capturing frames from a live camera feed, it loads a series of calibration images
    from a specified directory (e.g., "calibration_images/*.jpg"). The script then processes each image
    to detect ArUco markers and interpolate ChArUco board corners, collecting valid calibration data.
    
    The calibration data (ChArUco corners and IDs) are used with OpenCV's ArUco module to compute the camera's 
    intrinsic parameters (camera matrix) and distortion coefficients. The setup is tailored for high-resolution 
    (4K) images.
    
Usage:
    1. Ensure your calibration images are placed in the specified directory with the appropriate file extension.
    2. Modify the `image_paths` glob pattern and `num_images` variable if necessary.
    3. Run the script; it will process each image, display processing status, and output the calibration results.
    
Dependencies:
    - Python 3.x
    - OpenCV with the ArUco module enabled
    - NumPy
    - glob (standard library)
    
Output:
    - Prints the computed camera (intrinsic) matrix and distortion coefficients to the console.
    
Notes:
    - For visualization purposes, code lines for displaying the processed image are included but commented out.
    - Ensure that your images provide a sufficient view of the ChArUco board for successful calibration.
"""


import cv2
import numpy as np
import glob

# ArUco dictionary and ChArUco board parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Create a ChArUco board: (squaresX, squaresY, squareLength, markerLength)
board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.02, aruco_dict)

# Calibration parameters
all_corners = []
all_ids = []
image_size = None
frames_processed = 0
num_images = 20  # Number of images to use for calibration

# Replace with your image directory and file extension as needed.
image_paths = sorted(glob.glob("../../calib_img/*.jpg"))

if len(image_paths) < num_images:
    print(f"Error: Found only {len(image_paths)} images, but need at least {num_images}.")
    exit(1)

print("Processing calibration images...")

# Process only the first 'num_images' from the list
for i, image_path in enumerate(image_paths[:num_images]):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Warning: Unable to load image: {image_path}")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    
    # Refine with the ChArUco board if markers were detected
    charuco_corners = None
    charuco_ids = None
    if ids is not None and len(ids) > 0:
        # The function returns (retval, charucoCorners, charucoIds); we ignore retval here.
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        
        # Optionally, draw detected markers and corners for visualization
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
    
    # (Optional) Show the image briefly for visualization; press any key to continue
    # cv2.imshow("Calibration", frame)
    # cv2.waitKey(500)
    
    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        image_size = gray.shape[::-1]  # (width, height)
        frames_processed += 1
        print(f"✅ Processed image {i+1}/{num_images}")
    else:
        print(f"⚠️ No valid ChArUco corners found in image {i+1}.")

# Clean up any open windows (if visualization was enabled)
cv2.destroyAllWindows()

if frames_processed == 0:
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
