###DEEPSEEK###

import cv2
import numpy as np
import cv2.aruco as aruco

# Define ChArUco board parameters
CHARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
SQUARE_LENGTH = 0.04  # MUST match your printed board (meters)
MARKER_LENGTH = 0.02  # MUST match your printed board (meters)
CHARUCO_BOARD = aruco.CharucoBoard((5, 7), SQUARE_LENGTH, MARKER_LENGTH, CHARUCO_DICT)

# Lists to store detected corners
all_corners = []
all_ids = []
image_size = None

def capture_images(video_source=1, num_images=20):
    """Captures images from the camera for calibration."""
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # Set resolution (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    
    count = 0
    detector_params = aruco.DetectorParameters()  # Tune parameters if needed
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect markers with grayscale and parameters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            gray, CHARUCO_DICT, parameters=detector_params
        )
        
        # Visual feedback: Draw detected markers and rejected candidates
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        frame = aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Process the image (even if no markers detected)
            success = process_image(frame)
            if success:
                count += 1
                print(f"Valid capture: {count}/{num_images}")
            else:
                print("No board detected! Adjust position and try again.")
    
    cap.release()
    cv2.destroyAllWindows()

def process_image(image):
    """Process image to detect ChArUco corners. Returns success flag."""
    global image_size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, CHARUCO_DICT)
    
    if ids is None:
        return False  # No markers detected
    
    # Interpolate ChArUco corners
    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        corners, ids, gray, CHARUCO_BOARD
    )
    
    if (
        charuco_corners is not None 
        and charuco_ids is not None 
        and len(charuco_corners) >= 5  # Minimum for calibration
    ):
        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        if image_size is None:
            image_size = gray.shape[::-1]
        return True
    else:
        return False  # Not enough corners for calibration

def calibrate_camera():
    """Perform camera calibration."""
    if len(all_corners) < 10:
        print(f"Error: Only {len(all_corners)} valid images. Need at least 10.")
        return
    
    ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, CHARUCO_BOARD, image_size, None, None
    )
    
    print("Calibration successful!")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    np.save("camera_matrix.npy", camera_matrix)
    np.save("dist_coeffs.npy", dist_coeffs)

if __name__ == "__main__":
    capture_images()
    calibrate_camera()