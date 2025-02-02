import cv2
import numpy as np
import apriltag

# -------------------------
# Camera Setup and Calibration
# -------------------------
# Open the default camera (index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Replace these with your actual calibration parameters.
# The camera matrix (intrinsics) and distortion coefficients must be accurate.
cameraMatrix = np.array([[597.348,   0.0,   320.993],
                         [  0.0,   610.151, 289.052],
                         [  0.0,     0.0,     1.0]], dtype=np.float64)
distCoeffs = np.array([0.94, -141.48, 0, -0.02, 4017.88], dtype=np.float64)

# -------------------------
# AprilTag and Pose Setup
# -------------------------
# Define the tag size (length of one side) in meters.
# For example, if your tag is 1 meter x 1 meter:
tag_size = .15

# Define the 3D coordinates of the AprilTag corners in the tag's coordinate system.
# Here the tag is centered at (0, 0, 0) so that:
# - The bottom-left corner is at (-tag_size/2, -tag_size/2, 0)
# - The bottom-right at ( tag_size/2, -tag_size/2, 0), etc.
obj_pts = np.array([
    [-tag_size/2, -tag_size/2, 0.0],  # Bottom-left corner
    [ tag_size/2, -tag_size/2, 0.0],  # Bottom-right corner
    [ tag_size/2,  tag_size/2, 0.0],  # Top-right corner
    [-tag_size/2,  tag_size/2, 0.0]   # Top-left corner
], dtype=np.float64)

# If you need a different coordinate system for the tag (for example, bottom-left corner at (0,0,0)),
# you could define the object points like this:
# obj_pts = np.array([
#     [0,         0,         0.0],
#     [tag_size,  0,         0.0],
#     [tag_size,  tag_size,  0.0],
#     [0,         tag_size,  0.0]
# ], dtype=np.float64)

# Initialize the AprilTag detector.
options = apriltag.DetectorOptions(families="tag16h5")
detector = apriltag.Detector(options)

print("[INFO] Starting real-time AprilTag detection. Press 'q' to exit.")

# -------------------------
# Main Loop: Capture and Process Video Frames
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to grayscale (improves detection speed/accuracy)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the frame
    results = detector.detect(gray)

    for r in results:
        # Convert the corner coordinates to integers for drawing.
        pts = r.corners.astype(int)
        (ptA, ptB, ptC, ptD) = pts

        # Draw the bounding box of the detected AprilTag.
        cv2.line(frame, tuple(ptA), tuple(ptB), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptB), tuple(ptC), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptC), tuple(ptD), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptD), tuple(ptA), (0, 255, 0), 2)

        # Draw the center point of the tag.
        cX, cY = int(r.center[0]), int(r.center[1])
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        # -------------------------
        # Pose Estimation using solvePnP
        # -------------------------
        # Use the known 3D object points (obj_pts) and the 2D image points (r.corners)
        # to compute the pose of the tag relative to the camera.
        image_pts = r.corners.astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(obj_pts, image_pts, cameraMatrix, distCoeffs)
        if success:
            # Convert rotation vector to rotation matrix.
            R, _ = cv2.Rodrigues(rvec)
            # Invert the pose:
            # solvePnP gives the tag's pose in the camera coordinate system.
            # To get the camera (or drone) position in the tag's coordinate system, compute:
            drone_position = -R.T @ tvec

            # Extract the x, y, and z coordinates (in meters if tag_size is in meters).
            pos_x = drone_position[0][0]
            pos_y = drone_position[1][0]
            pos_z = drone_position[2][0]

            # Overlay the position information on the frame.
            text = f"X: {pos_x:.2f} m, Y: {pos_y:.2f} m, Z: {pos_z:.2f} m"
            # Place the text above the tag (adjust the position as needed).
            cv2.putText(frame, text, (ptA[0], ptA[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Also print the position to the console.
            print("Drone position relative to tag:", drone_position.flatten())
        else:
            cv2.putText(frame, "Pose estimation failed", (ptA[0], ptA[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame with all overlays.
    cv2.imshow("AprilTag Pose Estimation", frame)

    # Exit if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------
# Cleanup
# -------------------------
cap.release()
cv2.destroyAllWindows()
