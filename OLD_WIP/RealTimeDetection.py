import cv2
import numpy as np
import pupil_apriltags as apriltag

# -------------------------
# Camera Setup and Calibration
# -------------------------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Replace these with your actual calibration parameters.
cameraMatrix = np.array([[2.03955702e+03,   0.0,           9.64412115e+02],
                         [  0.0,           2.03694742e+03,   7.27311542e+02],
                         [  0.0,             0.0,           1.0]], dtype=np.float64)
distCoeffs = np.array([-1.69118596e-02, -1.00094621e-01, -1.53552265e-03, -4.94214647e-03, 1.91215518e+00], dtype=np.float64)

# -------------------------
# AprilTag and Pose Setup
# -------------------------
# Define the tag size (the length of one side) in meters.
tag_size = 0.155  # Adjust to your tag's physical size

# Define the 3D coordinates of the AprilTag corners in the tag's coordinate system.
# (Assuming the tag is centered at (0, 0, 0) and lies in the XY plane)
obj_pts = np.array([
    [-tag_size/2, -tag_size/2, 0.0],  # Bottom-left corner
    [ tag_size/2, -tag_size/2, 0.0],  # Bottom-right corner
    [ tag_size/2,  tag_size/2, 0.0],  # Top-right corner
    [-tag_size/2,  tag_size/2, 0.0]   # Top-left corner
], dtype=np.float64)

# Initialize the AprilTag detector.
detector = apriltag.Detector(families="tag16h5")


# Define the known global pose (rotation and translation) for each tag.
# In this example, we assume the tags are arranged in a square with side length 0.3 m.
# The rotation is assumed to be identity (i.e. the tag's axes are aligned with the global axes).
# Adjust these values to match your actual setup.
tag_global_poses = {
    0: (np.eye(3), np.array([0.0, 0.0, 0.0])),    # Tag 0 at origin
    1: (np.eye(3), np.array([0.40, 0.0, 0.0])),    # Tag 1 0.40 m to the right
    2: (np.eye(3), np.array([0.40, 0.40, 0.0])),   # Tag 2 0.40 m right and 0.40 m up
    3: (np.eye(3), np.array([0.0, 0.40, 0.0]))     # Tag 3 0.40 m up
}

print("[INFO] Starting real-time AprilTag detection. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale for detection.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags.
    results = detector.detect(gray)

    # List to collect drone (camera) positions in the global coordinate system.
    global_positions = []

    for r in results:
        # Draw bounding box for visualization.
        pts = r.corners.astype(int)
        (ptA, ptB, ptC, ptD) = pts
        cv2.line(frame, tuple(ptA), tuple(ptB), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptB), tuple(ptC), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptC), tuple(ptD), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptD), tuple(ptA), (0, 255, 0), 2)

        # Draw the center point.
        cX, cY = int(r.center[0]), int(r.center[1])
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        # -------------------------
        # Pose Estimation using solvePnP
        # -------------------------
        image_pts = r.corners.astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(obj_pts, image_pts, cameraMatrix, distCoeffs)
        if success:
            # Convert rotation vector to rotation matrix.
            R, _ = cv2.Rodrigues(rvec)
            # Invert the pose to get the camera (drone) position in the tag's coordinate system.
            drone_pos_tag = -R.T @ tvec  # 3x1 vector

            # Check if this tag's ID is in our known global poses.
            tag_id = r.tag_id
            if tag_id in tag_global_poses:
                # Retrieve the global pose of the tag.
                R_tag_global, t_tag_global = tag_global_poses[tag_id]
                # Transform the drone position from the tag's coordinate system to the global system.
                # Since R_tag_global is identity in this example, this simplifies to:
                global_pos = drone_pos_tag.flatten() + t_tag_global
                global_positions.append(global_pos)

                # Overlay the position relative to this tag.
                text = f"Tag {tag_id} -> X:{global_pos[0]:.2f}, Y:{global_pos[1]:.2f}, Z:{global_pos[2]:.2f}"
                cv2.putText(frame, text, (ptA[0], ptA[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"Drone position relative to tag {tag_id} (global):", global_pos)
            else:
                cv2.putText(frame, f"Tag {tag_id} unknown", (ptA[0], ptA[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Pose estimation failed", (ptA[0], ptA[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # If at least one tag with known global pose was detected, average the positions.
    if global_positions:
        global_positions = np.array(global_positions)
        avg_position = np.mean(global_positions, axis=0)
        avg_text = f"Avg Pos -> X:{avg_position[0]:.2f}, Y:{avg_position[1]:.2f}, Z:{avg_position[2]:.2f}"
        cv2.putText(frame, avg_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print("Estimated Drone Global Position:", avg_position)

    cv2.imshow("AprilTag Pose Estimation", frame)

    # Exit loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup.
cap.release()
cv2.destroyAllWindows()
