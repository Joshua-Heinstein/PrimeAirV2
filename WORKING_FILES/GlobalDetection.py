"""
File Summary:
-------------
This script processes a batch of image files to estimate the global position of a drone (or camera)
by detecting AprilTags in each image. The main functions of the script are as follows:

1. Camera Calibration and Tag Setup:
   - Loads camera calibration parameters.
   - Defines the physical size and 3D coordinates of the AprilTag corners.

2. Global Frame Definition:
   - Converts known latitude/longitude coordinates of each AprilTag into a local Cartesian (metric) 
     frame using a simple flat-Earth approximation.
   - Uses one of the tag's coordinates as the origin, with a specified altitude.

3. Image Processing Loop:
   - Loads and converts each image to grayscale.
   - Detects AprilTags using an AprilTag detector.
   - Estimates the camera's pose relative to each detected tag using OpenCV's solvePnP.
   - Transforms each computed pose into the global (local metric) coordinate system using the 
     pre-defined tag positions.

4. Position Averaging and Conversion:
   - Averages the estimated positions from multiple tags (if available).
   - Converts the averaged local (east, north) coordinates back to geographic latitude and longitude.
   - Overlays and prints the final estimated global position (latitude, longitude, altitude) on the image.

Usage:
------
- Update the camera calibration parameters, tag size, and tag geographic coordinates as needed.
- Adjust the file path and pattern to match your set of image files.
- Run the script to sequentially process images and display the computed results.
"""



import cv2
import numpy as np
import pupil_apriltags as apriltag
import glob
import csv

# -------------------------
# Camera Setup and Calibration
# -------------------------
# (Replace these parameters with your actual calibration values.)

cameraMatrix = np.array([[1.43374197e+04, 0.00000000e+00, 2.02802116e+03],
                            [0.00000000e+00, 1.43107610e+04, 1.13273654e+03],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
distCoeffs = np.array([ 1.44235621e+00,  2.32751001e+01, -4.09845990e-02,  4.01781273e-02,-2.28454214e+03],
                       dtype=np.float64)

# cameraMatrix = np.array([[2.03955702e+03,   0.0,           9.64412115e+02],
#                          [  0.0,           2.03694742e+03,   7.27311542e+02],
#                          [  0.0,             0.0,           1.0]], dtype=np.float64)
# distCoeffs = np.array([-1.69118596e-02, -1.00094621e-01, -1.53552265e-03, -4.94214647e-03, 1.91215518e+00],
#                       dtype=np.float64)

# -------------------------
# AprilTag and Pose Setup
# -------------------------
tag_size = 0.155  # Adjust to your tag's physical size

# Define the 3D coordinates of the AprilTag corners in the tag's coordinate system.
obj_pts = np.array([
    [-tag_size/2, -tag_size/2, 0.0],
    [ tag_size/2, -tag_size/2, 0.0],
    [ tag_size/2,  tag_size/2, 0.0],
    [-tag_size/2,  tag_size/2, 0.0]
], dtype=np.float64)

# Initialize the AprilTag detector.
detector = apriltag.Detector(families="tag16h5")


# -------------------------
# Global (Local Metric) Frame Setup Using Lat/Lon
# -------------------------
# Known latitude/longitude for each tag (in degrees); adjust to your actual coordinates.
tag_global_coords = {
    0: (37.4219999, -122.0840575),
    1: (37.4219998, -122.0840574),
    2: (37.4219997, -122.0840573),
    3: (37.4219996, -122.0840572)
}

# Assume that all tags lie on a plane at a known altitude.
base_alt = 312.208  # altitude in meters

# Use tag 0 as the base coordinate for our local frame.
base_lat, base_lon = tag_global_coords[0]

def latlon_to_local(lat, lon, base_lat, base_lon):
    """
    Convert latitude and longitude (in degrees) to local Cartesian coordinates (in meters)
    using a flat-Earth approximation (x: east, y: north).
    """
    R = 6378137  # Earth's radius in meters
    dlat = np.deg2rad(lat - base_lat)
    dlon = np.deg2rad(lon - base_lon)
    x = R * dlon * np.cos(np.deg2rad(base_lat))
    y = R * dlat
    return x, y

def local_to_latlon(x, y, base_lat, base_lon):
    """
    Convert local Cartesian coordinates (in meters) back to latitude and longitude (in degrees)
    using a flat-Earth approximation.
    """
    R = 6378137
    dlat = y / R
    dlon = x / (R * np.cos(np.deg2rad(base_lat)))
    lat = base_lat + np.rad2deg(dlat)
    lon = base_lon + np.rad2deg(dlon)
    return lat, lon

# Convert each tag's lat/lon into the local (x, y, z) metric frame.
tag_global_local = {}
for tag_id, (lat, lon) in tag_global_coords.items():
    x, y = latlon_to_local(lat, lon, base_lat, base_lon)
    tag_global_local[tag_id] = np.array([x, y, base_alt])

# -------------------------
# Process Multiple Image Files
# -------------------------
# Update the path/pattern to match your image files.
image_files = sorted(glob.glob("../../apriltag_img_test/*.jpg"))
if not image_files:
    print("No image files found. Check your path and file pattern.")
    exit()

# Open a CSV file for writing the results.
csv_file = "drone_positions.csv"
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Latitude", "Longitude", "Altitude"])

    # Process each image file.
    for image_path in image_files:
        print(f"\nProcessing: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not load image:", image_path)
            writer.writerow([image_path, "NA", "NA", "NA"])
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)
        global_positions = []

        for r in results:
            # Draw the tag bounding box.
            pts = r.corners.astype(int)
            (ptA, ptB, ptC, ptD) = pts
            cv2.line(frame, tuple(ptA), tuple(ptB), (0, 255, 0), 2)
            cv2.line(frame, tuple(ptB), tuple(ptC), (0, 255, 0), 2)
            cv2.line(frame, tuple(ptC), tuple(ptD), (0, 255, 0), 2)
            cv2.line(frame, tuple(ptD), tuple(ptA), (0, 255, 0), 2)

            # Draw the center of the tag.
            cX, cY = int(r.center[0]), int(r.center[1])
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

            # Pose Estimation using solvePnP.
            image_pts = r.corners.astype(np.float64)
            success, rvec, tvec = cv2.solvePnP(obj_pts, image_pts, cameraMatrix, distCoeffs)
            if success:
                R, _ = cv2.Rodrigues(rvec)
                # Get the camera (drone) position in the tag's coordinate system.
                drone_pos_tag = -R.T @ tvec

                tag_id = r.tag_id
                if tag_id in tag_global_local:
                    tag_global_position = tag_global_local[tag_id]
                    # Transform the drone position into the global (local metric) frame.
                    global_pos = tag_global_position + drone_pos_tag.flatten()
                    global_positions.append(global_pos)

                    text = f"Tag {tag_id}: E:{global_pos[0]:.2f}, N:{global_pos[1]:.2f}, Alt:{global_pos[2]:.2f}"
                    cv2.putText(frame, text, (ptA[0], ptA[1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    print(f"Drone pos from Tag {tag_id}: {global_pos}")
                else:
                    cv2.putText(frame, f"Tag {tag_id} unknown", (ptA[0], ptA[1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Pose estimation failed", (ptA[0], ptA[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Compute the final drone position if any tags were detected.
        if global_positions:
            global_positions = np.array(global_positions)
            avg_position = np.mean(global_positions, axis=0)
            # Convert local (east, north) offsets back to latitude and longitude.
            drone_lat, drone_lon = local_to_latlon(avg_position[0], avg_position[1], base_lat, base_lon)
            drone_alt = avg_position[2]

            output_text = f"Drone: Alt {drone_alt:.2f} m, Lat {drone_lat:.7f}, Lon {drone_lon:.7f}"
            cv2.putText(frame, output_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print("Estimated Drone Global Position:", output_text)
            writer.writerow([image_path, drone_lat, drone_lon, drone_alt])
        else:
            print("No known tags detected.")
            writer.writerow([image_path, "NA", "NA", "NA"])

        # Display the result for the current image.
        cv2.imshow("Drone Position Estimation", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

cv2.destroyAllWindows()
