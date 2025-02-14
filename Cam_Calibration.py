import numpy as np
import cv2 as cv
import glob

# Create (or overwrite) a log file where calibration output will be saved.
log_filename = "calibration_log.txt"
log_file = open(log_filename, "w")

# Termination criteria for corner subpixel accuracy
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for a 7x7 internal corner chessboard
objp = np.zeros((7 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

# Get all .png images in the current directory
images = glob.glob('*.png')

# Loop through each image
for fname in images:
    # Log the filename being processed.
    log_file.write(f"Processing file: {fname}\n")
    print(f"Processing file: {fname}")

    img = cv.imread(fname)
    if img is None:
        log_file.write(f"Failed to load image: {fname}\n")
        print(f"Failed to load image: {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (7, 7), None)
    if ret:
        log_file.write(f"Chessboard detected in {fname}\n")
        print(f"Chessboard detected in {fname}")

        # Refine corner locations for better accuracy.
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Prepare the object and image points for calibration (using only this image)
        single_objpoints = [objp]
        single_imgpoints = [corners2]
        img_shape = gray.shape[::-1]  # (width, height)

        # Calibrate the camera for this image
        ret_val, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
            single_objpoints, single_imgpoints, img_shape, None, None
        )

        # Write the calibration results in a format that can be parsed later.
        log_file.write(f"\nCamera matrix for {fname}:\n")
        # Writing as a list of lists for easier parsing.
        log_file.write(f"{camera_matrix.tolist()}\n")
        log_file.write(f"\nDistortion coefficients for {fname}:\n")
        log_file.write(f"{dist_coeffs.tolist()}\n")
        log_file.write("-" * 60 + "\n")

        # Optionally display the image with the detected chessboard corners.
        cv.drawChessboardCorners(img, (7, 7), corners2, ret)
        cv.imshow('img', img)
        log_file.write("Press any key to close this window and proceed to the next image.\n")
        cv.waitKey(0)
    else:
        log_file.write(f"Chessboard NOT detected in {fname}\n")
        print(f"Chessboard NOT detected in {fname}")

# Clean up: close the log file and all OpenCV windows.
log_file.close()
cv.destroyAllWindows()
