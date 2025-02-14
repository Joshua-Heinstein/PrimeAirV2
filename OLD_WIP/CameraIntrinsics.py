#!/usr/bin/env python
import numpy as np
import cv2 as cv
import glob
import argparse

def calibrate_camera_chessboard(image_pattern, chessboard_size=(8, 8), display=False, save_images=False):
    """
    Calibrate the camera using a series of chessboard images.
    
    Parameters:
        image_pattern (str): Glob pattern for calibration images (e.g., "*_Color.png")
        chessboard_size (tuple): Number of inner corners per chessboard row and column (e.g., (7,6))
        display (bool): If True, display each image with drawn chessboard corners.
        save_images (bool): If True, save output images with detected corners overlay.
        
    Returns:
        camera_matrix (ndarray): The intrinsic camera matrix.
        dist_coeffs (ndarray): The distortion coefficients.
    """
    # Termination criteria for refining corner locations.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, e.g., (0,0,0), (1,0,0), …,(7-1,6-1,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all images.
    objpoints = []  # 3D point in real world space.
    imgpoints = []  # 2D points in image plane.

    # Load calibration images based on the provided pattern.
    images = glob.glob(image_pattern)
    if not images:
        print(f"No images found matching pattern: {image_pattern}")
        return None, None

    print(f"Found {len(images)} images.")

    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"Cannot load image: {fname}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Detect chessboard corners.
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            # Append the object points (they are the same for each image).
            objpoints.append(objp)

            # Refine corner detection to sub-pixel accuracy.
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw the detected corners.
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)

            if display:
                cv.imshow('Detected Chessboard', img)
                cv.waitKey(500)  # Display each image for 500ms

            if save_images:
                outname = "output_" + fname
                cv.imwrite(outname, img)
                print(f"Saved {outname}")
        else:
            print(f"Chessboard corners not found in {fname}")

    if display:
        cv.destroyAllWindows()

    # Use the size of the last processed image. (All calibration images should have the same size.)
    image_size = gray.shape[::-1]  # (width, height)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    return camera_matrix, dist_coeffs

def main():
    parser = argparse.ArgumentParser(
        description="Camera Calibration using Chessboard images (checkers board)."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_Color.png",
        help="File pattern for calibration images (default: '*_Color.png')."
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display each calibration image with detected corners."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output images with drawn corners."
    )
    args = parser.parse_args()

    camera_matrix, dist_coeffs = calibrate_camera_chessboard(
        args.pattern, chessboard_size=(7, 6), display=args.display, save_images=args.save
    )
    if camera_matrix is not None:
        print("Camera Intrinsic Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
    else:
        print("Calibration failed. Please check your images and chessboard dimensions.")

if __name__ == "__main__":
    main()
