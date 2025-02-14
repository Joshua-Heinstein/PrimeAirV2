#!/usr/bin/env python3
import re
import ast
import sys
import numpy as np

def parse_calibration_data(text):
    """
    Extract camera matrices and distortion coefficients from the provided text.
    Returns two lists: one with 3x3 numpy arrays (camera matrices) and one with
    1xN numpy arrays (distortion coefficients).
    """
    # Updated regex to capture the entire literal on one line.
    cam_matrix_pattern = r"Camera matrix for [^\n]+:\s*(\[[^\n]+\])"
    distort_pattern = r"Distortion coefficients for [^\n]+:\s*(\[[^\n]+\])"

    cam_matrix_matches = re.findall(cam_matrix_pattern, text)
    distort_matches = re.findall(distort_pattern, text)

    camera_matrices = []
    distortions = []

    for mstr in cam_matrix_matches:
        try:
            arr = np.array(ast.literal_eval(mstr))
            camera_matrices.append(arr)
        except Exception as e:
            print("Error parsing camera matrix block:", e)
    
    for dstr in distort_matches:
        try:
            arr = np.array(ast.literal_eval(dstr))
            distortions.append(arr)
        except Exception as e:
            print("Error parsing distortion coefficient block:", e)
    
    return camera_matrices, distortions

def compute_averages(matrices, distortions):
    avg_camera_matrix = np.mean(matrices, axis=0)
    avg_distortion = np.mean(distortions, axis=0)
    return avg_camera_matrix, avg_distortion

def main():
    if len(sys.argv) < 2:
        print("Usage: python average_calibration.py <calibration_log.txt>")
        sys.exit(1)

    log_file = sys.argv[1]
    try:
        with open(log_file, "r") as f:
            text = f.read()
    except Exception as e:
        print("Error reading file:", e)
        sys.exit(1)
    
    camera_matrices, distortions = parse_calibration_data(text)

    if not camera_matrices:
        print("No camera matrices found in the file.")
        sys.exit(1)
    if not distortions:
        print("No distortion coefficients found in the file.")
        sys.exit(1)

    avg_cam, avg_dist = compute_averages(camera_matrices, distortions)

    print("Average Camera Matrix:")
    print(avg_cam)
    print("\nAverage Distortion Coefficients:")
    print(avg_dist)

if __name__ == '__main__':
    main()
