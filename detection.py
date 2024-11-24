import apriltag
import argparse
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image containing AprilTag")
args = vars(ap.parse_args())

# load the input image and convert it to grayscale
print("[INFO] loading image...")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# define the AprilTags detector options and then detect the AprilTags
# in the input image
print("[INFO] detecting AprilTags...")
options = apriltag.DetectorOptions(families="tag16h5")
detector = apriltag.Detector(options)
results = detector.detect(gray)
print("[INFO] {} total AprilTags detected".format(len(results)))

# 3x3 numpy array for the camera matrix
cameraMatrix=np.array([[3,3,3], [3,3,3], [3,3,3]])
# 1x5 numpy array for distortion coeffecients
distCoeffs=np.array([1], [2], [3], [4], [5])
#defining object points
tag_size=1
ob_pt1 = [-tag_size/2, -tag_size/2, 0.0]
ob_pt2 = [ tag_size/2, -tag_size/2, 0.0]
ob_pt3 = [ tag_size/2,  tag_size/2, 0.0]
ob_pt4 = [-tag_size/2,  tag_size/2, 0.0]
ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4

# loop over the AprilTag detection results
for r in results:
	# extract the bounding box (x, y)-coordinates for the AprilTag
	# and convert each of the (x, y)-coordinate pairs to integers
	(ptA, ptB, ptC, ptD) = r.corners
	ptB = (int(ptB[0]), int(ptB[1]))
	ptC = (int(ptC[0]), int(ptC[1]))
	ptD = (int(ptD[0]), int(ptD[1]))
	ptA = (int(ptA[0]), int(ptA[1]))

	#get x,y of bottom right
	smallx = int(ptB[0])
	smally = int(ptB[1])
	for c in r.corners:
		if (c[0]<smallx):
			smallx=int(c[0])
		if (c[1]<smally):
			smally=int(c[1])
	tlc=(smallx,smally)

	
	

	# draw the bounding box of the AprilTag detection
	cv2.line(image, ptA, ptB, (0, 255, 0), 2)
	cv2.line(image, ptB, ptC, (0, 255, 0), 2)
	cv2.line(image, ptC, ptD, (0, 255, 0), 2)
	cv2.line(image, ptD, ptA, (0, 255, 0), 2)
	# draw the center (x, y)-coordinates of the AprilTag
	(cX, cY) = (int(r.center[0]), int(r.center[1]))
	cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
	# draw the tag family on the image
	tagFamily = r.tag_family.decode("utf-8")
	#output for x,y center position
	cent = "X:"+str(int(r.center[0]))+" , Y:"+str(int(r.center[1]))
	cv2.putText(image, cent, (tlc[0], tlc[1] - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	print("[INFO] tag family: {}".format(tagFamily))
	print(ptA)
		

_,rvecs,tvecs = cv2.solvePnP(ob_pts, results[0].corners,cameraMatrix, distCoeffs)

# show the output image after AprilTag detection
cv2.imshow("Image", image)
cv2.waitKey(0)