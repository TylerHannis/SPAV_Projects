"""

    Sensor Processing for Autonomous Vehicles
    Camera Calibration Mini-Project 2

    Notes: Modified from OpenCV camera cal code by John Ball

    Refer to these websites for more information:

    [1] OpenCV Camera Calibration
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

    [2] OpenCV Camera Calibration and 3D Reconstruction
    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

"""

import os                        # Operating system
import numpy as np               # Numpy
import cv2                       # OpenCV
import glob2                     # Glob2
import pandas as pd

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Define the number of corners for findChessboardCorners and drawChessboardCorners
num_corner_rows = 6
num_corner_cols = 9
num_corners = (num_corner_rows, num_corner_cols)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(num_corner_rows,num_corner_cols,0)
objp = np.zeros((num_corner_rows * num_corner_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_corner_rows, 0:num_corner_cols].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []   # 3d point in real world space
imgpoints = []   # 2d points in image plane.

# Read filenames so we can process all the image files

# So we don't have to hard code paths and it looks like we know what
# we are doing
cwd = os.getcwd()
# currently, it looks in "./data2", update if needed
image_dir = os.path.join(cwd, "MP2", "data2")                                         # Choose images1 or images2
print("\nReading files from directory\n", image_dir)

# Search for all .png files
os.chdir(image_dir)
image_filenames = glob2.glob('*.png')

# Loop through all the image files
num_good_images = 0
for fname in image_filenames:

    # Load image and convert to grayscale
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Show the image
    cv2.imshow('img', gray)
    cv2.waitKey(100)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, num_corners, cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret:
        print("Processing file ", fname, " (good image)")
        num_good_images += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, num_corners, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)

    else:
        print("Processing file", fname, "(bad image)")

cv2.destroyAllWindows()

print("\nThe number of good images in this dataset is", num_good_images, ".")

# Now perform the calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print results
distortion_params = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]

if ret:
    print("Calibration was successful.")

    # Camera calibration matrix
    print("\n\nThe camera matrix is\n\n", mtx)

    # Distortion coeff.
    print("\n\nCamera distortion coefficents:\n")
    flattened = [val for sublist in dist for val in sublist]
    for jj in range(0, len(flattened)):
        print(distortion_params[jj], "=", flattened[jj], "\n")
    print("\n\n")

else:
    print("Calibration was not successful.")

# Add code here to use calibrated images to undistort one of the images and report overall error (hint: see [1] and [2])
img = cv2.imread(image_filenames[1])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Plot the image uncropped and cropped
cv2.imshow("Uncropped, undistorted image (dataset=" + image_dir + ")", dst)

x, y, w, h = roi
dst_crop = dst[y:y+h, x:x+w]
cv2.imshow("Cropped, undistorted image (dataset=" + image_dir + ")", dst_crop)

# Reprojection error
total_error = 0.0
total_error_squared = 0.0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error
    total_error_squared += error**2

print("Mean error:", total_error/len(objpoints), "\n\n")
print("Root mean square error:", (total_error_squared/len(objpoints))**0.5, "\n\n")

# Wait for a keypress
print("Put mouse on either output image and press any key to exit.\n\n")

#while True:
#    if cv2.waitKey(0) > -1:
#        break

cv2.destroyAllWindows()
