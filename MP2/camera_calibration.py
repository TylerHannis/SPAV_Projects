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
# currently, it looks in "data2", update if needed
image_dir = os.path.join(cwd, "MP2", "data2")                                         # Choose images1 or images2
print("\nReading files from directory\n", image_dir)

# Search for all .png files
os.chdir(image_dir)
image_filenames = glob2.glob('*.png')


# collect the image results
img_results = {}

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
    # add the filenames and status (good, bad) to the dataframe for analysis
    if ret:
        print("Processing file ", fname, " (good image)")
        img_results.update({fname: "good"})
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
        img_results.update({fname: "bad"})

cv2.destroyAllWindows()
img_df = pd.DataFrame([img_results])

print("\nThe number of good images in this dataset is", num_good_images, ".")

# Now perform the calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
rnd_mtx = np.round(mtx, decimals=3)
# Print results
distortion_params = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
camera_coeff = {}
if ret:
    print("Calibration was successful.")

    # Camera calibration matrix
    print("\n\nThe camera matrix is\n\n", rnd_mtx)

    # Distortion coeff.
    print("\n\nCamera distortion coefficents:\n")
    flattened = [val for sublist in dist for val in sublist]
    for jj in range(0, len(flattened)):
        print(distortion_params[jj], "=", flattened[jj], "\n")
        # Add the camera coeffs to the dataframe for later
        camera_coeff.update({distortion_params[jj]: format(flattened[jj], '.3f')})
    print("\n\n")

else:
    print("Calibration was not successful.")
camera_df = pd.DataFrame([camera_coeff])
# Use calibrated images to undistort one of the images and report overall error
img_index = 10  # or any index from 0 to len(image_filenames) - 1
source_filename = image_filenames[img_index]
img = cv2.imread(source_filename)
h, w = img.shape[:2]

# Compute new camera matrix for optimal undistortion
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort the image
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Save uncropped undistorted image
uncropped_filename = f"undistorted_uncropped_{os.path.basename(source_filename)}"

# Show the uncropped image with window name including the filename
cv2.imshow(f"Uncropped: {source_filename}", dst)



# Crop and show the image using the ROI
x, y, w, h = roi
dst_crop = dst[y:y+h, x:x+w]

# Save cropped undistorted image
cropped_filename = f"undistorted_cropped_{os.path.basename(source_filename)}"

# Show cropped image with filename in the window title
cv2.imshow(f"Cropped: {source_filename}", dst_crop)


success_uncropped = cv2.imwrite(uncropped_filename, dst)
success_cropped = cv2.imwrite(cropped_filename, dst_crop)

print(f"Current working directory: {os.getcwd()}")
print(f"Uncropped saved? {success_uncropped} → {uncropped_filename}")
print(f"Cropped saved?   {success_cropped} → {cropped_filename}")

# Reprojection error
total_error = 0.0
total_error_squared = 0.0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error
    total_error_squared += error**2

# Collect the error calculations for later
mean_error = total_error/len(objpoints)
rmse = (total_error/len(objpoints))**0.5
error_df = pd.DataFrame([{"mean error": format(mean_error, '.3f'), 
                 "rmse": format(rmse, '.3f'),
                 }])


print(f"Mean error:, {mean_error:.3f}, \n\n")
print("Root mean square error:", rmse, "\n\n")

# Wait for a keypress
print("Put mouse on either output image and press any key to exit.\n\n")

# Uncomment to be able to grab images for qualitative analysis
#while True:
#    if cv2.waitKey(0) > -1:
#        break

cv2.destroyAllWindows()

# Print the dataframes to a latex table for easy addition to the report
print("Image results table: ")
print(img_df.to_latex())

print("\nCamera coeff results table: ")
print(camera_df.to_latex())

print("\Error table: ")
print(error_df.to_latex())
