"""
    This code post processes imagery captured from MAVS and runs YOLO.
    It looks for files named XXX.jpg in the output_data folder.
    It will create a yolo-detections folder (if it doesn't exist) and place detections there.
"""


import os
import glob
import subprocess


# base folder - edit to match your system
base_folder = r'c://mavs_windows10//MAVS-Examples//PythonExamples//SPAV'

# create a folder for the saved data if it doesn't exist
yolo_output_folder = '.' + os.path.sep + 'detections'
if not os.path.isdir(yolo_output_folder):
    os.makedirs(yolo_output_folder)
    print("")
    print("Creating directory ", yolo_output_folder)
    print("")

# Scan the output_data folder for files, and process with YOLO
mavs_output_folder = 'output_data'
file_search_folder = base_folder + os.path.sep + mavs_output_folder + os.path.sep
print("Scanning ", file_search_folder)
print("")
for name in glob.glob(file_search_folder + '*image.jpg'):
    # Print filename
    print("Processing file ", name)
    # Run YOLO
    p = subprocess.run(['python', 'detect_rename.py', '--images', name])
print("")
print("Done.")
