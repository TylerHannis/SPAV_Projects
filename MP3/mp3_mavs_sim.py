"""
    Sensor Processing for Autonomous Vehicles
    Mini Project 3 - LiDAR Small Object Detection Simulation

    This simulation animates a brick and estimates the number of lidar points on the brick

    Follow install instructions in mini-project 3 directions.

    This module requires PIL, matplotlib, and numpy.
    If install is needed: In Anaconda, type

        conda install anaconda::numpy
        conda install anaconda::pil
        conda install conda-forge::matplotlib

        note: if conda install anaconda::pil doesn't work, try
        pip install pillow

     Authors:
            Chris Goodin
            John Ball

    Edit history:
            04/09/2025  Modified for MAVS 2025
            03/20/2020  Initial release
"""

import matplotlib
import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

import mavspy.mavs as mavs

#
# Clear screen
#
os.system('cls' if os.name == 'nt' else 'clear')

#
# Print welcome message
#
print('Sensor Processing for Autonomous Vehicles\n')
print('MAVS LiDAR Small Object Detection Simulation version 1.0.')
print('Starting simulation...')

# Set the path to the mavs data folder
mavs_data_path = mavs.mavs_data_path
print("\n")
print("MAVS data path is", mavs_data_path)
print("")

# The following three parameters can be changed to adjust the experiment
# This is the height of the sensor above the ground
sensor_height = 1.0

"""
    lidar type could be 
    "M8"
    "HDL-32E"
    "HDL-64E"
    "VLP-16"
    "OS1"
    "OS2"
"""
lidar_type = "M8"

# The distance the vehicle will move between scans
# scan_spacing = vehicle_velocity/scan_rate
#scan_spacing = 1.0        # Use this for class demo
scan_spacing = 0.25     # Use this for MP3 work

# Set the threshold number of points for detection
num_pts_threshold = 4


# Select a scene and load it - The json file needs to be in /data/scenes directory for MAVS
mavs_scenefile = "/scenes/brick_scene.json"
scene = mavs.MavsEmbreeScene()
scene.Load(mavs_data_path + mavs_scenefile)

print('Using scene file:', mavs_scenefile)
print('Using sensor:', lidar_type)
print('\n')

# Create a MAVS environment and add the scene to it
env = mavs.MavsEnvironment()
env.SetScene(scene)

# Flags to save data
flag_save_camera_data = False					# Save camera data files
flag_save_lidar_data = False					# Save lidar data files

# Plotting flags
flag_plot_camera_lidar_MAVS = False				# Plot using inbuilt MAVS functionality
flag_plot_camera_lidar_script = True			# Plot inside of script

# Set environment properties
env.SetTime(13)                   # 0-23
env.SetFog(0.0)                   # 0.0-100.0
env.SetSnow(0.0)                  # 0-25
env.SetTurbidity(7.0)             # 2-10
env.SetAlbedo(0.1)                # 0-1
env.SetCloudCover(0.0)            # 0-1
env.SetRainRate(0.0)              # 0-25
env.SetWind([2.5, 1.0])           # Horizontal windspeed in m/s

# Create a MAVS camera and set its properties
cam = mavs.MavsCamera()
# cam.Model('MachineVision')
img_rows = 512
img_cols = 512
cam.Initialize(img_cols, img_rows, 0.0035, 0.0035, 0.0035)    # 512 X 512, pix size 35 X 35 mm, f = 35 mm
imdim = cam.GetDimensions()

# Set camera properties
cam.RenderShadows(True)

# Increasing anti-aliasing will slow down simulation but gives nicer images
cam.SetAntiAliasingFactor(3)

# This should be called for the camera to know about
# environmental factor like rain, etc.
cam.SetEnvironmentProperties(env.obj)

# If raining, render drops splattered on the lens
cam.SetDropsOnLens(True)

# Set the gamma (0-1.0) and gain (0-2.0) of the camera
cam.SetGammaAndGain(0.5, 2.0)

# Set the camera offsets.
# This is the offset of the sensor from vehicle from the CG.
cam.SetOffset([1.0, 0.0, 2.0], [1.0, 0.0, 0.0, 0.0])

# Create a MAVS lidar and set its properties
lidar = mavs.MavsLidar(lidar_type)

# Set the same offset as the camera
lidar.SetOffset([0.0, 0.0, sensor_height], [1.0, 0.0, 0.0, 0.0])

# Now start the simulation main loop
px = -45.0            # initial x position in meters
dt = 0.1              # time step
n = 0                 # loop counter

# Variables for final plot
plot_dist = []        # Variable to save plot distances
plot_num_pts = []     # Variable to save number of points

# Plot window
if flag_plot_camera_lidar_script:
    fig1, [axs1, axs2] = plt.subplots(1, 2, figsize=(10, 6), dpi=100)

while px <= 0.0:

    # Set new position
    position = [px, 0.0, 0.0]
    orientation = [1.0, 0.0, 0.0, 0.0]

    # Update the camera
    cam.SetPose(position, orientation)
    cam.Update(env, dt)
    if flag_plot_camera_lidar_MAVS:
        cam.Display()

    # Get the camera image
    img_buff = cam.GetBuffer()
    buff_array = np.asarray(img_buff)
    img_array = np.zeros([imdim[0], imdim[1], imdim[2]])
    ndex = 0
    for k in range(imdim[2]):
        for i in range(imdim[0]):
            for j in range(imdim[1]):
                img_array[i][j][k] = buff_array[ndex] / 256.0
                ndex = ndex + 1

    # Save camera image
    if flag_save_camera_data:
        camera_im_name = '../MiniProject3_LiDAR_Brick_Sim_MAVS/data/camera_' + str(n) + '_img'
        cam.SaveCameraImage(camera_im_name + '.bmp')
        img = Image.open(camera_im_name + '.bmp')
        img.save(camera_im_name + '.jpg', 'JPEG', quality=100)

    # Update the lidar
    lidar.SetPose(position, orientation)
    lidar.Update(env, dt)

    if flag_plot_camera_lidar_MAVS:
        lidar.Display()

    # Save lidar point cloud
    if flag_save_camera_data:
        lidar_im_name = '../MiniProject3_LiDAR_Brick_Sim_MAVS/data/lidar_' + str(n) + '_cloud.txt'
        lidar.SaveColorizedPointCloud(lidar_im_name)

    # Get the lidar points. In MAVS, the lidar x axis is straight ahead
    p = np.array(lidar.GetPoints())    			# Convert GetPoints results into a numpy array

    x = p[:, 0]                        			# Extract x coordinates
    y = p[:, 1]                        			# Extract y coordinates
    z = p[:, 2]                        			# Extract z coordinates

    # Determine how many points
    brick_min_height = 0.075
    brickpts = np.array(np.where(z > brick_min_height))    	    # Brick points: throw ground out
    nonbrickpts = np.array(np.where(z <= brick_min_height)) 	# Non-brick points
    num_brickpts = np.prod(brickpts.shape)    	                # Count the number of points

    # Get total number of elements in the scan
    num_x = len(x)

    # Estimate the ground distance (in front of vehicle) to the brick.
    brick_distance = 0.0 - px

    # Print analysis results
    print("Loop %3d.  Lidar points=%9d.  Brick points=%6d.  Brick distance=%6.2f m." %
          (n, num_x, num_brickpts, brick_distance))

    # Update plot
    if flag_plot_camera_lidar_script:
        axs1.clear()
        axs1.imshow(img_array)
        axs1.set_title('Camera')
        axs1.axis('off')
        scatter_size_brickpts = 1.0
        scatter_size_nonbrickpts = 1.0
        axs2.clear()
        axs2.scatter(x[nonbrickpts] - px, y[nonbrickpts], s=scatter_size_brickpts, c="b")
        axs2.scatter(x[brickpts] - px, y[brickpts], s=scatter_size_nonbrickpts, c="r")
        axs2.axis('equal')
        axs2.set_title('LiDAR Top-Down View')
        axs2.set_xlabel('x (meters)')
        axs2.set_ylabel('y (meters)')
        axs2.set_xlim(-50.0, 50.0)
        axs2.set_ylim(-50.0, 50.0)
        plt.draw()
        plt.pause(0.001)
        plt.show(block=False)

    # Save data for plotting
    plot_dist.append(brick_distance)
    plot_num_pts.append(num_brickpts)

    # Update the loop counter
    n = n + 1
    px = px + scan_spacing

print("\nSim complete.")


# Save data to file
textfilename = 'simulation_' + lidar_type + '_results.txt'
plot_data = np.column_stack([plot_dist, plot_num_pts])
np.savetxt(textfilename, plot_data, fmt='%6.2f', newline='\n',
           header='Lidar:' + lidar_type + '. First column is range (meters), second is # points.')
print('\n\nSaved data to file', textfilename, '\n')

# Plot number of points versus distance
print('\n\nClose the plots to exit...\n')

fig1, ax2 = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
ax2.stem(plot_dist, plot_num_pts, basefmt='none', label='Number lidar points')                         # Plot number of points vs. distance
ax2.hlines(y=num_pts_threshold, xmin=0, xmax=np.max(plot_dist), color='red', linestyle='--', label='Threshold')   # Plot threshold
ax2.set(xlabel='distance (meters)', ylabel='# points', title='Number of brick points vs. distance for ' + lidar_type)
ax2.grid()
plt.legend()
plt.draw()
plt.pause(0.001)
plt.show()
