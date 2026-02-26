# ROS 2 Depth Estimation System

## Files

camera_node.py - Captures camera images, publishes to /camera/image

depth_estimation_node.py - Runs ZoeDepth AI model, publishes depth maps to /depth_estimation/depth  

direction_detection_node.py - Finds closest point, calculates 3D direction vector, publishes to /direction/*

visualization_node.py - Displays camera, depth map, and direction in one window

main_3d_system.py - ZoeDepth model wrapper

## How it connects

Camera captures images → Depth Estimation processes with AI → Direction Detection finds closest point → Visualization displays all three

Data flows through ROS topics:

/camera/image - Raw camera feed

/depth_estimation/depth - Depth map in meters

/direction/closest_point - 3D direction vector to closest object

/direction/visualization - Camera image with crosshair and arrow overlay

## How it functions

Camera node opens webcam and publishes 640x480 images at 30fps.

Depth estimation node subscribes to camera images, resizes to 64x64, runs ZoeDepth neural network, outputs depth in meters at 0.77fps.

Direction detection node subscribes to depth and camera topics, finds pixel with minimum depth, smooths over 5 frames, converts to 3D direction using camera focal length, publishes direction vector and visualization.

Visualization node subscribes to all topics, combines into three-panel display at 30fps.

## How to run

Build:
```bash
source /opt/ros/jazzy/setup.bash
cd ~/Documents/research/robotics-lab/dev
colcon build --packages-select depth_estimation_pkg
source install/setup.bash
```

Launch:
```bash
ros2 launch depth_estimation_pkg depth_launch.py
```

Controls: Press q to quit, s to save screenshot
