"""
ROS 2 Launch file for RViz-enabled depth estimation pipeline
Starts camera, depth, near, markers, and image viz nodes
"""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Generate launch description with all nodes for RViz"""
    
    return LaunchDescription([
        # Camera Node
        Node(
            package='depth_estimation_pkg',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {'camera_id': 0},
                {'fps': 30}
            ]
        ),
        
        # Depth Estimation Node
        Node(
            package='depth_estimation_pkg',
            executable='depth_estimation_node',
            name='depth_estimation_node',
            output='screen',
            parameters=[
                {'target_size_width': 64},
                {'target_size_height': 64},
                {'use_fast_processor': True},
                {'skip_frames': 3}
            ]
        ),
        
        # Near Node (finds closest point, publishes raw + smoothed)
        Node(
            package='depth_estimation_pkg',
            executable='near_node',
            name='near_node',
            output='screen',
            parameters=[
                {'focal_length_x': 600.0},
                {'focal_length_y': 600.0},
                {'principal_point_x': 320.0},
                {'principal_point_y': 240.0},
                {'smoothing_enabled': True},
                {'smoothing_window': 5}
            ]
        ),
        
        # RViz Marker Node (publishes 3D markers)
        Node(
            package='depth_estimation_pkg',
            executable='rviz_marker_node',
            name='rviz_marker_node',
            output='screen'
        ),
        
        # Image Visualization Node (OpenCV window)
        Node(
            package='depth_estimation_pkg',
            executable='image_viz_node',
            name='image_viz_node',
            output='screen',
            parameters=[
                {'focal_length_x': 600.0},
                {'focal_length_y': 600.0},
                {'principal_point_x': 320.0},
                {'principal_point_y': 240.0}
            ]
        ),
    ])
