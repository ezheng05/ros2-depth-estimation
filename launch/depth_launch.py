# launch camera node + depth estimation node
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='depth_estimation_pkg',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {'camera_id': 0},
                {'fps': 30},
            ]
        ),
        Node(
            package='depth_estimation_pkg',
            executable='depth_node',
            name='depth_node',
            output='screen',
        ),
    ])
