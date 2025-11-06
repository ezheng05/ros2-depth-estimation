from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'depth_estimation_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.py')),  # This line adds launch file support
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='parallels',
    maintainer_email='your_email@example.com',
    description='ROS 2 depth estimation package with ZoeDepth',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = depth_estimation_pkg.camera_node:main',
            'depth_estimation_node = depth_estimation_pkg.depth_estimation_node:main',
            'near_node = depth_estimation_pkg.near_node:main',
            'rviz_marker_node = depth_estimation_pkg.rviz_marker_node:main',
            'image_viz_node = depth_estimation_pkg.image_viz_node:main',
        ],
    },
)
