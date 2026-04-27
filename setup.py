from setuptools import find_packages, setup

package_name = 'depth_estimation_pkg'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ellen',
    maintainer_email='ellenz@bu.edu',
    description='Depth estimation using ZoeDepth',
    license='MIT',
    entry_points={
        'console_scripts': [
            'depth_node = depth_estimation_pkg.node:main',
            'force_viz = depth_estimation_pkg.force_viz:main',
            'overlay = depth_estimation_pkg.overlay:main',
            'validate = depth_estimation_pkg.validate:main',
            'cbf_node = depth_estimation_pkg.cbf_node:main',
        ],
    },
)