from setuptools import setup, find_packages

package_name = 'amr_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jinwei Lim',
    maintainer_email='jinwei@example.com',
    description='Perception pipeline for warehouse AMR',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obstacle_detector = amr_perception.nodes.obstacle_detector.obstacle_detector_node:main',
            'camera_detector = amr_perception.nodes.camera_detector.camera_detector_node:main',
            'sensor_fusion = amr_perception.nodes.sensor_fusion.sensor_fusion_node:main',
            'object_tracker = amr_perception.nodes.object_tracker.object_tracker_node:main',
            'perception_viz = amr_perception.nodes.perception_viz.perception_viz_node:main',
        ],
    },
)
