"""Launch file for SLAM mapping mode.

Starts pointcloud_to_laserscan to convert 3-D filtered clouds into a 2-D
laser scan, then launches slam_toolbox in asynchronous online mapping mode.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_amr_slam = get_package_share_directory("amr_slam")

    # ----- Launch arguments -----
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    # ----- Pointcloud to LaserScan -----
    pointcloud_to_laserscan_node = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        remappings=[
            ("cloud_in", "/cloud_filtered"),
            ("scan", "/scan_2d"),
        ],
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "target_frame": "base_footprint",
                "min_height": 0.1,
                "max_height": 1.5,
                "range_min": 0.3,
                "range_max": 20.0,
                "angle_min": -3.14159,
                "angle_max": 3.14159,
                "angle_increment": 0.00872665,  # ~0.5 deg
                "scan_time": 0.1,
                "inf_epsilon": 1.0,
                "concurrency_level": 1,
            }
        ],
    )

    # ----- SLAM Toolbox (async mapping) -----
    slam_params_file = os.path.join(pkg_amr_slam, "config", "slam_toolbox_params.yaml")

    slam_toolbox_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            slam_params_file,
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "mode": "mapping",
            },
        ],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            pointcloud_to_laserscan_node,
            slam_toolbox_node,
        ]
    )
