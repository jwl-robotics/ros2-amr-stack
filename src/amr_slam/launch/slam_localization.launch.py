"""Launch file for SLAM localization mode.

Same pipeline as mapping (pointcloud_to_laserscan + slam_toolbox) but the
slam_toolbox runs in localization mode using a previously saved pose-graph
map file.
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

    map_file_path_arg = DeclareLaunchArgument(
        "map_file_path",
        default_value="",
        description="Full path to the serialized slam_toolbox pose-graph "
        "(without extension). Required for localization mode.",
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
                "min_height": -0.1,
                "max_height": 0.5,
                "range_min": 0.3,
                "range_max": 20.0,
                "angle_min": -3.14159,
                "angle_max": 3.14159,
                "angle_increment": 0.00872665,
                "scan_time": 0.1,
                "inf_epsilon": 1.0,
                "concurrency_level": 1,
            }
        ],
    )

    # ----- SLAM Toolbox (localization) -----
    slam_params_file = os.path.join(pkg_amr_slam, "config", "slam_toolbox_params.yaml")

    slam_toolbox_node = Node(
        package="slam_toolbox",
        executable="localization_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            slam_params_file,
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "mode": "localization",
                "map_file_name": LaunchConfiguration("map_file_path"),
            },
        ],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            map_file_path_arg,
            pointcloud_to_laserscan_node,
            slam_toolbox_node,
        ]
    )
