"""Bringup launch file for SLAM.

Includes the amr_slam mapping or localization launch depending on the
selected mode argument.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PythonExpression,
)


def generate_launch_description():
    pkg_amr_slam = get_package_share_directory("amr_slam")

    # ----- Launch arguments -----
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="mapping",
        description="SLAM mode: 'mapping' or 'localization'",
    )

    map_file_path_arg = DeclareLaunchArgument(
        "map_file_path",
        default_value="",
        description="Path to slam_toolbox pose-graph file (localization mode)",
    )

    is_mapping = PythonExpression(
        ["'", LaunchConfiguration("mode"), "' == 'mapping'"]
    )

    # ----- SLAM mapping -----
    slam_mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_amr_slam, "launch", "slam_mapping.launch.py")
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
        condition=IfCondition(is_mapping),
    )

    # ----- SLAM localization -----
    slam_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_amr_slam, "launch", "slam_localization.launch.py")
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "map_file_path": LaunchConfiguration("map_file_path"),
        }.items(),
        condition=UnlessCondition(is_mapping),
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            mode_arg,
            map_file_path_arg,
            slam_mapping_launch,
            slam_localization_launch,
        ]
    )
