"""Bringup launch file for Nav2 navigation.

Thin wrapper that includes amr_navigation's navigation.launch.py and
forwards all relevant arguments.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_amr_navigation = get_package_share_directory("amr_navigation")

    default_params_file = os.path.join(
        pkg_amr_navigation, "config", "nav2_params.yaml"
    )

    # ----- Launch arguments -----
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Full path to the Nav2 parameter YAML file",
    )

    # ----- Navigation launch -----
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_amr_navigation, "launch", "navigation.launch.py"
            )
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "params_file": LaunchConfiguration("params_file"),
        }.items(),
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            params_file_arg,
            navigation_launch,
        ]
    )
