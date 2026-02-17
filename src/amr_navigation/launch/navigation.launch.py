"""Launch file for the Nav2 navigation stack.

Includes the standard nav2_bringup launch and passes through the project's
Nav2 parameter file together with the user-supplied map.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_amr_navigation = get_package_share_directory("amr_navigation")
    pkg_nav2_bringup = get_package_share_directory("nav2_bringup")

    default_params_file = os.path.join(
        pkg_amr_navigation, "config", "nav2_params.yaml"
    )

    # ----- Launch arguments -----
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    map_arg = DeclareLaunchArgument(
        "map",
        default_value="",
        description="Full path to the map YAML file for nav2_map_server",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Full path to the Nav2 parameter YAML file",
    )

    autostart_arg = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Automatically start the Nav2 lifecycle nodes",
    )

    # ----- Nav2 bringup -----
    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2_bringup, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "map": LaunchConfiguration("map"),
            "params_file": LaunchConfiguration("params_file"),
            "autostart": LaunchConfiguration("autostart"),
        }.items(),
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            map_arg,
            params_file_arg,
            autostart_arg,
            nav2_bringup_launch,
        ]
    )
