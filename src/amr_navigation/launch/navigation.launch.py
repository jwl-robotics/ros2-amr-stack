"""Launch file for the Nav2 navigation stack.

When used alongside slam_toolbox (which provides map + localization),
this launches only the navigation components: planner, controller,
behavior server, costmaps, etc. -- NOT amcl or map_server.
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

    # ----- Nav2 navigation launch (no localization) -----
    # Use nav2_bringup's navigation_launch.py which only starts
    # planner, controller, behavior, smoother, velocity_smoother, bt_navigator,
    # waypoint_follower, and lifecycle_manager -- NOT amcl or map_server.
    nav2_navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2_bringup, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "params_file": LaunchConfiguration("params_file"),
            "autostart": LaunchConfiguration("autostart"),
        }.items(),
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            params_file_arg,
            autostart_arg,
            nav2_navigation_launch,
        ]
    )
