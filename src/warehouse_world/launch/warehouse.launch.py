"""Launch file for starting Gazebo with the warehouse_simple world."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_warehouse_world = get_package_share_directory("warehouse_world")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    # ----- Launch arguments -----
    world_arg = DeclareLaunchArgument(
        "world",
        default_value=os.path.join(
            pkg_warehouse_world, "worlds", "warehouse_simple.world"
        ),
        description="Full path to the Gazebo world file to load",
    )

    verbose_arg = DeclareLaunchArgument(
        "verbose",
        default_value="false",
        description="Set to true for verbose Gazebo output",
    )

    paused_arg = DeclareLaunchArgument(
        "paused",
        default_value="false",
        description="Start Gazebo in a paused state",
    )

    # ----- Gazebo server -----
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={
            "world": LaunchConfiguration("world"),
            "verbose": LaunchConfiguration("verbose"),
            "pause": LaunchConfiguration("paused"),
        }.items(),
    )

    # ----- Gazebo client (GUI) -----
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        ),
        launch_arguments={
            "verbose": LaunchConfiguration("verbose"),
        }.items(),
    )

    return LaunchDescription(
        [
            world_arg,
            verbose_arg,
            paused_arg,
            gzserver,
            gzclient,
        ]
    )
