"""Full-stack bringup: simulation + perception + SLAM + navigation.

Launches every subsystem of the AMR stack in one go. Intended for
end-to-end testing in the Gazebo warehouse simulation.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_amr_bringup = get_package_share_directory("amr_bringup")

    # ----- Launch arguments -----
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    world_arg = DeclareLaunchArgument(
        "world",
        default_value="warehouse_simple",
        description="Name of the Gazebo world (without .world extension)",
    )

    slam_mode_arg = DeclareLaunchArgument(
        "slam_mode",
        default_value="mapping",
        description="SLAM mode: 'mapping' or 'localization'",
    )

    map_arg = DeclareLaunchArgument(
        "map",
        default_value="",
        description="Full path to the map YAML file (navigation)",
    )

    map_file_path_arg = DeclareLaunchArgument(
        "map_file_path",
        default_value="",
        description="Path to slam_toolbox pose-graph (localization mode)",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")

    # ----- 1. Simulation (Gazebo + robot description + spawn) -----
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_amr_bringup, "launch", "simulation.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "world": LaunchConfiguration("world"),
        }.items(),
    )

    # ----- 2. Perception pipeline (delay to let Gazebo start) -----
    perception_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        pkg_amr_bringup, "launch", "perception.launch.py"
                    )
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                }.items(),
            ),
        ],
    )

    # ----- 3. SLAM (delay to let perception start) -----
    slam_launch = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        pkg_amr_bringup, "launch", "slam.launch.py"
                    )
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "mode": LaunchConfiguration("slam_mode"),
                    "map_file_path": LaunchConfiguration("map_file_path"),
                }.items(),
            ),
        ],
    )

    # ----- 4. Navigation (delay to let SLAM publish map) -----
    navigation_launch = TimerAction(
        period=12.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        pkg_amr_bringup, "launch", "navigation.launch.py"
                    )
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "map": LaunchConfiguration("map"),
                }.items(),
            ),
        ],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            world_arg,
            slam_mode_arg,
            map_arg,
            map_file_path_arg,
            simulation_launch,
            perception_launch,
            slam_launch,
            navigation_launch,
        ]
    )
