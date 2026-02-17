"""Top-level launch file for the Gazebo simulation environment.

Starts:
  1. Gazebo with the warehouse world
  2. robot_state_publisher (URDF via amr_description)
  3. spawn_entity to place the robot in the Gazebo scene
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_warehouse_world = get_package_share_directory("warehouse_world")
    pkg_amr_description = get_package_share_directory("amr_description")

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

    # ----- Gazebo (warehouse world) -----
    warehouse_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_warehouse_world, "launch", "warehouse.launch.py")
        ),
        launch_arguments={
            "world": [
                os.path.join(pkg_warehouse_world, "worlds", ""),
                LaunchConfiguration("world"),
                ".world",
            ],
        }.items(),
    )

    # ----- Robot state publisher (URDF) -----
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_amr_description, "launch", "robot.launch.py")
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    # ----- Spawn robot in Gazebo -----
    spawn_entity_node = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_amr",
        output="screen",
        arguments=[
            "-entity", "amr_robot",
            "-topic", "robot_description",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.1",
            "-Y", "0.0",
        ],
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            world_arg,
            warehouse_launch,
            robot_description_launch,
            spawn_entity_node,
        ]
    )
