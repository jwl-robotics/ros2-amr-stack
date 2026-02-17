"""Launch file for AMR robot description: publishes URDF to robot_state_publisher."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_amr_description = get_package_share_directory("amr_description")

    # ----- Launch arguments -----
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    # ----- Process URDF via xacro -----
    xacro_file = os.path.join(pkg_amr_description, "urdf", "amr.urdf.xacro")
    robot_description_content = Command(["xacro ", xacro_file])

    # ----- Nodes -----
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description_content,
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
    )

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            robot_state_publisher_node,
            joint_state_publisher_node,
        ]
    )
