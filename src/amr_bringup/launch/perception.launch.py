"""Launch all perception pipeline nodes.

Starts:
  1. pointcloud_filter  (C++ node from amr_pointcloud_filter)
  2. obstacle_detector  (Python node from amr_perception)
  3. camera_detector    (Python node from amr_perception)
  4. sensor_fusion      (Python node from amr_perception)
  5. object_tracker     (Python node from amr_perception)
  6. perception_viz     (Python node from amr_perception)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ----- Launch arguments -----
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")

    # ----- Pointcloud filter (C++ node) -----
    pointcloud_filter_node = Node(
        package="amr_pointcloud_filter",
        executable="pointcloud_filter_node",
        name="pointcloud_filter",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # ----- Obstacle detector -----
    obstacle_detector_node = Node(
        package="amr_perception",
        executable="obstacle_detector",
        name="obstacle_detector",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # ----- Camera detector -----
    camera_detector_node = Node(
        package="amr_perception",
        executable="camera_detector",
        name="camera_detector",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # ----- Sensor fusion -----
    sensor_fusion_node = Node(
        package="amr_perception",
        executable="sensor_fusion",
        name="sensor_fusion",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # ----- Object tracker -----
    object_tracker_node = Node(
        package="amr_perception",
        executable="object_tracker",
        name="object_tracker",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # ----- Perception visualizer -----
    perception_viz_node = Node(
        package="amr_perception",
        executable="perception_viz",
        name="perception_viz",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            pointcloud_filter_node,
            obstacle_detector_node,
            camera_detector_node,
            sensor_fusion_node,
            object_tracker_node,
            perception_viz_node,
        ]
    )
