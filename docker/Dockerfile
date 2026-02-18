###############################################################################
# Stage 1: Build
###############################################################################
FROM osrf/ros:humble-desktop AS build

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    libpcl-dev \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions \
    ros-humble-pointcloud-to-laserscan \
    ros-humble-slam-toolbox \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-nav2-smac-planner \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-msgs \
    ros-humble-message-filters \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Copy source into the workspace
WORKDIR /ros2_ws
COPY src /ros2_ws/src

# Build the workspace (Python deps not needed for colcon build)
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

###############################################################################
# Stage 2: Runtime
###############################################################################
FROM osrf/ros:humble-desktop AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime ROS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions \
    ros-humble-pointcloud-to-laserscan \
    ros-humble-slam-toolbox \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-nav2-smac-planner \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-msgs \
    ros-humble-message-filters \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    open3d \
    ultralytics \
    opencv-python-headless

# Copy the built workspace from the build stage
COPY --from=build /ros2_ws/install /ros2_ws/install
COPY --from=build /ros2_ws/src /ros2_ws/src

# NVIDIA runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /ros2_ws

# Auto-source ROS2 and workspace in interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo '[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash' >> /root/.bashrc

# Entrypoint: source ROS2 and workspace, then execute the user command
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
