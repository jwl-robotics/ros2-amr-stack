#!/bin/bash
# Launch keyboard teleop for the AMR
# Usage: ./teleop.sh
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel
