// Copyright 2025 AMR Stack Authors
// Licensed under Apache-2.0
//
// Pointcloud filter node header for the AMR platform.
// Provides a configurable PCL processing pipeline: voxel downsampling,
// range gating, RANSAC ground-plane removal, and statistical outlier removal.

#ifndef AMR_POINTCLOUD_FILTER__POINTCLOUD_FILTER_NODE_HPP_
#define AMR_POINTCLOUD_FILTER__POINTCLOUD_FILTER_NODE_HPP_

#include <memory>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace amr_pointcloud_filter {

/// @brief ROS 2 node that applies a multi-stage PCL filter pipeline to
///        incoming 3-D LiDAR data and publishes the filtered result.
class PointcloudFilterNode : public rclcpp::Node
{
public:
  /// @brief Construct the node, declare parameters, and wire up pub/sub.
  explicit PointcloudFilterNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // ---------------------------------------------------------------
  // Callback
  // ---------------------------------------------------------------

  /// @brief Main processing callback.  Runs the full filter pipeline on each
  ///        incoming pointcloud and publishes the results.
  /// @param msg  Raw sensor_msgs/PointCloud2 from the LiDAR driver.
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  // ---------------------------------------------------------------
  // Filter stages
  // ---------------------------------------------------------------

  /// @brief Voxel-grid downsample to reduce point density uniformly.
  /// @param cloud  Input cloud (not modified).
  /// @return New cloud with reduced point count.
  pcl::PointCloud<pcl::PointXYZI>::Ptr applyVoxelFilter(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud) const;

  /// @brief Remove points that fall outside the [min_range, max_range] sphere.
  /// @param cloud  Input cloud (not modified).
  /// @return New cloud containing only points within the valid range.
  pcl::PointCloud<pcl::PointXYZI>::Ptr applyRangeFilter(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud) const;

  /// @brief Segment the ground plane with RANSAC and split the cloud.
  /// @param cloud  Input cloud (not modified).
  /// @return Pair of (non-ground / obstacle cloud, ground cloud).
  std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr>
  removeGround(const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud) const;

  /// @brief Remove sparse statistical outliers.
  /// @param cloud  Input cloud (not modified).
  /// @return New cloud with outliers removed.
  pcl::PointCloud<pcl::PointXYZI>::Ptr applyOutlierRemoval(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud) const;

  // ---------------------------------------------------------------
  // ROS 2 interfaces
  // ---------------------------------------------------------------

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;

  // ---------------------------------------------------------------
  // Tunable parameters (declared with defaults in the constructor)
  // ---------------------------------------------------------------

  // Voxel grid
  double voxel_leaf_size_;

  // Range gate
  double min_range_;
  double max_range_;

  // Ground-plane RANSAC
  double ground_distance_threshold_;
  int ground_max_iterations_;

  // Statistical outlier removal
  int outlier_mean_k_;
  double outlier_stddev_mul_;

  // Per-stage enable flags
  bool enable_voxel_filter_;
  bool enable_range_filter_;
  bool enable_ground_removal_;
  bool enable_outlier_removal_;
};

}  // namespace amr_pointcloud_filter

#endif  // AMR_POINTCLOUD_FILTER__POINTCLOUD_FILTER_NODE_HPP_
