// Copyright 2025 AMR Stack Authors
// Licensed under Apache-2.0
//
// Implementation of the multi-stage PCL pointcloud filter node.
// Pipeline order: voxel downsample -> range gate -> ground removal -> outlier removal.

#include "amr_pointcloud_filter/pointcloud_filter_node.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

namespace amr_pointcloud_filter
{

// ============================================================================
// Constructor
// ============================================================================

PointcloudFilterNode::PointcloudFilterNode(const rclcpp::NodeOptions & options)
: Node("pointcloud_filter", options)
{
  // ------------------------------------------------------------------
  // Declare parameters with sensible defaults
  // ------------------------------------------------------------------
  voxel_leaf_size_            = this->declare_parameter<double>("voxel_leaf_size", 0.03);
  min_range_                  = this->declare_parameter<double>("min_range", 0.3);
  max_range_                  = this->declare_parameter<double>("max_range", 50.0);
  ground_distance_threshold_  = this->declare_parameter<double>("ground_distance_threshold", 0.1);
  ground_max_iterations_      = this->declare_parameter<int>("ground_max_iterations", 100);
  outlier_mean_k_             = this->declare_parameter<int>("outlier_mean_k", 50);
  outlier_stddev_mul_         = this->declare_parameter<double>("outlier_stddev_mul", 1.0);
  enable_voxel_filter_        = this->declare_parameter<bool>("enable_voxel_filter", true);
  enable_range_filter_        = this->declare_parameter<bool>("enable_range_filter", true);
  enable_ground_removal_      = this->declare_parameter<bool>("enable_ground_removal", true);
  enable_outlier_removal_     = this->declare_parameter<bool>("enable_outlier_removal", true);

  // ------------------------------------------------------------------
  // Subscriber – use SensorDataQoS (best-effort, volatile) which is the
  // standard profile for high-throughput sensor streams.
  // ------------------------------------------------------------------
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/velodyne_points",
    rclcpp::SensorDataQoS(),
    std::bind(&PointcloudFilterNode::cloudCallback, this, std::placeholders::_1));

  // ------------------------------------------------------------------
  // Publishers – default QoS (reliable) so downstream consumers such as
  // obstacle detectors can rely on every message arriving.
  // ------------------------------------------------------------------
  filtered_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/cloud_filtered", 10);
  ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/cloud_ground", 10);

  // ------------------------------------------------------------------
  // Startup log
  // ------------------------------------------------------------------
  RCLCPP_INFO(this->get_logger(), "PointcloudFilterNode initialised");
  RCLCPP_INFO(this->get_logger(),
    "  voxel_leaf_size=%.3f  range=[%.1f, %.1f]  ground_thresh=%.3f  "
    "ground_iters=%d  outlier_k=%d  outlier_std=%.2f",
    voxel_leaf_size_, min_range_, max_range_,
    ground_distance_threshold_, ground_max_iterations_,
    outlier_mean_k_, outlier_stddev_mul_);
  RCLCPP_INFO(this->get_logger(),
    "  stages enabled — voxel:%s  range:%s  ground:%s  outlier:%s",
    enable_voxel_filter_   ? "yes" : "no",
    enable_range_filter_   ? "yes" : "no",
    enable_ground_removal_ ? "yes" : "no",
    enable_outlier_removal_? "yes" : "no");
}

// ============================================================================
// Main callback
// ============================================================================

void PointcloudFilterNode::cloudCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  // Start a wall-clock timer for profiling.
  const auto t_start = std::chrono::steady_clock::now();

  // Convert the incoming ROS 2 message to a PCL cloud.
  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  pcl::fromROSMsg(*msg, *cloud);

  if (cloud->empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
      "Received an empty pointcloud — skipping processing");
    return;
  }

  const size_t input_size = cloud->size();

  // ------------------------------------------------------------------
  // 1. Voxel grid downsample
  // ------------------------------------------------------------------
  if (enable_voxel_filter_) {
    cloud = applyVoxelFilter(cloud);
  }

  // ------------------------------------------------------------------
  // 2. Range filter
  // ------------------------------------------------------------------
  if (enable_range_filter_) {
    cloud = applyRangeFilter(cloud);
  }

  // ------------------------------------------------------------------
  // 3. Ground-plane removal (RANSAC)
  // ------------------------------------------------------------------
  pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud;
  if (enable_ground_removal_) {
    auto [non_ground, ground] = removeGround(cloud);
    cloud        = non_ground;
    ground_cloud = ground;
  }

  // ------------------------------------------------------------------
  // 4. Statistical outlier removal
  // ------------------------------------------------------------------
  if (enable_outlier_removal_) {
    cloud = applyOutlierRemoval(cloud);
  }

  // ------------------------------------------------------------------
  // Publish filtered (non-ground / obstacle) cloud
  // ------------------------------------------------------------------
  sensor_msgs::msg::PointCloud2 filtered_msg;
  pcl::toROSMsg(*cloud, filtered_msg);
  filtered_msg.header.frame_id = msg->header.frame_id;
  filtered_msg.header.stamp    = msg->header.stamp;
  filtered_pub_->publish(filtered_msg);

  // ------------------------------------------------------------------
  // Publish ground cloud (only when ground removal is active)
  // ------------------------------------------------------------------
  const size_t ground_size = ground_cloud ? ground_cloud->size() : 0u;
  if (ground_cloud) {
    sensor_msgs::msg::PointCloud2 ground_msg;
    pcl::toROSMsg(*ground_cloud, ground_msg);
    ground_msg.header.frame_id = msg->header.frame_id;
    ground_msg.header.stamp    = msg->header.stamp;
    ground_pub_->publish(ground_msg);
  }

  // ------------------------------------------------------------------
  // Diagnostics
  // ------------------------------------------------------------------
  const auto t_end = std::chrono::steady_clock::now();
  const double elapsed_ms =
    std::chrono::duration<double, std::milli>(t_end - t_start).count();

  RCLCPP_DEBUG(this->get_logger(),
    "in=%zu  out=%zu  ground=%zu  time=%.1f ms",
    input_size, cloud->size(), ground_size, elapsed_ms);

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
    "Pipeline stats — input: %zu pts | output: %zu pts | ground: %zu pts | "
    "latency: %.1f ms",
    input_size, cloud->size(), ground_size, elapsed_ms);
}

// ============================================================================
// Voxel grid downsample
// ============================================================================

pcl::PointCloud<pcl::PointXYZI>::Ptr PointcloudFilterNode::applyVoxelFilter(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud) const
{
  auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

  pcl::VoxelGrid<pcl::PointXYZI> voxel;
  voxel.setInputCloud(cloud);
  voxel.setLeafSize(
    static_cast<float>(voxel_leaf_size_),
    static_cast<float>(voxel_leaf_size_),
    static_cast<float>(voxel_leaf_size_));
  voxel.filter(*filtered);

  return filtered;
}

// ============================================================================
// Range filter
// ============================================================================

pcl::PointCloud<pcl::PointXYZI>::Ptr PointcloudFilterNode::applyRangeFilter(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud) const
{
  auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  filtered->header   = cloud->header;
  filtered->is_dense = cloud->is_dense;
  filtered->points.reserve(cloud->size());

  const double min_sq = min_range_ * min_range_;
  const double max_sq = max_range_ * max_range_;

  for (const auto & pt : cloud->points) {
    const double range_sq =
      static_cast<double>(pt.x) * pt.x +
      static_cast<double>(pt.y) * pt.y +
      static_cast<double>(pt.z) * pt.z;

    if (range_sq >= min_sq && range_sq <= max_sq) {
      filtered->points.push_back(pt);
    }
  }

  filtered->width  = static_cast<uint32_t>(filtered->points.size());
  filtered->height = 1;  // unorganised cloud after filtering

  return filtered;
}

// ============================================================================
// Ground-plane removal (RANSAC)
// ============================================================================

std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr,
          pcl::PointCloud<pcl::PointXYZI>::Ptr>
PointcloudFilterNode::removeGround(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud) const
{
  // Set up the planar RANSAC segmenter.
  pcl::SACSegmentation<pcl::PointXYZI> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(ground_distance_threshold_);
  seg.setMaxIterations(ground_max_iterations_);
  seg.setInputCloud(cloud);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  seg.segment(*inliers, *coefficients);

  // If RANSAC found no plane, return the original cloud unchanged and an
  // empty ground cloud.  This can happen with very sparse or already-clean
  // data and is not an error.
  if (inliers->indices.empty()) {
    RCLCPP_DEBUG(this->get_logger(),
      "RANSAC found no ground plane — returning cloud unchanged");
    auto empty_ground = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    return {cloud, empty_ground};
  }

  // Extract ground (inliers) and non-ground (outliers).
  pcl::ExtractIndices<pcl::PointXYZI> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);

  auto non_ground = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  extract.setNegative(true);   // keep everything *except* the plane
  extract.filter(*non_ground);

  auto ground = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  extract.setNegative(false);  // keep only the plane
  extract.filter(*ground);

  return {non_ground, ground};
}

// ============================================================================
// Statistical outlier removal
// ============================================================================

pcl::PointCloud<pcl::PointXYZI>::Ptr PointcloudFilterNode::applyOutlierRemoval(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud) const
{
  // Guard: SOR requires at least mean_k neighbours to compute statistics.
  if (static_cast<int>(cloud->size()) <= outlier_mean_k_) {
    RCLCPP_DEBUG(this->get_logger(),
      "Cloud size (%zu) <= mean_k (%d) — skipping outlier removal",
      cloud->size(), outlier_mean_k_);
    return cloud;
  }

  auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

  pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(outlier_mean_k_);
  sor.setStddevMulThresh(outlier_stddev_mul_);
  sor.filter(*filtered);

  return filtered;
}

}  // namespace amr_pointcloud_filter

// ============================================================================
// Entry point
// ============================================================================

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<amr_pointcloud_filter::PointcloudFilterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
