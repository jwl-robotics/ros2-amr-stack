# Copyright 2025 AMR Stack Authors
# Licensed under MIT
#
# Unit tests for the obstacle detector logic.
# These tests exercise the pure functions (pointcloud parsing, classification,
# clustering) WITHOUT any ROS dependency -- all ROS message structures are
# mocked with lightweight dataclass stand-ins.

import struct
from collections import namedtuple
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock ROS message types so we can import the detector helpers without rclpy
# ---------------------------------------------------------------------------

PointField = namedtuple("PointField", ["name", "offset", "datatype", "count"])


class MockPointCloud2:
    """Lightweight stand-in for sensor_msgs.msg.PointCloud2."""

    def __init__(
        self,
        fields: List[PointField],
        point_step: int,
        width: int,
        height: int,
        data: bytes,
    ):
        self.fields = fields
        self.point_step = point_step
        self.width = width
        self.height = height
        self.data = data


def _make_pointcloud2(points: np.ndarray) -> MockPointCloud2:
    """Build a mock PointCloud2 from an (N, 3) float32 numpy array.

    Uses the standard xyz layout with FLOAT32 fields at offsets 0, 4, 8 and
    a point_step of 12 bytes (tightly packed).
    """
    fields = [
        PointField(name="x", offset=0, datatype=7, count=1),
        PointField(name="y", offset=4, datatype=7, count=1),
        PointField(name="z", offset=8, datatype=7, count=1),
    ]
    pts = np.asarray(points, dtype=np.float32)
    data = pts.tobytes()
    return MockPointCloud2(
        fields=fields,
        point_step=12,
        width=len(pts),
        height=1,
        data=data,
    )


# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------

from amr_perception.nodes.obstacle_detector.obstacle_detector_node import (
    classify_cluster,
    pointcloud2_to_xyz_array,
)

# Default classification parameters that match the node defaults.
DEFAULT_PARAMS = {
    "person_width_min": 0.3,
    "person_width_max": 0.8,
    "person_height_min": 1.2,
    "person_height_max": 2.0,
    "shelf_min_height": 1.5,
    "shelf_max_width": 0.6,
    "pallet_max_height": 0.3,
    "pallet_min_width": 0.8,
}


# ===================================================================
# PointCloud2 parsing
# ===================================================================


class TestPointcloudParsing:
    """Tests for pointcloud2_to_xyz_array."""

    def test_pointcloud_parsing(self):
        """A simple 3-point cloud should parse into a matching (3, 3) array."""
        input_pts = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, -2.0, -3.0]],
            dtype=np.float32,
        )
        msg = _make_pointcloud2(input_pts)
        result = pointcloud2_to_xyz_array(msg)

        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, input_pts, atol=1e-6)

    def test_nan_filtering(self):
        """NaN points should be silently discarded."""
        pts = np.array(
            [[1.0, 2.0, 3.0], [float("nan"), 0.0, 0.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        )
        msg = _make_pointcloud2(pts)
        result = pointcloud2_to_xyz_array(msg)

        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(result[1], [4.0, 5.0, 6.0], atol=1e-6)

    def test_missing_fields(self):
        """A cloud missing the 'z' field should return an empty (0, 3) array."""
        fields = [
            PointField(name="x", offset=0, datatype=7, count=1),
            PointField(name="y", offset=4, datatype=7, count=1),
        ]
        msg = MockPointCloud2(
            fields=fields,
            point_step=8,
            width=1,
            height=1,
            data=struct.pack("<ff", 1.0, 2.0),
        )
        result = pointcloud2_to_xyz_array(msg)
        assert result.shape == (0, 3)

    def test_empty_cloud(self):
        """An empty PointCloud2 (width=0) should return (0, 3)."""
        msg = _make_pointcloud2(np.empty((0, 3), dtype=np.float32))
        result = pointcloud2_to_xyz_array(msg)
        assert result.shape == (0, 3)


# ===================================================================
# Geometric classification
# ===================================================================


class TestClassifyCluster:
    """Tests for classify_cluster."""

    def test_classify_person(self):
        """Narrow footprint, human-height cluster -> 'person'."""
        assert classify_cluster(0.5, 0.4, 1.7, DEFAULT_PARAMS) == "person"

    def test_classify_shelf(self):
        """Tall, narrow cluster -> 'shelf'."""
        assert classify_cluster(0.4, 0.3, 2.0, DEFAULT_PARAMS) == "shelf"

    def test_classify_pallet(self):
        """Low, wide cluster -> 'pallet'."""
        assert classify_cluster(1.0, 0.8, 0.15, DEFAULT_PARAMS) == "pallet"

    def test_classify_obstacle(self):
        """A cube that matches no specific rule -> 'obstacle'."""
        assert classify_cluster(0.5, 0.5, 0.5, DEFAULT_PARAMS) == "obstacle"

    def test_too_wide_for_person(self):
        """Human height but footprint too large -> not 'person'."""
        result = classify_cluster(1.5, 1.5, 1.7, DEFAULT_PARAMS)
        assert result != "person"

    def test_too_short_for_shelf(self):
        """Narrow but not tall enough for a shelf -> 'obstacle'."""
        result = classify_cluster(0.3, 0.3, 0.8, DEFAULT_PARAMS)
        assert result == "obstacle"


# ===================================================================
# Open3D DBSCAN clustering (uses Open3D directly, not the ROS node)
# ===================================================================


class TestClustering:
    """Tests for DBSCAN clustering via Open3D."""

    @pytest.fixture(autouse=True)
    def _import_open3d(self):
        """Skip the entire class if Open3D is not installed."""
        open3d = pytest.importorskip("open3d")
        self.o3d = open3d

    def test_clustering_separable_clusters(self):
        """Two well-separated blobs should produce exactly 2 clusters."""
        rng = np.random.default_rng(42)
        cluster_a = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.05, size=(50, 3))
        cluster_b = rng.normal(loc=[5.0, 5.0, 5.0], scale=0.05, size=(50, 3))
        points = np.vstack([cluster_a, cluster_b])

        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(points)
        labels = np.asarray(pcd.cluster_dbscan(eps=0.3, min_points=10, print_progress=False))

        unique = set(labels)
        unique.discard(-1)
        assert len(unique) == 2

    def test_clustering_single_cluster(self):
        """A single tight blob should produce exactly 1 cluster."""
        rng = np.random.default_rng(7)
        points = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.05, size=(80, 3))

        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(points)
        labels = np.asarray(pcd.cluster_dbscan(eps=0.3, min_points=10, print_progress=False))

        unique = set(labels)
        unique.discard(-1)
        assert len(unique) == 1

    def test_empty_cloud(self):
        """An empty point cloud should produce 0 clusters."""
        points = np.empty((0, 3), dtype=np.float64)

        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(points)
        labels = np.asarray(pcd.cluster_dbscan(eps=0.3, min_points=10, print_progress=False))

        unique = set(labels)
        unique.discard(-1)
        assert len(unique) == 0
