# Copyright 2025 AMR Stack Authors
# Licensed under MIT
#
# Unit tests for amr_perception.fusion_logic -- the association, merging and
# projection code the sensor_fusion node runs.  No ROS runtime required.

import math

import numpy as np
import pytest

from amr_perception.fusion_logic import (
    AGREEMENT_BONUS,
    CameraIntrinsics,
    LidarDetection,
    PixelDetection,
    ProjectedDetection,
    RigidTransform,
    associate,
    decode_depth_image,
    merge_labels,
    project_camera_detections,
)

W_LIDAR = 0.6
W_CAMERA = 0.4


def lidar(x, y, z, class_name="obstacle", confidence=1.0):
    return LidarDetection(x, y, z, class_name, confidence)


def camera(x, y, z, class_name="obstacle", confidence=1.0):
    return ProjectedDetection(x, y, z, class_name, confidence)


# ===================================================================
# Association
# ===================================================================


class TestAssociation:

    def test_close_detections_match(self):
        matched, unmatched_l, unmatched_c = associate(
            [lidar(1.0, 2.0, 0.0, "person", 0.9)],
            [camera(1.05, 2.05, 0.0, "person", 0.8)],
            distance_threshold=1.0,
        )
        assert matched == [(0, 0)]
        assert unmatched_l == []
        assert unmatched_c == []

    def test_far_detections_stay_unmatched(self):
        matched, unmatched_l, unmatched_c = associate(
            [lidar(0.0, 0.0, 0.0)], [camera(10.0, 10.0, 0.0)], distance_threshold=1.0
        )
        assert matched == []
        assert unmatched_l == [0]
        assert unmatched_c == [0]

    def test_empty_inputs_pass_through(self):
        assert associate([], [], 1.0) == ([], [], [])
        assert associate([lidar(1, 2, 0)], [], 1.0) == ([], [0], [])
        assert associate([], [camera(3, 4, 0)], 1.0) == ([], [], [0])

    def test_greedy_takes_closest_pair_first(self):
        # Camera det 0 is near both lidar dets, but closest to lidar det 1.
        li = [lidar(0.0, 0.0, 0.0), lidar(0.5, 0.0, 0.0)]
        c = [camera(0.6, 0.0, 0.0), camera(5.0, 0.0, 0.0)]
        matched, unmatched_l, unmatched_c = associate(li, c, distance_threshold=1.0)
        assert matched == [(1, 0)]
        assert unmatched_l == [0]
        assert unmatched_c == [1]

    def test_one_to_one(self):
        # Two lidar dets both near one camera det: only one may claim it.
        li = [lidar(0.0, 0.0, 0.0), lidar(0.1, 0.0, 0.0)]
        c = [camera(0.05, 0.0, 0.0)]
        matched, unmatched_l, _ = associate(li, c, distance_threshold=1.0)
        assert len(matched) == 1
        assert len(unmatched_l) == 1


# ===================================================================
# Label merging
# ===================================================================


class TestMergeLabels:

    def test_agreement_adds_fixed_bonus(self):
        cls, conf = merge_labels("obstacle", 0.9, "obstacle", 0.7, W_LIDAR, W_CAMERA)
        expected = 0.9 * W_LIDAR + 0.7 * W_CAMERA + AGREEMENT_BONUS
        assert cls == "obstacle"
        assert conf == pytest.approx(expected)

    def test_agreement_is_capped_at_one(self):
        _, conf = merge_labels("person", 1.0, "person", 1.0, W_LIDAR, W_CAMERA)
        assert conf == 1.0

    def test_disagreement_picks_more_confident_source(self):
        cls, conf = merge_labels("shelf", 0.5, "person", 0.9, W_LIDAR, W_CAMERA)
        assert cls == "person"
        assert conf == pytest.approx(0.5 * W_LIDAR + 0.9 * W_CAMERA)

    def test_disagreement_tie_goes_to_lidar(self):
        cls, _ = merge_labels("shelf", 0.7, "person", 0.7, W_LIDAR, W_CAMERA)
        assert cls == "shelf"


# ===================================================================
# Depth decoding
# ===================================================================


class TestDecodeDepth:

    def test_16uc1_millimetres_to_metres(self):
        raw = np.array([[1000, 2500]], dtype=np.uint16)
        depth = decode_depth_image(raw.tobytes(), 1, 2, "16UC1")
        assert depth.dtype == np.float64
        assert depth.tolist() == [[1.0, 2.5]]

    def test_32fc1_passthrough(self):
        raw = np.array([[1.25], [3.0]], dtype=np.float32)
        depth = decode_depth_image(raw.tobytes(), 2, 1, "32FC1")
        assert depth.tolist() == [[1.25], [3.0]]

    def test_unsupported_encoding_raises(self):
        with pytest.raises(ValueError):
            decode_depth_image(b"\x00" * 4, 1, 1, "rgb8")

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            decode_depth_image(b"\x00" * 6, 2, 2, "16UC1")


# ===================================================================
# Geometry
# ===================================================================


class TestGeometry:

    def test_back_project_principal_point_lies_on_optical_axis(self):
        k = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        assert k.back_project(320, 240, 3.0) == pytest.approx((0.0, 0.0, 3.0))

    def test_back_project_off_centre(self):
        k = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        # 100 px right of centre at 2 m -> 0.4 m right; 50 px below -> 0.2 m down.
        assert k.back_project(420, 290, 2.0) == pytest.approx((0.4, 0.2, 2.0))

    def test_identity_transform(self):
        t = RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        assert t.apply((1.0, 2.0, 3.0)) == pytest.approx((1.0, 2.0, 3.0))

    def test_translation_only(self):
        t = RigidTransform((1.0, -2.0, 0.5), (0.0, 0.0, 0.0, 1.0))
        assert t.apply((1.0, 1.0, 1.0)) == pytest.approx((2.0, -1.0, 1.5))

    def test_from_rpy_yaw_quarter_turn(self):
        t = RigidTransform.from_rpy(0.0, 0.0, math.pi / 2)
        assert t.apply((1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)

    def test_from_rpy_optical_convention(self):
        # URDF <origin rpy="-pi/2 0 -pi/2"/> between a body frame (x forward, z up)
        # and its optical frame (z forward, x right, y down).
        optical_to_body = RigidTransform.from_rpy(-math.pi / 2, 0.0, -math.pi / 2)
        assert optical_to_body.apply((0.0, 0.0, 1.0)) == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)   # forward
        assert optical_to_body.apply((1.0, 0.0, 0.0)) == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)  # right
        assert optical_to_body.apply((0.0, 1.0, 0.0)) == pytest.approx((0.0, 0.0, -1.0), abs=1e-12)  # down


# ===================================================================
# Projection
# ===================================================================


K = CameraIntrinsics(fx=554.25, fy=554.25, cx=320.0, cy=240.0)


def depth_image(value: float, height: int = 480, width: int = 640) -> np.ndarray:
    return np.full((height, width), value, dtype=np.float64)


class TestProjection:

    def test_transform_unavailable_returns_none(self):
        result = project_camera_detections(
            [PixelDetection(320, 240, "person", 0.9)], depth_image(3.0),
            "camera_optical_frame", "velodyne_link", lambda target, source: None, K,
        )
        assert result is None

    def test_pixel_outside_image_is_dropped(self):
        identity = RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        result = project_camera_detections(
            [PixelDetection(640, 240, "person", 0.9), PixelDetection(10, -1, "person", 0.9)],
            depth_image(3.0), "cam", "lidar", lambda target, source: identity, K,
        )
        assert result == []

    def test_invalid_depth_is_dropped(self):
        identity = RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        depth = depth_image(3.0)
        depth[240, 320] = 0.0
        depth[100, 100] = float("nan")
        depth[200, 200] = float("inf")
        result = project_camera_detections(
            [PixelDetection(320, 240, "a", 1.0), PixelDetection(100, 100, "b", 1.0),
             PixelDetection(200, 200, "c", 1.0)],
            depth, "cam", "lidar", lambda target, source: identity, K,
        )
        assert result == []

    def test_label_is_carried_through(self):
        identity = RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        [result] = project_camera_detections(
            [PixelDetection(320, 240, "pallet", 0.42)], depth_image(1.0),
            "cam", "lidar", lambda target, source: identity, K,
        )
        assert (result.class_name, result.confidence) == ("pallet", 0.42)

    def test_lookup_is_asked_for_the_given_frames(self):
        asked = []

        def lookup(target, source):
            asked.append((target, source))
            return RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

        project_camera_detections([], depth_image(1.0), "camera_optical_frame", "velodyne_link", lookup, K)
        assert asked == [("velodyne_link", "camera_optical_frame")]
