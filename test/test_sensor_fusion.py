# Copyright 2025 AMR Stack Authors
# Licensed under MIT
#
# Unit tests for the sensor fusion logic.
# All tests are pure Python -- no ROS runtime required.  We define a
# lightweight fusion helper that mirrors the association / merging rules
# that the real sensor_fusion_node would use, so the logic can be validated
# independently of the ROS 2 plumbing.

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal Detection data-class (mirrors the information in Detection3D)
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """Lightweight detection used by the fusion logic under test."""

    x: float
    y: float
    z: float
    width: float = 0.0
    depth: float = 0.0
    height: float = 0.0
    class_name: str = "obstacle"
    confidence: float = 1.0
    source: str = "lidar"  # "lidar" or "camera"


# ---------------------------------------------------------------------------
# Fusion helpers (extracted logic, testable without ROS)
# ---------------------------------------------------------------------------


def euclidean_distance(a: Detection, b: Detection) -> float:
    """3-D Euclidean distance between two detection centres."""
    return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2))


def associate_detections(
    lidar_dets: List[Detection],
    camera_dets: List[Detection],
    distance_threshold: float = 1.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy nearest-neighbour association between LiDAR and camera detections.

    Returns
    -------
    matched : list of (lidar_idx, camera_idx)
    unmatched_lidar : list of lidar indices
    unmatched_camera : list of camera indices
    """
    if not lidar_dets or not camera_dets:
        return (
            [],
            list(range(len(lidar_dets))),
            list(range(len(camera_dets))),
        )

    # Build distance matrix.
    n_lidar = len(lidar_dets)
    n_camera = len(camera_dets)
    dist_matrix = np.zeros((n_lidar, n_camera))
    for i, ld in enumerate(lidar_dets):
        for j, cd in enumerate(camera_dets):
            dist_matrix[i, j] = euclidean_distance(ld, cd)

    matched: List[Tuple[int, int]] = []
    used_lidar = set()
    used_camera = set()

    # Greedy assignment: pick the closest pair first.
    flat_order = np.argsort(dist_matrix, axis=None)
    for flat_idx in flat_order:
        li = int(flat_idx // n_camera)
        ci = int(flat_idx % n_camera)
        if li in used_lidar or ci in used_camera:
            continue
        if dist_matrix[li, ci] > distance_threshold:
            break
        matched.append((li, ci))
        used_lidar.add(li)
        used_camera.add(ci)

    unmatched_lidar = [i for i in range(n_lidar) if i not in used_lidar]
    unmatched_camera = [i for i in range(n_camera) if i not in used_camera]
    return matched, unmatched_lidar, unmatched_camera


def merge_detection(
    lidar_det: Detection,
    camera_det: Detection,
    lidar_weight: float = 0.6,
) -> Detection:
    """Merge a matched LiDAR + camera detection pair.

    Position is taken from the LiDAR detection (more accurate in 3-D).
    Confidence is a weighted average. When both sensors agree on the class,
    confidence gets a bonus. When they disagree, the class from the
    higher-confidence source wins.
    """
    camera_weight = 1.0 - lidar_weight
    merged_confidence = (
        lidar_det.confidence * lidar_weight
        + camera_det.confidence * camera_weight
    )

    if lidar_det.class_name == camera_det.class_name:
        # Agreement bonus -- cap at 1.0.
        merged_confidence = min(merged_confidence * 1.2, 1.0)
        merged_class = lidar_det.class_name
    else:
        # Disagreement: trust the source with higher confidence.
        if lidar_det.confidence >= camera_det.confidence:
            merged_class = lidar_det.class_name
        else:
            merged_class = camera_det.class_name

    return Detection(
        x=lidar_det.x,
        y=lidar_det.y,
        z=lidar_det.z,
        width=lidar_det.width,
        depth=lidar_det.depth,
        height=lidar_det.height,
        class_name=merged_class,
        confidence=merged_confidence,
        source="fused",
    )


def fuse(
    lidar_dets: List[Detection],
    camera_dets: List[Detection],
    distance_threshold: float = 1.0,
) -> List[Detection]:
    """Run the full fusion pipeline and return a merged detection list."""
    matched, unmatched_l, unmatched_c = associate_detections(
        lidar_dets, camera_dets, distance_threshold
    )

    output: List[Detection] = []
    for li, ci in matched:
        output.append(merge_detection(lidar_dets[li], camera_dets[ci]))
    for li in unmatched_l:
        output.append(lidar_dets[li])
    for ci in unmatched_c:
        output.append(camera_dets[ci])
    return output


# ===================================================================
# Tests
# ===================================================================


class TestAssociation:
    """Tests for detection association (matching)."""

    def test_association_close_detections(self):
        """Two detections within the threshold should be matched."""
        lidar = [Detection(x=1.0, y=2.0, z=0.0, class_name="person", confidence=0.9)]
        camera = [Detection(x=1.05, y=2.05, z=0.0, class_name="person", confidence=0.8)]

        matched, unmatched_l, unmatched_c = associate_detections(
            lidar, camera, distance_threshold=1.0
        )

        assert len(matched) == 1
        assert matched[0] == (0, 0)
        assert len(unmatched_l) == 0
        assert len(unmatched_c) == 0

    def test_association_far_detections(self):
        """Two detections far apart should remain unmatched."""
        lidar = [Detection(x=0.0, y=0.0, z=0.0)]
        camera = [Detection(x=10.0, y=10.0, z=0.0)]

        matched, unmatched_l, unmatched_c = associate_detections(
            lidar, camera, distance_threshold=1.0
        )

        assert len(matched) == 0
        assert len(unmatched_l) == 1
        assert len(unmatched_c) == 1


class TestConfidenceMerging:
    """Tests for confidence calculation during merging."""

    def test_confidence_merging(self):
        """Weighted confidence should combine both sources correctly."""
        lidar = Detection(x=1.0, y=2.0, z=0.0, class_name="obstacle",
                          confidence=0.9, source="lidar")
        camera = Detection(x=1.0, y=2.0, z=0.0, class_name="obstacle",
                           confidence=0.7, source="camera")

        merged = merge_detection(lidar, camera, lidar_weight=0.6)

        # Same class => base = 0.9*0.6 + 0.7*0.4 = 0.82, then *1.2 = 0.984
        expected_base = 0.9 * 0.6 + 0.7 * 0.4
        expected = min(expected_base * 1.2, 1.0)
        assert abs(merged.confidence - expected) < 1e-6

    def test_class_agreement(self):
        """Same class from both sensors should result in a confidence bonus."""
        lidar = Detection(x=0.0, y=0.0, z=0.0, class_name="person",
                          confidence=0.8, source="lidar")
        camera = Detection(x=0.0, y=0.0, z=0.0, class_name="person",
                           confidence=0.8, source="camera")

        merged = merge_detection(lidar, camera)
        base = 0.8 * 0.6 + 0.8 * 0.4  # 0.8
        boosted = min(base * 1.2, 1.0)  # 0.96

        assert merged.class_name == "person"
        assert abs(merged.confidence - boosted) < 1e-6
        # Boosted should be higher than the raw weighted average.
        assert merged.confidence > base

    def test_class_disagreement(self):
        """Different classes should use the higher-confidence source's class."""
        lidar = Detection(x=0.0, y=0.0, z=0.0, class_name="shelf",
                          confidence=0.5, source="lidar")
        camera = Detection(x=0.0, y=0.0, z=0.0, class_name="person",
                           confidence=0.9, source="camera")

        merged = merge_detection(lidar, camera)

        # Camera has higher confidence, so its class wins.
        assert merged.class_name == "person"
        # No bonus applied on disagreement.
        expected = 0.5 * 0.6 + 0.9 * 0.4  # 0.66
        assert abs(merged.confidence - expected) < 1e-6


class TestFusionPipeline:
    """Integration-level tests for the full fuse() function."""

    def test_empty_inputs(self):
        """No detections from either source should produce an empty output."""
        result = fuse([], [])
        assert result == []

    def test_empty_camera(self):
        """LiDAR detections with no camera input should pass through."""
        lidar = [Detection(x=1.0, y=2.0, z=0.0)]
        result = fuse(lidar, [])
        assert len(result) == 1
        assert result[0].x == 1.0

    def test_empty_lidar(self):
        """Camera detections with no LiDAR input should pass through."""
        camera = [Detection(x=3.0, y=4.0, z=0.0, source="camera")]
        result = fuse([], camera)
        assert len(result) == 1
        assert result[0].x == 3.0

    def test_one_matched_one_unmatched(self):
        """One matching pair plus one unmatched camera detection."""
        lidar = [Detection(x=1.0, y=1.0, z=0.0, class_name="person", confidence=0.9)]
        camera = [
            Detection(x=1.05, y=1.05, z=0.0, class_name="person",
                      confidence=0.8, source="camera"),
            Detection(x=10.0, y=10.0, z=0.0, class_name="shelf",
                      confidence=0.7, source="camera"),
        ]
        result = fuse(lidar, camera, distance_threshold=1.0)
        assert len(result) == 2  # 1 fused + 1 unmatched camera
