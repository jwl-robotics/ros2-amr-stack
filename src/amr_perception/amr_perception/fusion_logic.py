"""
LiDAR-camera fusion logic, independent of ROS.

Everything here operates on plain Python and numpy types so it can be unit
tested without an rclpy runtime.  ``sensor_fusion_node`` is a thin adapter
that converts ROS messages to and from these types and looks up TF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

Point3 = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float

    def back_project(self, u: float, v: float, depth: float) -> Point3:
        """Pixel (u, v) at ``depth`` metres to a point in the optical frame (z forward, x right, y down)."""
        z = depth
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return (x, y, z)


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform: rotation as a unit quaternion (x, y, z, w) then translation."""

    translation: Point3
    rotation: Tuple[float, float, float, float]

    @classmethod
    def from_rpy(cls, roll: float, pitch: float, yaw: float, translation: Point3 = (0.0, 0.0, 0.0)) -> 'RigidTransform':
        """Build from fixed-axis roll/pitch/yaw, the convention used by URDF ``<origin rpy>``."""
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        return cls(translation, (qx, qy, qz, qw))

    def apply(self, p: Point3) -> Point3:
        """Rotate then translate ``p``."""
        x, y, z = p
        qx, qy, qz, qw = self.rotation
        tx, ty, tz = self.translation

        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qz * qw)
        r02 = 2.0 * (qx * qz + qy * qw)

        r10 = 2.0 * (qx * qy + qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qx * qw)

        r20 = 2.0 * (qx * qz - qy * qw)
        r21 = 2.0 * (qy * qz + qx * qw)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

        return (
            r00 * x + r01 * y + r02 * z + tx,
            r10 * x + r11 * y + r12 * z + ty,
            r20 * x + r21 * y + r22 * z + tz,
        )


# ``lookup(target_frame, source_frame)`` -> transform taking points from source to target,
# or None if the transform is unavailable.
TransformLookup = Callable[[str, str], Optional[RigidTransform]]


# --------------------------------------------------------------------------- #
# Detections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PixelDetection:
    """A 2D camera detection reduced to its bounding-box centre."""

    u: int
    v: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class ProjectedDetection:
    """A camera detection back-projected into 3D in the LiDAR frame."""

    x: float
    y: float
    z: float
    class_name: str
    confidence: float


@dataclass(frozen=True)
class LidarDetection:
    """A LiDAR 3D detection reduced to its centre and label."""

    x: float
    y: float
    z: float
    class_name: str
    confidence: float


def decode_depth_image(data: bytes, height: int, width: int, encoding: str) -> np.ndarray:
    """Decode raw sensor_msgs/Image depth bytes to a (height, width) float64 array in metres.

    Supports 16UC1 / mono16 (millimetres) and 32FC1 (metres).  Raises
    ValueError for other encodings or a buffer that does not match the shape.
    """
    if encoding in ('16UC1', 'mono16'):
        depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
        return depth.astype(np.float64) / 1000.0
    if encoding == '32FC1':
        depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
        return depth.astype(np.float64)
    raise ValueError(f'Unsupported depth encoding: {encoding}')


def project_camera_detections(
    detections: Sequence[PixelDetection],
    depth: np.ndarray,
    camera_frame: str,
    lidar_frame: str,
    lookup: TransformLookup,
    intrinsics: CameraIntrinsics,
) -> Optional[List[ProjectedDetection]]:
    """Back-project camera detections through the depth image into the LiDAR frame.

    Returns None if the camera -> LiDAR transform is unavailable.  Detections
    whose centre falls outside the image or on an invalid depth pixel are dropped.
    """
    transform = lookup(lidar_frame, camera_frame)
    if transform is None:
        return None

    rows, cols = depth.shape
    projected: List[ProjectedDetection] = []
    for det in detections:
        if not (0 <= det.v < rows and 0 <= det.u < cols):
            continue
        d = float(depth[det.v, det.u])
        if d <= 0.0 or math.isnan(d) or math.isinf(d):
            continue
        x, y, z = transform.apply(intrinsics.back_project(det.u, det.v, d))
        projected.append(ProjectedDetection(x, y, z, det.class_name, det.confidence))
    return projected


# --------------------------------------------------------------------------- #
# Association and merging
# --------------------------------------------------------------------------- #


def associate(
    lidar: Sequence[LidarDetection],
    camera: Sequence[ProjectedDetection],
    distance_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy nearest-neighbour association on 3D centre distance.

    Returns (matched (lidar_idx, camera_idx) pairs, unmatched lidar indices,
    unmatched camera indices).  Pairs are taken in ascending distance order
    and never exceed ``distance_threshold``.
    """
    n_lidar, n_cam = len(lidar), len(camera)
    if n_lidar == 0 or n_cam == 0:
        return [], list(range(n_lidar)), list(range(n_cam))

    cost = np.empty((n_lidar, n_cam), dtype=np.float64)
    for i, ld in enumerate(lidar):
        for j, cd in enumerate(camera):
            cost[i, j] = math.sqrt((ld.x - cd.x) ** 2 + (ld.y - cd.y) ** 2 + (ld.z - cd.z) ** 2)

    matched: List[Tuple[int, int]] = []
    used_lidar: set = set()
    used_cam: set = set()
    for flat_idx in np.argsort(cost, axis=None):
        li = int(flat_idx // n_cam)
        ci = int(flat_idx % n_cam)
        if li in used_lidar or ci in used_cam:
            continue
        if cost[li, ci] > distance_threshold:
            break
        matched.append((li, ci))
        used_lidar.add(li)
        used_cam.add(ci)

    unmatched_lidar = [i for i in range(n_lidar) if i not in used_lidar]
    unmatched_cam = [j for j in range(n_cam) if j not in used_cam]
    return matched, unmatched_lidar, unmatched_cam


AGREEMENT_BONUS = 0.1


def merge_labels(
    lidar_class: str,
    lidar_conf: float,
    camera_class: str,
    camera_conf: float,
    w_lidar: float,
    w_camera: float,
) -> Tuple[str, float]:
    """Combine the labels of a matched LiDAR / camera pair.

    Confidence is the weighted sum of both sources.  If the classes agree the
    confidence gets a fixed bonus, capped at 1.0; if they disagree the class
    of the more confident source wins and no bonus is applied.
    """
    weighted = w_lidar * lidar_conf + w_camera * camera_conf
    if lidar_class == camera_class:
        return lidar_class, min(1.0, weighted + AGREEMENT_BONUS)
    winner = lidar_class if lidar_conf >= camera_conf else camera_class
    return winner, weighted
