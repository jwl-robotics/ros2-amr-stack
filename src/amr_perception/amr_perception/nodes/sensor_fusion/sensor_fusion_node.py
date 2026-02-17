"""
Sensor fusion node for the AMR perception pipeline.

Fuses LiDAR 3D detections with camera 2D detections using depth-based
back-projection and Hungarian-style greedy association. Publishes a unified
Detection3DArray with merged confidence scores.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters
import tf2_ros
from tf2_ros import TransformException

from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseWithCovariance,
    Quaternion,
    TransformStamped,
    Vector3,
)
from std_msgs.msg import Header


def _lazy_import_numpy():
    """Lazy import for numpy to speed up node discovery."""
    import numpy as np
    return np


class ProjectedDetection:
    """A camera 2D detection that has been back-projected into 3D space."""

    __slots__ = ('x', 'y', 'z', 'class_name', 'confidence')

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        class_name: str,
        confidence: float,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.class_name = class_name
        self.confidence = confidence


class SensorFusionNode(Node):
    """Fuses LiDAR 3D detections with camera 2D detections."""

    def __init__(self) -> None:
        super().__init__('sensor_fusion')

        # -- Parameters ----------------------------------------------------------
        self.declare_parameter('association_distance_threshold', 1.0)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('lidar_frame', 'velodyne_link')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('fusion_confidence_weight_lidar', 0.6)
        self.declare_parameter('fusion_confidence_weight_camera', 0.4)

        # Camera intrinsics (Intel RealSense D435 defaults at 640x480).
        self.declare_parameter('camera_fx', 554.25)
        self.declare_parameter('camera_fy', 554.25)
        self.declare_parameter('camera_cx', 320.0)
        self.declare_parameter('camera_cy', 240.0)

        self._assoc_thresh: float = (
            self.get_parameter('association_distance_threshold')
            .get_parameter_value().double_value
        )
        self._camera_frame: str = (
            self.get_parameter('camera_frame')
            .get_parameter_value().string_value
        )
        self._lidar_frame: str = (
            self.get_parameter('lidar_frame')
            .get_parameter_value().string_value
        )
        self._depth_topic: str = (
            self.get_parameter('depth_topic')
            .get_parameter_value().string_value
        )
        self._w_lidar: float = (
            self.get_parameter('fusion_confidence_weight_lidar')
            .get_parameter_value().double_value
        )
        self._w_camera: float = (
            self.get_parameter('fusion_confidence_weight_camera')
            .get_parameter_value().double_value
        )
        self._fx: float = (
            self.get_parameter('camera_fx')
            .get_parameter_value().double_value
        )
        self._fy: float = (
            self.get_parameter('camera_fy')
            .get_parameter_value().double_value
        )
        self._cx: float = (
            self.get_parameter('camera_cx')
            .get_parameter_value().double_value
        )
        self._cy: float = (
            self.get_parameter('camera_cy')
            .get_parameter_value().double_value
        )

        # -- TF2 ----------------------------------------------------------------
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # -- Publisher -----------------------------------------------------------
        self._fused_pub = self.create_publisher(
            Detection3DArray, '/perception/fused_objects', 10
        )

        # -- Synchronized subscribers -------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._sub_lidar = message_filters.Subscriber(
            self, Detection3DArray, '/perception/obstacles_3d', qos_profile=qos
        )
        self._sub_camera = message_filters.Subscriber(
            self, Detection2DArray, '/perception/detections_camera', qos_profile=qos
        )
        self._sub_depth = message_filters.Subscriber(
            self, Image, self._depth_topic, qos_profile=qos
        )

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_lidar, self._sub_camera, self._sub_depth],
            queue_size=10,
            slop=0.1,
        )
        self._sync.registerCallback(self._sync_callback)

        self.get_logger().info(
            f'SensorFusionNode initialised  '
            f'(assoc_thresh={self._assoc_thresh:.2f}, '
            f'w_lidar={self._w_lidar:.2f}, w_camera={self._w_camera:.2f})'
        )

    # --------------------------------------------------------------------- #
    # Callback
    # --------------------------------------------------------------------- #

    def _sync_callback(
        self,
        lidar_msg: Detection3DArray,
        camera_msg: Detection2DArray,
        depth_msg: Image,
    ) -> None:
        """Time-synchronised callback: fuse LiDAR + camera detections."""
        np = _lazy_import_numpy()

        # 1. Project camera 2D detections into 3D (lidar frame).
        projected = self._project_camera_detections(camera_msg, depth_msg, np)

        # 2. Extract LiDAR detection centres and metadata.
        lidar_centres: List[Tuple[float, float, float]] = []
        lidar_classes: List[str] = []
        lidar_confs: List[float] = []
        for det in lidar_msg.detections:
            pos = det.bbox.center.position
            lidar_centres.append((pos.x, pos.y, pos.z))
            if det.results:
                lidar_classes.append(det.results[0].hypothesis.class_id)
                lidar_confs.append(det.results[0].hypothesis.score)
            else:
                lidar_classes.append('unknown')
                lidar_confs.append(0.0)

        # 3. Build association cost matrix and run greedy matching.
        matched_pairs, unmatched_lidar, unmatched_camera = self._associate(
            lidar_centres, projected, np
        )

        # 4. Build fused output.
        fused_msg = Detection3DArray()
        fused_msg.header = Header()
        fused_msg.header.stamp = self.get_clock().now().to_msg()
        fused_msg.header.frame_id = self._lidar_frame

        # 4a. Matched pairs -- merge detections.
        for li, ci in matched_pairs:
            det3d = self._merge_detections(
                lidar_msg.detections[li],
                lidar_classes[li],
                lidar_confs[li],
                projected[ci],
            )
            fused_msg.detections.append(det3d)

        # 4b. Unmatched LiDAR detections -- pass through.
        for li in unmatched_lidar:
            det3d = self._copy_lidar_detection(
                lidar_msg.detections[li],
                lidar_classes[li],
                lidar_confs[li],
            )
            fused_msg.detections.append(det3d)

        # 4c. Unmatched camera detections -- create from projected 3D point.
        for ci in unmatched_camera:
            det3d = self._make_detection_from_projected(projected[ci])
            fused_msg.detections.append(det3d)

        self._fused_pub.publish(fused_msg)
        self.get_logger().debug(
            f'Published {len(fused_msg.detections)} fused detections '
            f'(matched={len(matched_pairs)}, '
            f'lidar_only={len(unmatched_lidar)}, '
            f'camera_only={len(unmatched_camera)})'
        )

    # --------------------------------------------------------------------- #
    # Projection helpers
    # --------------------------------------------------------------------- #

    def _project_camera_detections(
        self,
        camera_msg: Detection2DArray,
        depth_msg: Image,
        np,
    ) -> List[ProjectedDetection]:
        """Back-project each camera 2D detection to 3D using depth image."""
        projected: List[ProjectedDetection] = []

        # Decode depth image.  Supports 16UC1 (mm) and 32FC1 (metres).
        depth_array = self._decode_depth_image(depth_msg, np)
        if depth_array is None:
            return projected

        # Look up camera -> lidar transform (once per callback).
        try:
            tf_stamped: TransformStamped = self._tf_buffer.lookup_transform(
                self._lidar_frame,
                self._camera_frame,
                rclpy.time.Time(),  # latest available
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'TF lookup {self._camera_frame} -> {self._lidar_frame} '
                f'failed: {exc}'
            )
            return projected

        for det in camera_msg.detections:
            # Bbox centre in pixel coordinates.
            u = int(det.bbox.center.position.x)
            v = int(det.bbox.center.position.y)

            # Bounds check.
            if not (0 <= v < depth_array.shape[0] and 0 <= u < depth_array.shape[1]):
                continue

            depth_val: float = float(depth_array[v, u])
            if depth_val <= 0.0 or math.isnan(depth_val) or math.isinf(depth_val):
                continue

            # Back-project to 3D in camera frame (pinhole model).
            z_cam = depth_val
            x_cam = (u - self._cx) * z_cam / self._fx
            y_cam = (v - self._cy) * z_cam / self._fy

            # Transform to lidar frame.
            pt_lidar = self._transform_point(
                x_cam, y_cam, z_cam, tf_stamped
            )
            if pt_lidar is None:
                continue

            # Extract class / confidence.
            class_name = 'unknown'
            confidence = 0.0
            if det.results:
                class_name = det.results[0].hypothesis.class_id
                confidence = det.results[0].hypothesis.score

            projected.append(
                ProjectedDetection(
                    pt_lidar[0], pt_lidar[1], pt_lidar[2],
                    class_name, confidence,
                )
            )

        return projected

    def _decode_depth_image(self, depth_msg: Image, np) -> Optional['np.ndarray']:
        """Convert a sensor_msgs/Image depth message to a numpy float array (metres)."""
        height = depth_msg.height
        width = depth_msg.width
        encoding = depth_msg.encoding

        try:
            if encoding in ('16UC1', 'mono16'):
                depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
                    height, width
                )
                return depth.astype(np.float64) / 1000.0  # mm -> m
            elif encoding in ('32FC1',):
                depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(
                    height, width
                )
                return depth.astype(np.float64)
            else:
                self.get_logger().warn(
                    f'Unsupported depth encoding: {encoding}'
                )
                return None
        except ValueError as exc:
            self.get_logger().error(f'Depth decode error: {exc}')
            return None

    @staticmethod
    def _transform_point(
        x: float, y: float, z: float, tf_stamped: TransformStamped
    ) -> Optional[Tuple[float, float, float]]:
        """Apply a rigid transform (translation + quaternion rotation) to a point."""
        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation

        # Quaternion rotation: p' = q * p * q_conj
        # Expand for performance instead of pulling in a library.
        qw, qx, qy, qz = q.w, q.x, q.y, q.z

        # Rotation matrix from quaternion.
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qz * qw)
        r02 = 2.0 * (qx * qz + qy * qw)

        r10 = 2.0 * (qx * qy + qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qx * qw)

        r20 = 2.0 * (qx * qz - qy * qw)
        r21 = 2.0 * (qy * qz + qx * qw)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

        px = r00 * x + r01 * y + r02 * z + t.x
        py = r10 * x + r11 * y + r12 * z + t.y
        pz = r20 * x + r21 * y + r22 * z + t.z

        return (px, py, pz)

    # --------------------------------------------------------------------- #
    # Association
    # --------------------------------------------------------------------- #

    def _associate(
        self,
        lidar_centres: List[Tuple[float, float, float]],
        projected: List[ProjectedDetection],
        np,
    ) -> Tuple[
        List[Tuple[int, int]], List[int], List[int]
    ]:
        """
        Greedy nearest-neighbour association (Hungarian-style).

        Returns (matched_pairs, unmatched_lidar_indices, unmatched_camera_indices).
        """
        n_lidar = len(lidar_centres)
        n_cam = len(projected)

        if n_lidar == 0 or n_cam == 0:
            return (
                [],
                list(range(n_lidar)),
                list(range(n_cam)),
            )

        # Build cost matrix.
        cost = np.zeros((n_lidar, n_cam), dtype=np.float64)
        for i, (lx, ly, lz) in enumerate(lidar_centres):
            for j, pd in enumerate(projected):
                dx = lx - pd.x
                dy = ly - pd.y
                dz = lz - pd.z
                cost[i, j] = math.sqrt(dx * dx + dy * dy + dz * dz)

        matched: List[Tuple[int, int]] = []
        used_lidar: set = set()
        used_cam: set = set()

        # Flatten and sort by ascending cost.
        flat_indices = np.argsort(cost, axis=None)
        for flat_idx in flat_indices:
            li = int(flat_idx // n_cam)
            ci = int(flat_idx % n_cam)
            if li in used_lidar or ci in used_cam:
                continue
            if cost[li, ci] > self._assoc_thresh:
                break  # All remaining are above threshold.
            matched.append((li, ci))
            used_lidar.add(li)
            used_cam.add(ci)

        unmatched_lidar = [i for i in range(n_lidar) if i not in used_lidar]
        unmatched_cam = [j for j in range(n_cam) if j not in used_cam]

        return matched, unmatched_lidar, unmatched_cam

    # --------------------------------------------------------------------- #
    # Detection builders
    # --------------------------------------------------------------------- #

    def _merge_detections(
        self,
        lidar_det: Detection3D,
        lidar_class: str,
        lidar_conf: float,
        cam_proj: ProjectedDetection,
    ) -> Detection3D:
        """Merge a matched LiDAR detection with a projected camera detection."""
        det = Detection3D()

        # Use LiDAR bounding box (more accurate geometry).
        det.bbox = lidar_det.bbox

        # Class arbitration.
        if lidar_class == cam_proj.class_name:
            # Both agree -- boost confidence.
            merged_class = lidar_class
            merged_conf = min(
                1.0,
                self._w_lidar * lidar_conf + self._w_camera * cam_proj.confidence + 0.1,
            )
        else:
            # Disagree -- use higher-confidence source.
            if lidar_conf >= cam_proj.confidence:
                merged_class = lidar_class
            else:
                merged_class = cam_proj.class_name
            merged_conf = (
                self._w_lidar * lidar_conf
                + self._w_camera * cam_proj.confidence
            )

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = merged_class
        hyp.hypothesis.score = float(merged_conf)
        det.results.append(hyp)

        return det

    @staticmethod
    def _copy_lidar_detection(
        lidar_det: Detection3D,
        lidar_class: str,
        lidar_conf: float,
    ) -> Detection3D:
        """Pass through an unmatched LiDAR detection."""
        det = Detection3D()
        det.bbox = lidar_det.bbox

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = lidar_class
        hyp.hypothesis.score = float(lidar_conf)
        det.results.append(hyp)

        return det

    def _make_detection_from_projected(
        self, cam_proj: ProjectedDetection
    ) -> Detection3D:
        """Create a Detection3D from a camera-only projected 3D point."""
        det = Detection3D()

        det.bbox.center.position.x = cam_proj.x
        det.bbox.center.position.y = cam_proj.y
        det.bbox.center.position.z = cam_proj.z

        # Default small bbox size since camera cannot estimate geometry well.
        det.bbox.size.x = 0.5
        det.bbox.size.y = 0.5
        det.bbox.size.z = 0.5

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = cam_proj.class_name
        hyp.hypothesis.score = float(cam_proj.confidence)
        det.results.append(hyp)

        return det


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
