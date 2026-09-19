"""
Sensor fusion node for the AMR perception pipeline.

Fuses LiDAR 3D detections with camera 2D detections using depth-based
back-projection and greedy nearest-neighbour association, and publishes a
unified Detection3DArray with merged confidence scores.  The geometry and
association rules live in ``amr_perception.fusion_logic``; this node only
adapts ROS messages and TF to that module.

The two detection streams are time-synchronised with an
ApproximateTimeSynchronizer.  The depth image is *not* part of that
synchroniser: on Humble, message_filters.Subscriber on sensor_msgs/Image
raises "Unable to convert call argument to Python object" the first time a
three-way sync fires, which kills the node.  The depth image is instead
received on a plain subscription and the most recent frame is used, guarded
by a maximum age relative to the camera detections.
"""

from __future__ import annotations

from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

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
from std_msgs.msg import Header

from amr_perception.fusion_logic import (
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


class SensorFusionNode(Node):
    """Fuses LiDAR 3D detections with camera 2D detections."""

    def __init__(self) -> None:
        super().__init__('sensor_fusion')

        # -- Parameters ----------------------------------------------------------
        self.declare_parameter('association_distance_threshold', 1.0)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('lidar_frame', 'velodyne_link')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_max_age', 0.5)
        self.declare_parameter('fusion_confidence_weight_lidar', 0.6)
        self.declare_parameter('fusion_confidence_weight_camera', 0.4)

        # Camera intrinsics (Intel RealSense D435 defaults at 640x480).
        self.declare_parameter('camera_fx', 554.25)
        self.declare_parameter('camera_fy', 554.25)
        self.declare_parameter('camera_cx', 320.0)
        self.declare_parameter('camera_cy', 240.0)

        self._assoc_thresh: float = self._double('association_distance_threshold')
        self._camera_frame: str = self._string('camera_frame')
        self._lidar_frame: str = self._string('lidar_frame')
        self._depth_topic: str = self._string('depth_topic')
        self._depth_max_age: float = self._double('depth_max_age')
        self._w_lidar: float = self._double('fusion_confidence_weight_lidar')
        self._w_camera: float = self._double('fusion_confidence_weight_camera')
        self._intrinsics = CameraIntrinsics(
            fx=self._double('camera_fx'),
            fy=self._double('camera_fy'),
            cx=self._double('camera_cx'),
            cy=self._double('camera_cy'),
        )

        # -- TF2 ----------------------------------------------------------------
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # -- Publisher -----------------------------------------------------------
        self._fused_pub = self.create_publisher(
            Detection3DArray, '/perception/fused_objects', 10
        )

        # -- Subscribers --------------------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Depth image: plain subscription, latest frame cached (see module docstring).
        self._latest_depth: Optional[Image] = None
        self.create_subscription(Image, self._depth_topic, self._depth_callback, qos)

        # Detection streams: time-synchronised.
        self._sub_lidar = message_filters.Subscriber(
            self, Detection3DArray, '/perception/obstacles_3d', qos_profile=qos
        )
        self._sub_camera = message_filters.Subscriber(
            self, Detection2DArray, '/perception/detections_camera', qos_profile=qos
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_lidar, self._sub_camera],
            queue_size=10,
            slop=0.1,
        )
        self._sync.registerCallback(self._sync_callback)

        self.get_logger().info(
            f'SensorFusionNode initialised  '
            f'(assoc_thresh={self._assoc_thresh:.2f}, '
            f'w_lidar={self._w_lidar:.2f}, w_camera={self._w_camera:.2f})'
        )

    def _double(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _string(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    # --------------------------------------------------------------------- #
    # Callbacks
    # --------------------------------------------------------------------- #

    def _depth_callback(self, msg: Image) -> None:
        self._latest_depth = msg

    def _sync_callback(
        self,
        lidar_msg: Detection3DArray,
        camera_msg: Detection2DArray,
    ) -> None:
        """Time-synchronised callback: fuse LiDAR + camera detections."""
        # 1. Project camera 2D detections into 3D (lidar frame) using the
        #    most recent depth image, provided it is close enough in time.
        projected: List[ProjectedDetection] = []
        depth_msg = self._depth_for(camera_msg)
        if depth_msg is not None:
            projected = self._project(camera_msg, depth_msg)

        # 2. Reduce LiDAR detections to centres and labels.
        lidar: List[LidarDetection] = []
        for det in lidar_msg.detections:
            pos = det.bbox.center.position
            class_name, conf = _label_of(det)
            lidar.append(LidarDetection(pos.x, pos.y, pos.z, class_name, conf))

        # 3. Associate.
        matched_pairs, unmatched_lidar, unmatched_camera = associate(
            lidar, projected, self._assoc_thresh
        )

        # 4. Build fused output.
        fused_msg = Detection3DArray()
        fused_msg.header = Header()
        fused_msg.header.stamp = self.get_clock().now().to_msg()
        fused_msg.header.frame_id = self._lidar_frame

        for li, ci in matched_pairs:
            merged_class, merged_conf = merge_labels(
                lidar[li].class_name, lidar[li].confidence,
                projected[ci].class_name, projected[ci].confidence,
                self._w_lidar, self._w_camera,
            )
            fused_msg.detections.append(
                _detection_with_bbox(lidar_msg.detections[li].bbox, merged_class, merged_conf)
            )
        for li in unmatched_lidar:
            fused_msg.detections.append(
                _detection_with_bbox(lidar_msg.detections[li].bbox, lidar[li].class_name, lidar[li].confidence)
            )
        for ci in unmatched_camera:
            fused_msg.detections.append(_detection_from_projected(projected[ci]))

        self._fused_pub.publish(fused_msg)
        self.get_logger().debug(
            f'Published {len(fused_msg.detections)} fused detections '
            f'(matched={len(matched_pairs)}, '
            f'lidar_only={len(unmatched_lidar)}, '
            f'camera_only={len(unmatched_camera)})'
        )

    # --------------------------------------------------------------------- #
    # Projection adapters
    # --------------------------------------------------------------------- #

    def _depth_for(self, camera_msg: Detection2DArray) -> Optional[Image]:
        """Return the cached depth image if it is recent enough for these detections."""
        depth_msg = self._latest_depth
        if depth_msg is None:
            self.get_logger().warn(
                'No depth image received yet; camera detections not projected',
                throttle_duration_sec=5.0,
            )
            return None

        age = Time.from_msg(camera_msg.header.stamp) - Time.from_msg(depth_msg.header.stamp)
        age_sec = age.nanoseconds * 1e-9
        if abs(age_sec) > self._depth_max_age:
            self.get_logger().warn(
                f'Depth image is {age_sec:.2f}s from camera detections '
                f'(limit {self._depth_max_age:.2f}s); camera detections not projected',
                throttle_duration_sec=5.0,
            )
            return None
        return depth_msg

    def _project(self, camera_msg: Detection2DArray, depth_msg: Image) -> List[ProjectedDetection]:
        """Back-project the camera detections into the LiDAR frame."""
        try:
            depth = decode_depth_image(
                depth_msg.data, depth_msg.height, depth_msg.width, depth_msg.encoding
            )
        except ValueError as exc:
            self.get_logger().warn(f'Depth decode error: {exc}', throttle_duration_sec=5.0)
            return []

        pixels = [
            PixelDetection(
                int(det.bbox.center.position.x),
                int(det.bbox.center.position.y),
                *_label_of(det),
            )
            for det in camera_msg.detections
        ]

        projected = project_camera_detections(
            pixels, depth, self._camera_frame, self._lidar_frame,
            self._lookup_transform, self._intrinsics,
        )
        return projected if projected is not None else []

    def _lookup_transform(self, target_frame: str, source_frame: str) -> Optional[RigidTransform]:
        """Latest TF transform taking points from ``source_frame`` into ``target_frame``."""
        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),  # latest available
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'TF lookup {source_frame} -> {target_frame} failed: {exc}',
                throttle_duration_sec=5.0,
            )
            return None
        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation
        return RigidTransform((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


# --------------------------------------------------------------------------- #
# Message helpers
# --------------------------------------------------------------------------- #


def _label_of(det) -> tuple:
    """(class_name, confidence) of a Detection2D / Detection3D, defaulting to ('unknown', 0.0)."""
    if det.results:
        hyp = det.results[0].hypothesis
        return hyp.class_id, hyp.score
    return 'unknown', 0.0


def _detection_with_bbox(bbox, class_name: str, confidence: float) -> Detection3D:
    """Detection3D carrying an existing (LiDAR) bounding box and the given label."""
    det = Detection3D()
    det.bbox = bbox
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = class_name
    hyp.hypothesis.score = float(confidence)
    det.results.append(hyp)
    return det


def _detection_from_projected(cam_proj: ProjectedDetection) -> Detection3D:
    """Detection3D for a camera-only detection: projected centre, nominal box size."""
    det = Detection3D()
    det.bbox.center.position.x = cam_proj.x
    det.bbox.center.position.y = cam_proj.y
    det.bbox.center.position.z = cam_proj.z
    # The camera alone cannot estimate extent; use a nominal box.
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
