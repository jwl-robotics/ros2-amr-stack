"""
Object tracker node for the AMR perception pipeline.

Maintains persistent track identities across frames and publishes confirmed
tracks as Detection3DArray plus a MarkerArray for RViz.  The association and
update rules live in ``amr_perception.tracking_logic``; this node only adapts
ROS messages to that module.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from amr_perception.tracking_logic import Observation, Track, Tracker


# --------------------------------------------------------------------------- #
# Colour palette for visualisation markers
# --------------------------------------------------------------------------- #

_PALETTE: List[Tuple[float, float, float]] = [
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
]


def _colour_for_id(track_id: int) -> ColorRGBA:
    """Deterministic colour from track ID."""
    r, g, b = _PALETTE[track_id % len(_PALETTE)]
    return ColorRGBA(r=r, g=g, b=b, a=1.0)


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #


class ObjectTrackerNode(Node):
    """Tracks detected objects across frames with ID persistence."""

    def __init__(self) -> None:
        super().__init__('object_tracker')

        # -- Parameters --------------------------------------------------
        self.declare_parameter('association_threshold', 1.5)
        self.declare_parameter('max_lost_frames', 10)
        self.declare_parameter('min_hits_to_confirm', 3)
        self.declare_parameter('velocity_smoothing_alpha', 0.3)

        self._tracker = Tracker(
            association_threshold=(
                self.get_parameter('association_threshold')
                .get_parameter_value().double_value
            ),
            max_lost=(
                self.get_parameter('max_lost_frames')
                .get_parameter_value().integer_value
            ),
            min_hits=(
                self.get_parameter('min_hits_to_confirm')
                .get_parameter_value().integer_value
            ),
            alpha=(
                self.get_parameter('velocity_smoothing_alpha')
                .get_parameter_value().double_value
            ),
        )

        # -- Publishers --------------------------------------------------
        self._tracked_pub = self.create_publisher(
            Detection3DArray, '/perception/tracked_objects', 10
        )
        self._viz_pub = self.create_publisher(
            MarkerArray, '/perception/tracks_viz', 10
        )

        # -- Subscriber --------------------------------------------------
        self.create_subscription(
            Detection3DArray,
            '/perception/fused_objects',
            self._detections_callback,
            10,
        )

        self.get_logger().info(
            f'ObjectTrackerNode initialised '
            f'(assoc_thresh={self._tracker.association_threshold:.2f}, '
            f'max_lost={self._tracker.max_lost}, '
            f'min_hits={self._tracker.min_hits}, '
            f'alpha={self._tracker.alpha:.2f})'
        )

    # --------------------------------------------------------------------- #
    # Callback
    # --------------------------------------------------------------------- #

    def _detections_callback(self, msg: Detection3DArray) -> None:
        # Node clock so that dt (and hence velocity) follows use_sim_time.
        now = self.get_clock().now().nanoseconds * 1e-9

        observations: List[Observation] = []
        for det in msg.detections:
            pos = det.bbox.center.position
            if det.results:
                hyp = det.results[0].hypothesis
                observations.append(Observation(pos.x, pos.y, pos.z, hyp.class_id, hyp.score))
            else:
                observations.append(Observation(pos.x, pos.y, pos.z))

        removed = self._tracker.update(observations, now)

        # Publish confirmed tracks as Detection3DArray.
        frame_id = msg.header.frame_id if msg.header.frame_id else 'velodyne_link'
        stamp = self.get_clock().now().to_msg()

        out_msg = Detection3DArray()
        out_msg.header = Header(stamp=stamp, frame_id=frame_id)

        confirmed_tracks: List[Track] = self._tracker.confirmed()

        for track in confirmed_tracks:
            det = Detection3D()
            det.bbox.center.position.x = track.x
            det.bbox.center.position.y = track.y
            det.bbox.center.position.z = track.z
            # Preserve a reasonable bbox size.
            det.bbox.size.x = 0.6
            det.bbox.size.y = 0.6
            det.bbox.size.z = 1.0

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = f'{track.class_name}:{track.id}'
            hyp.hypothesis.score = float(track.confidence)
            det.results.append(hyp)
            out_msg.detections.append(det)

        self._tracked_pub.publish(out_msg)

        # Publish visualisation markers.
        self._publish_markers(confirmed_tracks, stamp, frame_id)

        self.get_logger().debug(
            f'Tracking: {len(self._tracker.tracks)} total, '
            f'{len(confirmed_tracks)} confirmed, '
            f'{len(removed)} removed'
        )

    # --------------------------------------------------------------------- #
    # Visualisation
    # --------------------------------------------------------------------- #

    def _publish_markers(
        self,
        tracks: List[Track],
        stamp,
        frame_id: str,
    ) -> None:
        """Build and publish a MarkerArray with text labels and velocity arrows."""
        marker_array = MarkerArray()

        # First, add a DELETE_ALL marker to clean up stale markers.
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for track in tracks:
            colour = _colour_for_id(track.id)

            # -- Text label: "class_name #id" above the track ----------------
            text_marker = Marker()
            text_marker.header.stamp = stamp
            text_marker.header.frame_id = frame_id
            text_marker.ns = 'track_labels'
            text_marker.id = track.id * 2
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = Point(
                x=track.x, y=track.y, z=track.z + 1.2
            )
            text_marker.scale.z = 0.4  # text height
            text_marker.color = colour
            text_marker.text = f'{track.class_name} #{track.id}'
            text_marker.lifetime = DurationMsg(sec=0, nanosec=500_000_000)
            marker_array.markers.append(text_marker)

            # -- Velocity arrow -----------------------------------------------
            if track.speed > 0.05:  # Only draw arrow if moving appreciably.
                arrow = Marker()
                arrow.header.stamp = stamp
                arrow.header.frame_id = frame_id
                arrow.ns = 'track_velocity'
                arrow.id = track.id * 2 + 1
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD

                start = Point(x=track.x, y=track.y, z=track.z)
                end = Point(
                    x=track.x + track.vx,
                    y=track.y + track.vy,
                    z=track.z + track.vz,
                )
                arrow.points = [start, end]

                arrow.scale = Vector3(x=0.08, y=0.15, z=0.15)
                arrow.color = colour
                arrow.lifetime = DurationMsg(sec=0, nanosec=500_000_000)
                marker_array.markers.append(arrow)

        self._viz_pub.publish(marker_array)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = ObjectTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
