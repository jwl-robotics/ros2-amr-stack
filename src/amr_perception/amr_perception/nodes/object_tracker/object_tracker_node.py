"""
Object tracker node for the AMR perception pipeline.

Maintains persistent track identities across frames using greedy
nearest-neighbour association with exponential-moving-average velocity
estimation.  Publishes confirmed tracks as Detection3DArray and a
MarkerArray for RViz visualisation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def _lazy_import_numpy():
    """Lazy import for numpy to speed up node discovery."""
    import numpy as np
    return np


# --------------------------------------------------------------------------- #
# Track data structure
# --------------------------------------------------------------------------- #


@dataclass
class Track:
    """Internal representation of a tracked object."""

    id: int
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    class_name: str = 'unknown'
    confidence: float = 0.0
    hits: int = 1
    lost_count: int = 0
    last_seen_time: float = 0.0


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

        self._assoc_thresh: float = (
            self.get_parameter('association_threshold')
            .get_parameter_value().double_value
        )
        self._max_lost: int = (
            self.get_parameter('max_lost_frames')
            .get_parameter_value().integer_value
        )
        self._min_hits: int = (
            self.get_parameter('min_hits_to_confirm')
            .get_parameter_value().integer_value
        )
        self._alpha: float = (
            self.get_parameter('velocity_smoothing_alpha')
            .get_parameter_value().double_value
        )

        # -- Internal state ----------------------------------------------
        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 0

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
            f'(assoc_thresh={self._assoc_thresh:.2f}, '
            f'max_lost={self._max_lost}, '
            f'min_hits={self._min_hits}, '
            f'alpha={self._alpha:.2f})'
        )

    # --------------------------------------------------------------------- #
    # Callback
    # --------------------------------------------------------------------- #

    def _detections_callback(self, msg: Detection3DArray) -> None:
        np = _lazy_import_numpy()
        now = time.monotonic()

        # 1. Extract detection positions and metadata.
        det_positions: List[Tuple[float, float, float]] = []
        det_classes: List[str] = []
        det_confs: List[float] = []

        for det in msg.detections:
            pos = det.bbox.center.position
            det_positions.append((pos.x, pos.y, pos.z))
            if det.results:
                det_classes.append(det.results[0].hypothesis.class_id)
                det_confs.append(det.results[0].hypothesis.score)
            else:
                det_classes.append('unknown')
                det_confs.append(0.0)

        # 2. Associate detections with existing tracks.
        matched, unmatched_dets, unmatched_tracks = self._associate(
            det_positions, np
        )

        # 3. Update matched tracks.
        for det_idx, track_id in matched:
            track = self._tracks[track_id]
            dx, dy, dz = det_positions[det_idx]

            dt = now - track.last_seen_time if track.last_seen_time > 0.0 else 0.0

            # Compute instantaneous velocity.
            if dt > 1e-6:
                inst_vx = (dx - track.x) / dt
                inst_vy = (dy - track.y) / dt
                inst_vz = (dz - track.z) / dt
            else:
                inst_vx = inst_vy = inst_vz = 0.0

            # Exponential moving average for velocity.
            a = self._alpha
            track.vx = a * inst_vx + (1.0 - a) * track.vx
            track.vy = a * inst_vy + (1.0 - a) * track.vy
            track.vz = a * inst_vz + (1.0 - a) * track.vz

            # Update position with smoothing.
            track.x = a * dx + (1.0 - a) * track.x
            track.y = a * dy + (1.0 - a) * track.y
            track.z = a * dz + (1.0 - a) * track.z

            track.class_name = det_classes[det_idx]
            track.confidence = det_confs[det_idx]
            track.hits += 1
            track.lost_count = 0
            track.last_seen_time = now

        # 4. Create new tracks for unmatched detections.
        for det_idx in unmatched_dets:
            dx, dy, dz = det_positions[det_idx]
            track = Track(
                id=self._next_id,
                x=dx, y=dy, z=dz,
                class_name=det_classes[det_idx],
                confidence=det_confs[det_idx],
                last_seen_time=now,
            )
            self._tracks[self._next_id] = track
            self._next_id += 1

        # 5. Handle unmatched (lost) tracks.
        to_remove: List[int] = []
        for track_id in unmatched_tracks:
            self._tracks[track_id].lost_count += 1
            if self._tracks[track_id].lost_count > self._max_lost:
                to_remove.append(track_id)
        for track_id in to_remove:
            del self._tracks[track_id]

        # 6. Publish confirmed tracks as Detection3DArray.
        frame_id = msg.header.frame_id if msg.header.frame_id else 'velodyne_link'
        stamp = self.get_clock().now().to_msg()

        out_msg = Detection3DArray()
        out_msg.header = Header(stamp=stamp, frame_id=frame_id)

        confirmed_tracks: List[Track] = [
            t for t in self._tracks.values() if t.hits >= self._min_hits
        ]

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

        # 7. Publish visualisation markers.
        self._publish_markers(confirmed_tracks, stamp, frame_id)

        self.get_logger().debug(
            f'Tracking: {len(self._tracks)} total, '
            f'{len(confirmed_tracks)} confirmed, '
            f'{len(to_remove)} removed'
        )

    # --------------------------------------------------------------------- #
    # Association
    # --------------------------------------------------------------------- #

    def _associate(
        self,
        det_positions: List[Tuple[float, float, float]],
        np,
    ) -> Tuple[
        List[Tuple[int, int]],  # (det_idx, track_id) matched pairs
        List[int],              # unmatched detection indices
        List[int],              # unmatched track IDs
    ]:
        """Greedy nearest-neighbour association."""
        track_ids = list(self._tracks.keys())
        n_det = len(det_positions)
        n_trk = len(track_ids)

        if n_det == 0:
            return [], [], list(track_ids)
        if n_trk == 0:
            return [], list(range(n_det)), []

        # Build cost matrix (n_det x n_trk).
        cost = np.zeros((n_det, n_trk), dtype=np.float64)
        for i, (dx, dy, dz) in enumerate(det_positions):
            for j, tid in enumerate(track_ids):
                t = self._tracks[tid]
                cost[i, j] = math.sqrt(
                    (dx - t.x) ** 2 + (dy - t.y) ** 2 + (dz - t.z) ** 2
                )

        matched: List[Tuple[int, int]] = []
        used_dets: set = set()
        used_trks: set = set()

        # Greedy: iterate in order of ascending distance.
        flat = np.argsort(cost, axis=None)
        for flat_idx in flat:
            di = int(flat_idx // n_trk)
            tj = int(flat_idx % n_trk)
            if di in used_dets or tj in used_trks:
                continue
            if cost[di, tj] > self._assoc_thresh:
                break
            matched.append((di, track_ids[tj]))
            used_dets.add(di)
            used_trks.add(tj)

        unmatched_dets = [i for i in range(n_det) if i not in used_dets]
        unmatched_trks = [track_ids[j] for j in range(n_trk) if j not in used_trks]

        return matched, unmatched_dets, unmatched_trks

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
            speed = math.sqrt(
                track.vx ** 2 + track.vy ** 2 + track.vz ** 2
            )
            if speed > 0.05:  # Only draw arrow if moving appreciably.
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
