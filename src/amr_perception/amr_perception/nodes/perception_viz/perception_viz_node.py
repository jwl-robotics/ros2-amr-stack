"""
Perception visualisation node for the AMR perception pipeline.

Produces two image streams:
  /perception/viz       -- camera image with 2D detection overlays and track info
  /perception/bird_eye  -- top-down bird's eye view of tracked objects
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection3DArray


def _lazy_imports():
    """Lazy-load heavy libraries so that node discovery stays fast."""
    import cv2
    import numpy as np
    return cv2, np


# --------------------------------------------------------------------------- #
# Colour palette (BGR for OpenCV)
# --------------------------------------------------------------------------- #

_PALETTE_BGR: List[Tuple[int, int, int]] = [
    (181, 120, 31),   # blue-ish
    (13, 128, 255),   # orange
    (43, 161, 43),    # green
    (41, 38, 214),    # red
    (189, 103, 148),  # purple
    (74, 87, 140),    # brown
    (194, 120, 227),  # pink
    (128, 128, 128),  # grey
    (33, 189, 189),   # yellow-ish
    (207, 191, 23),   # cyan-ish
]


def _colour_for_id(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE_BGR[track_id % len(_PALETTE_BGR)]


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #


class PerceptionVizNode(Node):
    """Creates combined visualisation images for the perception pipeline."""

    def __init__(self) -> None:
        super().__init__('perception_viz')

        # -- Parameters -------------------------------------------------------
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        self._img_w: int = (
            self.get_parameter('image_width')
            .get_parameter_value().integer_value
        )
        self._img_h: int = (
            self.get_parameter('image_height')
            .get_parameter_value().integer_value
        )

        # Bird's eye view parameters.
        self._bev_size: int = 500          # pixels (square image)
        self._bev_range: float = 20.0      # metres total width/height
        self._bev_scale: float = self._bev_size / self._bev_range  # px/m

        # -- Publishers -------------------------------------------------------
        self._viz_pub = self.create_publisher(Image, '/perception/viz', 10)
        self._bev_pub = self.create_publisher(Image, '/perception/bird_eye', 10)

        # -- Subscribers (independent -- not time-synchronised) ----------------
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            10,
        )
        self.create_subscription(
            Detection3DArray,
            '/perception/tracked_objects',
            self._tracked_callback,
            10,
        )
        self.create_subscription(
            Detection2DArray,
            '/perception/detections_camera',
            self._camera_det_callback,
            10,
        )

        # -- Cached latest messages -------------------------------------------
        self._latest_image: Optional[Image] = None
        self._latest_tracks: Optional[Detection3DArray] = None
        self._latest_camera_dets: Optional[Detection2DArray] = None

        # FPS tracking.
        self._frame_times: List[float] = []
        self._fps: float = 0.0

        self.get_logger().info(
            f'PerceptionVizNode initialised '
            f'(image={self._img_w}x{self._img_h}, '
            f'bev={self._bev_size}px / {self._bev_range:.0f}m)'
        )

    # --------------------------------------------------------------------- #
    # Subscriber callbacks -- cache latest data
    # --------------------------------------------------------------------- #

    def _image_callback(self, msg: Image) -> None:
        self._latest_image = msg
        self._render_camera_overlay()

    def _tracked_callback(self, msg: Detection3DArray) -> None:
        self._latest_tracks = msg
        self._render_bird_eye_view()

    def _camera_det_callback(self, msg: Detection2DArray) -> None:
        self._latest_camera_dets = msg

    # --------------------------------------------------------------------- #
    # Camera overlay (/perception/viz)
    # --------------------------------------------------------------------- #

    def _render_camera_overlay(self) -> None:
        """Draw 2D detections and track labels on the camera image."""
        if self._latest_image is None:
            return

        cv2, np = _lazy_imports()

        # Decode incoming image.
        frame = self._decode_image(self._latest_image, cv2, np)
        if frame is None:
            return

        # Update FPS counter.
        now = time.monotonic()
        self._frame_times.append(now)
        # Keep only last 30 timestamps.
        self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                self._fps = (len(self._frame_times) - 1) / dt

        # Draw 2D bounding boxes from camera detections.
        if self._latest_camera_dets is not None:
            for det in self._latest_camera_dets.detections:
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                w = det.bbox.size_x
                h = det.bbox.size_y

                x1 = int(cx - w / 2.0)
                y1 = int(cy - h / 2.0)
                x2 = int(cx + w / 2.0)
                y2 = int(cy + h / 2.0)

                # Determine class and confidence.
                class_name = 'unknown'
                conf = 0.0
                if det.results:
                    class_name = det.results[0].hypothesis.class_id
                    conf = det.results[0].hypothesis.score

                # Colour by class hash for consistency.
                colour = _PALETTE_BGR[hash(class_name) % len(_PALETTE_BGR)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                label = f'{class_name} {conf:.2f}'
                label_size, baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                # Background rectangle for readability.
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - baseline - 4),
                    (x1 + label_size[0], y1),
                    colour,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # FPS counter in the top-left corner.
        fps_text = f'FPS: {self._fps:.1f}'
        cv2.putText(
            frame, fps_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

        # Track summary in the top-right corner.
        if self._latest_tracks is not None:
            n_tracks = len(self._latest_tracks.detections)
            track_text = f'Tracks: {n_tracks}'
            text_size, _ = cv2.getTextSize(
                track_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.putText(
                frame,
                track_text,
                (frame.shape[1] - text_size[0] - 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Publish.
        self._publish_image(frame, self._viz_pub, cv2, np)

    # --------------------------------------------------------------------- #
    # Bird's eye view (/perception/bird_eye)
    # --------------------------------------------------------------------- #

    def _render_bird_eye_view(self) -> None:
        """Create a top-down view of tracked objects around the robot."""
        if self._latest_tracks is None:
            return

        cv2, np = _lazy_imports()

        size = self._bev_size
        scale = self._bev_scale

        # Blank dark image.
        bev = np.zeros((size, size, 3), dtype=np.uint8)
        bev[:] = (30, 30, 30)  # dark grey background

        # Draw grid lines for scale reference (every 2 metres).
        grid_spacing = 2.0  # metres
        grid_colour = (60, 60, 60)
        steps = int(self._bev_range / grid_spacing) + 1
        for i in range(steps):
            offset = int(i * grid_spacing * scale)
            cv2.line(bev, (offset, 0), (offset, size - 1), grid_colour, 1)
            cv2.line(bev, (0, offset), (size - 1, offset), grid_colour, 1)

        # Draw coordinate axes through the centre (robot position).
        centre = size // 2
        # X axis (forward) -- red, pointing up in BEV.
        cv2.arrowedLine(
            bev, (centre, centre), (centre, centre - int(2.0 * scale)),
            (0, 0, 200), 2, tipLength=0.2,
        )
        cv2.putText(
            bev, 'X', (centre + 5, centre - int(2.0 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA,
        )
        # Y axis (left) -- green, pointing left in BEV.
        cv2.arrowedLine(
            bev, (centre, centre), (centre - int(2.0 * scale), centre),
            (0, 200, 0), 2, tipLength=0.2,
        )
        cv2.putText(
            bev, 'Y', (centre - int(2.0 * scale) - 12, centre - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA,
        )

        # Draw robot at centre.
        cv2.circle(bev, (centre, centre), 8, (255, 255, 255), -1)
        cv2.putText(
            bev, 'Robot', (centre + 12, centre + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Draw tracked objects.
        for det in self._latest_tracks.detections:
            pos = det.bbox.center.position
            obj_x = pos.x  # forward
            obj_y = pos.y  # left

            # Convert world coords to pixel coords.
            # In BEV: pixel_x corresponds to -Y (right is positive pixel x),
            #          pixel_y corresponds to -X (down is positive pixel y).
            px = int(centre - obj_y * scale)
            py = int(centre - obj_x * scale)

            # Skip if out of view.
            if not (0 <= px < size and 0 <= py < size):
                continue

            # Parse track info.
            class_name = 'unknown'
            track_id = 0
            if det.results:
                full_label = det.results[0].hypothesis.class_id
                parts = full_label.split(':')
                class_name = parts[0]
                if len(parts) == 2:
                    try:
                        track_id = int(parts[1])
                    except ValueError:
                        pass

            colour = _colour_for_id(track_id)

            # Draw filled circle for the object.
            radius = max(5, int(0.3 * scale))
            cv2.circle(bev, (px, py), radius, colour, -1)

            # Draw label.
            label = f'{class_name} #{track_id}'
            cv2.putText(
                bev, label, (px + radius + 3, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA,
            )

            # Draw velocity arrow (if we can infer it from successive positions
            # we don't have velocity in Detection3D directly, so we parse the
            # class_id for the track_id and skip velocity here; the arrow markers
            # are handled by the tracker node's MarkerArray).
            # However, to still provide a visual cue, we can draw a short line
            # in the direction of the object from the robot (a simple heuristic).

        # Scale bar in bottom-right corner.
        bar_m = 2.0  # 2-metre bar
        bar_px = int(bar_m * scale)
        bar_y = size - 20
        bar_x_end = size - 20
        bar_x_start = bar_x_end - bar_px
        cv2.line(bev, (bar_x_start, bar_y), (bar_x_end, bar_y), (200, 200, 200), 2)
        cv2.putText(
            bev, f'{bar_m:.0f}m',
            (bar_x_start, bar_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Publish.
        self._publish_image(bev, self._bev_pub, cv2, np)

    # --------------------------------------------------------------------- #
    # Image encoding / decoding helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _decode_image(msg, cv2, np):  # noqa: N805
        """Decode a sensor_msgs/Image to a BGR numpy array."""
        try:
            height = msg.height
            width = msg.width
            encoding = msg.encoding

            if encoding in ('bgr8',):
                return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    height, width, 3
                ).copy()
            elif encoding in ('rgb8',):
                rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    height, width, 3
                )
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif encoding in ('mono8',):
                grey = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    height, width
                )
                return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
            elif encoding in ('bgra8',):
                bgra = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    height, width, 4
                )
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            elif encoding in ('rgba8',):
                rgba = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    height, width, 4
                )
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            else:
                # Fallback: attempt to use cv_bridge if available.
                try:
                    from cv_bridge import CvBridge
                    bridge = CvBridge()
                    return bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                except ImportError:
                    return None
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _publish_image(frame, publisher, cv2, np) -> None:
        """Encode a BGR numpy array as a sensor_msgs/Image and publish."""
        msg = Image()
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        publisher.publish(msg)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = PerceptionVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
