# Copyright 2025 AMR Stack Authors
# Licensed under Apache-2.0
#
# 2D camera-based object detection node for the AMR perception pipeline.
# Subscribes to raw camera images, runs YOLOv8 inference for warehouse-
# relevant object classes (person, forklift, truck, car), and publishes
# structured Detection2DArray messages alongside an annotated overlay
# image for real-time monitoring in RViz or Foxglove.

from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header

# cv_bridge is imported at the top level because it is a lightweight
# wrapper with no heavy initialisation cost.
from cv_bridge import CvBridge


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BGR colour palette for bounding-box overlays, keyed by class name.
# These are intentionally vivid so they remain legible on compressed
# camera streams.
_CLASS_COLOURS_BGR: Dict[str, Tuple[int, int, int]] = {
    "person": (0, 0, 255),      # red
    "forklift": (0, 165, 255),  # orange
    "truck": (255, 100, 0),     # blue
    "car": (0, 255, 255),       # yellow
}

_DEFAULT_COLOUR_BGR: Tuple[int, int, int] = (0, 255, 0)  # green fallback


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class CameraDetectorNode(Node):
    """ROS 2 node that performs 2-D object detection on camera images.

    The node wraps an Ultralytics YOLOv8 model and publishes per-frame
    detections as ``vision_msgs/Detection2DArray`` messages.  An annotated
    image with bounding boxes and labels is also published for real-time
    monitoring.

    The YOLOv8 model is loaded lazily on the first received image so that:

    * Node startup is fast and does not block the executor.
    * Failure to load the model is reported as a runtime error rather than
      a crash during ``__init__``.
    * The actual device (CUDA / CPU) is logged once the model is ready.

    Outputs
    -------
    * ``/perception/detections_camera`` -- ``vision_msgs/Detection2DArray``
    * ``/perception/camera_overlay`` -- ``sensor_msgs/Image`` (annotated)
    """

    def __init__(self) -> None:
        super().__init__("camera_detector")

        # ---------------------------------------------------------------
        # Declare parameters with defaults
        # ---------------------------------------------------------------
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter(
            "target_classes", ["person", "forklift", "truck", "car"]
        )
        self.declare_parameter("image_scale", 0.5)

        # ---------------------------------------------------------------
        # Cache resolved parameter values
        # ---------------------------------------------------------------
        self._model_path: str = (
            self.get_parameter("model_path").value
        )
        self._confidence_threshold: float = (
            self.get_parameter("confidence_threshold").value
        )
        self._device: str = (
            self.get_parameter("device").value
        )
        self._target_classes: List[str] = list(
            self.get_parameter("target_classes").value
        )
        self._image_scale: float = (
            self.get_parameter("image_scale").value
        )

        # ---------------------------------------------------------------
        # QoS: Sensor-data profile (best-effort, keep-last 5)
        # ---------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---------------------------------------------------------------
        # Subscriber
        # ---------------------------------------------------------------
        self._image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_callback,
            sensor_qos,
        )

        # ---------------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------------
        self._detections_pub = self.create_publisher(
            Detection2DArray, "/perception/detections_camera", 10
        )
        self._overlay_pub = self.create_publisher(
            Image, "/perception/camera_overlay", 10
        )

        # ---------------------------------------------------------------
        # Internal state
        # ---------------------------------------------------------------
        self._bridge = CvBridge()
        self._model = None  # Lazy-loaded YOLO model
        self._model_class_names: Dict[int, str] = {}  # id -> name from model
        self._callback_count: int = 0
        self._last_log_time: Optional[float] = None

        self.get_logger().info(
            "CameraDetectorNode initialised  "
            f"(model={self._model_path}, "
            f"conf={self._confidence_threshold}, "
            f"device={self._device}, "
            f"scale={self._image_scale})"
        )

    # -------------------------------------------------------------------
    # Lazy model loading
    # -------------------------------------------------------------------

    def _ensure_model_loaded(self) -> bool:
        """Load the YOLO model on first use.

        Returns ``True`` if the model is ready, ``False`` on failure.
        """
        if self._model is not None:
            return True

        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().error(
                "ultralytics is required but not installed.  "
                "Install with: pip install ultralytics"
            )
            return False

        try:
            self.get_logger().info(
                f"Loading YOLO model '{self._model_path}' ..."
            )
            model = YOLO(self._model_path)

            # Determine the actual device.  If CUDA was requested but is not
            # available, fall back to CPU transparently.
            import torch
            if "cuda" in self._device and not torch.cuda.is_available():
                self.get_logger().warn(
                    f"CUDA requested ({self._device}) but not available; "
                    "falling back to CPU"
                )
                self._device = "cpu"

            # Cache the model class-name mapping (COCO-style).
            self._model_class_names = (
                model.names if hasattr(model, "names") else {}
            )
            self._model = model

            self.get_logger().info(
                f"YOLO model loaded on device '{self._device}'  "
                f"({len(self._model_class_names)} classes)"
            )
            return True

        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Failed to load YOLO model: {exc}"
            )
            return False

    # -------------------------------------------------------------------
    # Main callback
    # -------------------------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        """Process an incoming camera image.

        Pipeline stages:
        1. Convert ``sensor_msgs/Image`` to an OpenCV BGR array.
        2. Resize for faster inference.
        3. Run YOLOv8 forward pass.
        4. Filter by confidence and target class list.
        5. Scale bounding boxes back to original resolution.
        6. Publish ``Detection2DArray``.
        7. Draw annotated overlay and publish.
        """
        import cv2

        # ------ Lazy load the model on first call ------
        if not self._ensure_model_loaded():
            return

        # ------ 1. ROS Image -> OpenCV ------
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"cv_bridge conversion failed: {exc}"
            )
            return

        orig_h, orig_w = cv_image.shape[:2]

        # ------ 2. Resize for faster inference ------
        if self._image_scale != 1.0:
            scaled_w = int(orig_w * self._image_scale)
            scaled_h = int(orig_h * self._image_scale)
            inference_image = cv2.resize(
                cv_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            inference_image = cv_image
            scaled_w, scaled_h = orig_w, orig_h

        # ------ 3. YOLOv8 inference ------
        results = self._model(
            inference_image,
            device=self._device,
            conf=self._confidence_threshold,
            verbose=False,
        )

        # ------ 4-6. Build Detection2DArray ------
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        # Scale factors to map bounding boxes from the inference resolution
        # back to the original image dimensions.
        sx = orig_w / scaled_w
        sy = orig_h / scaled_h

        overlay = cv_image.copy()
        class_counts: Dict[str, int] = {}

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Extract fields from the result tensor.
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                class_name = self._model_class_names.get(cls_id, str(cls_id))

                # ------ Filter by target classes ------
                if class_name not in self._target_classes:
                    continue

                # Bounding box in inference-image pixel coordinates (xyxy).
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                # ------ 5. Scale back to original resolution ------
                x1 *= sx
                y1 *= sy
                x2 *= sx
                y2 *= sy

                bbox_w = x2 - x1
                bbox_h = y2 - y1
                bbox_cx = x1 + bbox_w / 2.0
                bbox_cy = y1 + bbox_h / 2.0

                # ------ 6. Build Detection2D ------
                det = Detection2D()
                det.header = msg.header

                det.bbox.center.position.x = bbox_cx
                det.bbox.center.position.y = bbox_cy
                det.bbox.size_x = bbox_w
                det.bbox.size_y = bbox_h

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = conf
                det.results.append(hypothesis)

                detection_array.detections.append(det)

                class_counts[class_name] = (
                    class_counts.get(class_name, 0) + 1
                )

                # ------ 7. Draw on overlay ------
                colour = _CLASS_COLOURS_BGR.get(
                    class_name, _DEFAULT_COLOUR_BGR
                )
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(overlay, pt1, pt2, colour, 2)

                label = f"{class_name} {conf:.2f}"
                label_size, baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                # Draw a filled rectangle behind the text for readability.
                cv2.rectangle(
                    overlay,
                    (pt1[0], pt1[1] - label_size[1] - baseline - 4),
                    (pt1[0] + label_size[0], pt1[1]),
                    colour,
                    cv2.FILLED,
                )
                cv2.putText(
                    overlay,
                    label,
                    (pt1[0], pt1[1] - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # ------ Publish detections ------
        self._detections_pub.publish(detection_array)

        # ------ Publish annotated overlay ------
        try:
            overlay_msg = self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            overlay_msg.header = msg.header
            self._overlay_pub.publish(overlay_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Failed to publish overlay image: {exc}"
            )

        # ------ Throttled logging (every 2 seconds) ------
        self._callback_count += 1
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_log_time is None or (now - self._last_log_time) >= 2.0:
            self._last_log_time = now
            total = sum(class_counts.values())
            breakdown = ", ".join(
                f"{k}: {v}" for k, v in sorted(class_counts.items())
            )
            self.get_logger().info(
                f"Frame {self._callback_count}: {total} detection(s) "
                f"[{breakdown}]"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: Optional[list] = None) -> None:
    """Spin the camera detector node."""
    rclpy.init(args=args)
    node = CameraDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
