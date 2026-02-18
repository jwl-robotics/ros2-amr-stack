# Copyright 2025 AMR Stack Authors
# Licensed under Apache-2.0
#
# 3D obstacle detection node for the AMR perception pipeline.
# Subscribes to filtered point clouds, performs Euclidean clustering via
# Open3D DBSCAN, classifies detected clusters by geometric heuristics
# (person, shelf, pallet, generic obstacle), and publishes structured
# Detection3DArray messages alongside RViz-ready MarkerArray visualizations.

import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Colour palette for RViz markers keyed by semantic class.
_CLASS_COLOURS: Dict[str, Tuple[float, float, float, float]] = {
    "person": (1.0, 0.2, 0.2, 0.85),   # red
    "shelf": (0.2, 0.4, 1.0, 0.85),     # blue
    "pallet": (1.0, 0.9, 0.2, 0.85),    # yellow
    "obstacle": (0.2, 0.9, 0.3, 0.85),  # green
}

# Twelve edges of an axis-aligned bounding box, each expressed as a pair
# of corner indices (see _BOX_CORNERS ordering below).
_BOX_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
]


# ---------------------------------------------------------------------------
# Helper: PointCloud2 binary parser
# ---------------------------------------------------------------------------

def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """Convert a PointCloud2 message to an (N, 3) float64 numpy array.

    This function reads the raw binary payload directly using :mod:`struct`,
    so it has **no dependency on Open3D or PCL** at import time.  It handles
    both organised and unorganised clouds and filters NaN / Inf values.

    Parameters
    ----------
    msg : PointCloud2
        Incoming ROS 2 point cloud message.  Must contain ``x``, ``y``, ``z``
        fields of type ``FLOAT32`` (datatype 7).

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` with dtype ``float64``.  Returns an empty
        ``(0, 3)`` array when the cloud contains no valid points.
    """
    # Build a lookup table: field name -> byte offset within one point record.
    field_offsets: Dict[str, int] = {}
    for field in msg.fields:
        field_offsets[field.name] = field.offset

    # Verify required fields exist.
    for axis in ("x", "y", "z"):
        if axis not in field_offsets:
            return np.empty((0, 3), dtype=np.float64)

    x_off = field_offsets["x"]
    y_off = field_offsets["y"]
    z_off = field_offsets["z"]

    point_step = msg.point_step
    num_points = msg.width * msg.height
    data = bytes(msg.data)

    # Pre-allocate and fill via struct unpacking.  Using a memoryview avoids
    # repeated slicing copies for large clouds.
    points = np.empty((num_points, 3), dtype=np.float64)
    view = memoryview(data)

    for i in range(num_points):
        base = i * point_step
        points[i, 0] = struct.unpack_from("<f", view, base + x_off)[0]
        points[i, 1] = struct.unpack_from("<f", view, base + y_off)[0]
        points[i, 2] = struct.unpack_from("<f", view, base + z_off)[0]

    # Discard any NaN / Inf points that survived the upstream filter.
    valid_mask = np.isfinite(points).all(axis=1)
    return points[valid_mask]


# ---------------------------------------------------------------------------
# Helper: geometric classification
# ---------------------------------------------------------------------------

def classify_cluster(
    width: float,
    depth: float,
    height: float,
    params: Dict[str, float],
) -> str:
    """Classify a cluster into a semantic class based on bounding-box geometry.

    The classification scheme is designed for typical warehouse environments:

    * **person** -- upright, narrow objects of human proportions.
    * **shelf** -- tall, narrow vertical structures.
    * **pallet** -- low, wide flat objects on the floor.
    * **obstacle** -- anything that does not match the above rules.

    Parameters
    ----------
    width : float
        Bounding-box extent along the X axis (metres).
    depth : float
        Bounding-box extent along the Y axis (metres).
    height : float
        Bounding-box extent along the Z axis (metres).
    params : dict
        Threshold dictionary with keys ``person_width_min``,
        ``person_width_max``, ``person_height_min``, ``person_height_max``,
        ``shelf_min_height``, ``shelf_max_width``, ``pallet_max_height``,
        ``pallet_min_width``.

    Returns
    -------
    str
        One of ``"person"``, ``"shelf"``, ``"pallet"``, or ``"obstacle"``.
    """
    # Horizontal footprint -- use the larger of width/depth so orientation
    # does not matter.
    footprint = max(width, depth)
    footprint_min = min(width, depth)

    # Person: narrow footprint in the expected width band AND human height.
    if (params["person_width_min"] <= footprint_min <= params["person_width_max"]
            and params["person_height_min"] <= height <= params["person_height_max"]):
        return "person"

    # Shelf: tall and narrow.
    if height > params["shelf_min_height"] and footprint < params["shelf_max_width"]:
        return "shelf"

    # Pallet: low and wide.
    if height < params["pallet_max_height"] and footprint > params["pallet_min_width"]:
        return "pallet"

    return "obstacle"


# ---------------------------------------------------------------------------
# Helper: wireframe bounding-box marker for RViz
# ---------------------------------------------------------------------------

def create_bbox_marker(
    marker_id: int,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    class_name: str,
    header: Header,
) -> Marker:
    """Build a LINE_LIST marker that draws a wireframe bounding box in RViz.

    Parameters
    ----------
    marker_id : int
        Unique identifier for the marker within its namespace.
    center : tuple of float
        ``(cx, cy, cz)`` centre of the bounding box in the cloud frame.
    size : tuple of float
        ``(sx, sy, sz)`` full extents of the bounding box.
    class_name : str
        Semantic label used to select the line colour.
    header : Header
        ROS header (frame_id + stamp) to attach to the marker.

    Returns
    -------
    Marker
        A ``visualization_msgs/Marker`` of type ``LINE_LIST``.
    """
    marker = Marker()
    marker.header = header
    marker.ns = "obstacle_clusters"
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    # Line width in RViz (metres in world space for LINE_LIST).
    marker.scale.x = 0.03

    r, g, b, a = _CLASS_COLOURS.get(class_name, (0.8, 0.8, 0.8, 0.8))
    marker.color = ColorRGBA(r=r, g=g, b=b, a=a)

    # Lifetime: auto-delete if no update within 0.5 s (prevents ghost markers
    # when a cluster disappears).
    marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

    # Compute the eight corners of the axis-aligned bounding box.
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0

    corners = [
        (cx - hx, cy - hy, cz - hz),  # 0  bottom-back-left
        (cx + hx, cy - hy, cz - hz),  # 1  bottom-back-right
        (cx + hx, cy + hy, cz - hz),  # 2  bottom-front-right
        (cx - hx, cy + hy, cz - hz),  # 3  bottom-front-left
        (cx - hx, cy - hy, cz + hz),  # 4  top-back-left
        (cx + hx, cy - hy, cz + hz),  # 5  top-back-right
        (cx + hx, cy + hy, cz + hz),  # 6  top-front-right
        (cx - hx, cy + hy, cz + hz),  # 7  top-front-left
    ]

    for i0, i1 in _BOX_EDGES:
        p0 = Point(x=corners[i0][0], y=corners[i0][1], z=corners[i0][2])
        p1 = Point(x=corners[i1][0], y=corners[i1][1], z=corners[i1][2])
        marker.points.append(p0)
        marker.points.append(p1)

    return marker


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ObstacleDetectorNode(Node):
    """ROS 2 node that detects and classifies 3-D obstacles from point clouds.

    The node consumes filtered ``sensor_msgs/PointCloud2`` messages (typically
    produced by the ``amr_pointcloud_filter`` node), clusters the points using
    Open3D DBSCAN, then classifies each cluster by its axis-aligned bounding-box
    dimensions.

    Outputs
    -------
    * ``/perception/obstacles_3d`` -- ``vision_msgs/Detection3DArray``
    * ``/perception/clusters_viz`` -- ``visualization_msgs/MarkerArray``

    All heavyweight dependencies (Open3D) are imported lazily on the first
    callback so that the node starts quickly and fails gracefully if the
    library is missing.
    """

    def __init__(self) -> None:
        super().__init__("obstacle_detector")

        # ---------------------------------------------------------------
        # Declare parameters with defaults
        # ---------------------------------------------------------------
        self.declare_parameter("cluster_tolerance", 0.3)
        self.declare_parameter("min_cluster_size", 10)
        self.declare_parameter("max_cluster_size", 5000)
        self.declare_parameter("person_width_range", [0.3, 0.8])
        self.declare_parameter("person_height_range", [1.2, 2.0])
        self.declare_parameter("shelf_min_height", 1.5)
        self.declare_parameter("shelf_max_width", 0.6)
        self.declare_parameter("pallet_max_height", 0.3)
        self.declare_parameter("pallet_min_width", 0.8)

        # ---------------------------------------------------------------
        # Cache resolved parameter values
        # ---------------------------------------------------------------
        self._cluster_tolerance: float = (
            self.get_parameter("cluster_tolerance").value
        )
        self._min_cluster_size: int = (
            self.get_parameter("min_cluster_size").value
        )
        self._max_cluster_size: int = (
            self.get_parameter("max_cluster_size").value
        )

        person_width = self.get_parameter("person_width_range").value
        person_height = self.get_parameter("person_height_range").value

        self._classify_params: Dict[str, float] = {
            "person_width_min": float(person_width[0]),
            "person_width_max": float(person_width[1]),
            "person_height_min": float(person_height[0]),
            "person_height_max": float(person_height[1]),
            "shelf_min_height": float(
                self.get_parameter("shelf_min_height").value
            ),
            "shelf_max_width": float(
                self.get_parameter("shelf_max_width").value
            ),
            "pallet_max_height": float(
                self.get_parameter("pallet_max_height").value
            ),
            "pallet_min_width": float(
                self.get_parameter("pallet_min_width").value
            ),
        }

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
        self._cloud_sub = self.create_subscription(
            PointCloud2,
            "/cloud_filtered",
            self._cloud_callback,
            sensor_qos,
        )

        # ---------------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------------
        self._detections_pub = self.create_publisher(
            Detection3DArray, "/perception/obstacles_3d", 10
        )
        self._markers_pub = self.create_publisher(
            MarkerArray, "/perception/clusters_viz", 10
        )

        # ---------------------------------------------------------------
        # Internal state
        # ---------------------------------------------------------------
        self._o3d = None  # Lazy-loaded Open3D module reference
        self._callback_count: int = 0
        self._last_log_time: Optional[float] = None

        self.get_logger().info(
            "ObstacleDetectorNode initialised  "
            f"(eps={self._cluster_tolerance}, "
            f"min_pts={self._min_cluster_size}, "
            f"max_pts={self._max_cluster_size})"
        )

    # -------------------------------------------------------------------
    # Main callback
    # -------------------------------------------------------------------

    def _cloud_callback(self, msg: PointCloud2) -> None:
        """Process an incoming filtered point cloud.

        Pipeline stages:
        1. Parse binary ``PointCloud2`` into an (N, 3) numpy array.
        2. Cluster points with Open3D DBSCAN.
        3. Compute axis-aligned bounding boxes per cluster.
        4. Classify each cluster by geometric heuristics.
        5. Publish ``Detection3DArray`` and ``MarkerArray``.
        """
        # ------ Lazy import of Open3D on the first callback ------
        if self._o3d is None:
            try:
                import open3d as o3d
                self._o3d = o3d
                self.get_logger().info(
                    f"Open3D {o3d.__version__} loaded successfully"
                )
            except ImportError:
                self.get_logger().error(
                    "Open3D is required but not installed.  "
                    "Install with: pip install open3d"
                )
                return

        o3d = self._o3d

        # ------ 1. Parse PointCloud2 to numpy ------
        points = pointcloud2_to_xyz_array(msg)

        if points.shape[0] < self._min_cluster_size:
            # Not enough points to form even one cluster -- publish empties
            # so downstream nodes see a heartbeat.
            self._publish_empty(msg.header)
            return

        # ------ 2. DBSCAN clustering via Open3D ------
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        labels = np.asarray(
            pcd.cluster_dbscan(
                eps=self._cluster_tolerance,
                min_points=self._min_cluster_size,
                print_progress=False,
            )
        )

        if labels.size == 0:
            self._publish_empty(msg.header)
            return

        unique_labels = set(labels)
        unique_labels.discard(-1)  # -1 = noise

        # ------ 3-6. Process clusters ------
        detection_array = Detection3DArray()
        detection_array.header = msg.header

        marker_array = MarkerArray()
        marker_id = 0

        class_counts: Dict[str, int] = {}

        for label in sorted(unique_labels):
            cluster_mask = labels == label
            cluster_points = points[cluster_mask]

            # Enforce maximum cluster size (very large clusters are usually
            # walls or other background geometry).
            if cluster_points.shape[0] > self._max_cluster_size:
                continue

            # Axis-aligned bounding box.
            mins = cluster_points.min(axis=0)
            maxs = cluster_points.max(axis=0)
            center = (mins + maxs) / 2.0
            size = maxs - mins  # (width_x, depth_y, height_z)

            width = float(size[0])
            depth = float(size[1])
            height = float(size[2])

            # ------ 4. Classify ------
            class_name = classify_cluster(
                width, depth, height, self._classify_params
            )
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # ------ 5. Build Detection3D message ------
            det = Detection3D()
            det.header = msg.header

            det.bbox.center.position.x = float(center[0])
            det.bbox.center.position.y = float(center[1])
            det.bbox.center.position.z = float(center[2])
            det.bbox.size.x = width
            det.bbox.size.y = depth
            det.bbox.size.z = height

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = 1.0
            det.results.append(hypothesis)

            detection_array.detections.append(det)

            # ------ 6. Build wireframe marker ------
            marker = create_bbox_marker(
                marker_id,
                (float(center[0]), float(center[1]), float(center[2])),
                (width, depth, height),
                class_name,
                msg.header,
            )
            marker_array.markers.append(marker)
            marker_id += 1

        # Publish results.
        self._detections_pub.publish(detection_array)
        self._markers_pub.publish(marker_array)

        # ------ 7. Throttled logging (every 2 seconds) ------
        self._callback_count += 1
        now = self.get_clock().now().nanoseconds * 1e-9
        if (
            self._last_log_time is None
            or (now - self._last_log_time) >= 2.0
        ):
            self._last_log_time = now
            total = sum(class_counts.values())
            breakdown = ", ".join(
                f"{k}: {v}" for k, v in sorted(class_counts.items())
            )
            self.get_logger().info(
                f"Detected {total} cluster(s) from {points.shape[0]} pts "
                f"[{breakdown}]"
            )

    # -------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------

    def _publish_empty(self, header: Header) -> None:
        """Publish empty detection and marker arrays as a heartbeat."""
        det = Detection3DArray()
        det.header = header
        self._detections_pub.publish(det)
        self._markers_pub.publish(MarkerArray())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: Optional[list] = None) -> None:
    """Spin the obstacle detector node."""
    rclpy.init(args=args)
    node = ObstacleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
