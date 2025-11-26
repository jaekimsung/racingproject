"""ROS2 node wiring path handling, speed PID, and Stanley steering control."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Vector3Stamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from racingproject.speed_pid import SpeedPID

from .stanley_controller import StanleyController


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class RacingNodeStanley(Node):
    """Main ROS2 node using Stanley lateral control."""

    def __init__(self):
        super().__init__("racing_node_stanley")
        self.declare_parameters(
            "",
            [
                ("path_csv", ""),  # 기준 경로 CSV 파일 경로
                ("lookahead_distance", 15.0),  # - lookahead_distance: 제어 시 앞쪽으로 볼 거리 [m]
                ("speed_kp", 0.5),  # - speed_kp/ki/kd: 속도 PID 게인
                ("speed_ki", 0.1),
                ("speed_kd", 0.01),
                ("v_high", 15.0),  # - v_high/v_low: 직선/코너에서 목표 속도 [m/s]
                ("v_low", 8.0),
                ("kappa_th", 0.05),  # - kappa_th: 코너 판단용 곡률 임계값
                ("control_dt", 0.05),  # - control_dt: 제어 주기/샘플 타임 [s]
                ("max_steer_deg", 20.0),  # - max_steer_deg: 최대 조향각 [deg]
                ("max_steer_rate_deg", 45.0),  # - max_steer_rate_deg: 최대 조향각 속도 [deg/s]
                ("wheelbase", 1.023),  # - wheelbase: 휠베이스 길이 [m]
                ("stanley_k", 1.5),  # - stanley_k: 횡방향 오차 게인
                ("stanley_softening", 0.1),  # - stanley_softening: 저속에서의 소프트닝 항 [m/s]
                ("stanley_heading_gain", 1.0),  # - stanley_heading_gain: 헤딩 오차 가중치
            ],
        )

        path_csv_param = self.get_parameter("path_csv").get_parameter_value().string_value
        path_csv = self._resolve_path_csv(path_csv_param)

        # Speed/curvature thresholds are needed before visualization markers are prepared.
        self.v_high = float(self.get_parameter("v_high").value)
        self.v_low = float(self.get_parameter("v_low").value)
        self.kappa_th = float(self.get_parameter("kappa_th").value)
        self.control_dt = float(self.get_parameter("control_dt").value)
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)

        self.max_steer = math.radians(float(self.get_parameter("max_steer_deg").value))
        self.max_steer_rate = math.radians(float(self.get_parameter("max_steer_rate_deg").value))
        self.wheelbase = float(self.get_parameter("wheelbase").value)

        self.path_xy: Optional[np.ndarray] = None
        self.path_kappa: Optional[np.ndarray] = None
        self.viz_timer = None
        if path_csv:
            try:
                self.path_xy, self.path_kappa = self._load_path(path_csv)
                self.get_logger().info(f"Loaded path from {path_csv} with {len(self.path_xy)} points.")
                self._prepare_visualization_msgs()
            except Exception as exc:
                self.get_logger().error(f"Failed to initialize path: {exc}")
        else:
            self.get_logger().warn("No path CSV could be resolved. Set parameter 'path_csv' to begin.")

        self.speed_pid = SpeedPID(
            kp=float(self.get_parameter("speed_kp").value),
            ki=float(self.get_parameter("speed_ki").value),
            kd=float(self.get_parameter("speed_kd").value),
        )

        self.stanley = StanleyController(
            k=float(self.get_parameter("stanley_k").value),
            ks=float(self.get_parameter("stanley_softening").value),
            heading_gain=float(self.get_parameter("stanley_heading_gain").value),
            max_steer=self.max_steer,
            max_steer_rate=self.max_steer_rate,
            wheelbase=self.wheelbase,
        )

        self.state_sub = self.create_subscription(
            Float32MultiArray,
            "/mobile_system_control/ego_vehicle",
            self.state_callback,
            10,
        )

        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.cmd_pub = self.create_publisher(Vector3Stamped, "/mobile_system_control/control_msg", 10)
        self.path_pub = self.create_publisher(PathMsg, "/mobile_system_control/racing_path", latched_qos)
        self.pose_pub = self.create_publisher(PathMsg, "/mobile_system_control/ego_pose_path", 1)
        self.decel_marker_pub = self.create_publisher(MarkerArray, "/mobile_system_control/decel_zones", latched_qos)
        self.timer = self.create_timer(self.control_dt, self.control_loop)

        # Kick off visualization publication after publishers exist.
        if self.path_xy is not None:
            self._publish_visualization()
            self.viz_timer = self.create_timer(0.5, self._publish_visualization)

        self.current_state: Optional[Tuple[float, float, float, float, float]] = None
        self.last_time = self.get_clock().now()
        self.racing_path_msg: Optional[PathMsg] = None
        self.decel_markers: Optional[MarkerArray] = None
        self.pose_path_msg = PathMsg()

    def state_callback(self, msg: Float32MultiArray) -> None:
        """Store the latest ego state from the simulator."""
        if len(msg.data) < 5:
            self.get_logger().warn("Received ego state with insufficient length.")
            return
        self.current_state = (
            float(msg.data[0]),
            float(msg.data[1]),
            float(msg.data[2]),
            float(msg.data[3]),
            float(msg.data[4]),
        )

    def control_loop(self) -> None:
        """Main control loop invoked by ROS timer."""
        if self.current_state is None or self.path_xy is None:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 1e-5:
            return
        self.last_time = now

        x, y, theta, v_meas, delta_meas = self.current_state
        if self.path_xy is None or len(self.path_xy) == 0:
            return

        idx = self._find_closest_index(x, y)
        path_segment, kappa_segment = self._segment_by_distance(idx, self.lookahead_distance)

        if len(path_segment) < 2:
            self.get_logger().warn("Path segment too short for control.")
            return

        k_local = float(np.max(np.abs(kappa_segment))) if len(kappa_segment) > 0 else 0.0
        v_ref = self.v_low if k_local > self.kappa_th else self.v_high

        throttle, brake = self.speed_pid.step(v_ref, v_meas, dt)

        e_y, e_psi = self._compute_errors(path_segment, x, y, theta)
        kappa_ref = float(self.path_kappa[idx]) if self.path_kappa is not None else 0.0
        steer_cmd = self.stanley.compute(e_y, e_psi, kappa_ref, v_meas, delta_meas, dt)
        steer_norm = float(np.clip(steer_cmd / max(self.max_steer, 1e-6), -1.0, 1.0))

        # Log target speed and steering command for each control cycle.
        steer_deg = math.degrees(steer_cmd)
        print(
            f"[STANLEY CONTROL] v_ref={v_ref:.2f} m/s, v={v_meas:.2f} m/s, "
            f"steer_cmd={steer_cmd:.3f} rad ({steer_deg:.2f} deg) "
            f"(norm={steer_norm:.3f}), e_y={e_y:.2f} m"
        )

        msg = Vector3Stamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "Team8"
        msg.vector.x = throttle
        msg.vector.y = steer_norm
        msg.vector.z = brake
        self.cmd_pub.publish(msg)
        print(msg)

        if self.racing_path_msg is not None:
            self.racing_path_msg.header.stamp = now.to_msg()
            self.path_pub.publish(self.racing_path_msg)
        if self.decel_markers is not None:
            for marker in self.decel_markers.markers:
                marker.header.stamp = now.to_msg()
            self.decel_marker_pub.publish(self.decel_markers)

        pose_msg = self._pose_stamped(x, y, theta, now)
        self.pose_path_msg.header.stamp = now.to_msg()
        self.pose_path_msg.header.frame_id = "map"
        self.pose_path_msg.poses.append(pose_msg)
        if len(self.pose_path_msg.poses) > 500:
            self.pose_path_msg.poses = self.pose_path_msg.poses[-500:]
        self.pose_pub.publish(self.pose_path_msg)

    def _publish_visualization(self) -> None:
        """Publish static path and decel markers for RViz visualization."""
        if self.racing_path_msg is not None:
            now = self.get_clock().now()
            self.racing_path_msg.header.stamp = now.to_msg()
            self.path_pub.publish(self.racing_path_msg)
        if self.decel_markers is not None:
            now = self.get_clock().now()
            for marker in self.decel_markers.markers:
                marker.header.stamp = now.to_msg()
            self.decel_marker_pub.publish(self.decel_markers)
        elif self.path_xy is None:
            self.get_logger().warn_once("Visualization not published because path is unavailable. Check 'path_csv'.")

    def _compute_errors(self, path_segment: np.ndarray, x: float, y: float, theta: float) -> Tuple[float, float]:
        """
        Compute lateral and heading error to the first segment of the lookahead path.
        """
        target = path_segment[min(1, len(path_segment) - 1)]
        dx = target[0] - x
        dy = target[1] - y

        # Transform into vehicle frame: forward = x-axis, left = y-axis.
        e_y = -math.sin(theta) * dx + math.cos(theta) * dy

        if len(path_segment) >= 2:
            next_pt = path_segment[min(2, len(path_segment) - 1)]
            path_yaw = math.atan2(next_pt[1] - target[1], next_pt[0] - target[0])
        else:
            path_yaw = math.atan2(dy, dx)
        e_psi = wrap_angle(path_yaw - theta)
        return e_y, e_psi

    def _resolve_path_csv(self, param_value: str) -> str:
        """
        Resolve the path CSV location, preferring an explicit parameter then packaged default.
        """
        if param_value:
            return param_value

        candidates = []
        for pkg in ("racingproject_stanley", "racingproject"):
            try:
                share_dir = get_package_share_directory(pkg)
                candidates.append(Path(share_dir) / "data" / "waypoints.csv")
            except (PackageNotFoundError, Exception):
                continue

        candidates.append(Path(__file__).resolve().parent / ".." / "racingproject" / "data" / "waypoints.csv")
        candidates.append(Path(__file__).resolve().parent / "data" / "waypoints.csv")

        for candidate in candidates:
            if candidate.exists():
                self.get_logger().info(f"Using packaged path CSV at {candidate}")
                return str(candidate)

        self.get_logger().warn(
            "Could not find default waypoints.csv. Please set 'path_csv' parameter to a valid CSV file."
        )
        return ""

    def _load_path(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load XY path directly from CSV and compute curvature."""
        try:
            data = np.loadtxt(csv_path, delimiter=",")
        except ValueError:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError("CSV must contain at least two numeric columns for x and y.")
        xy = data[:, :2]
        s = self._compute_arclength(xy)
        kappa = self._compute_curvature(xy, s)
        return xy, kappa

    @staticmethod
    def _compute_arclength(xy: np.ndarray) -> np.ndarray:
        dx = np.diff(xy[:, 0])
        dy = np.diff(xy[:, 1])
        ds = np.hypot(dx, dy)
        return np.concatenate([[0.0], np.cumsum(ds)])

    def _compute_curvature(self, xy: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Compute signed curvature along the path."""
        if len(xy) < 3:
            return np.zeros(len(xy))

        dx_ds = np.gradient(xy[:, 0], s)
        dy_ds = np.gradient(xy[:, 1], s)
        ddx_ds = np.gradient(dx_ds, s)
        ddy_ds = np.gradient(dy_ds, s)

        cross = dx_ds * ddy_ds - dy_ds * ddx_ds
        denom = (dx_ds**2 + dy_ds**2) ** 1.5 + 1e-6
        return cross / denom

    def _find_closest_index(self, x: float, y: float) -> int:
        """Return index of the closest point on the loaded path."""
        if self.path_xy is None or len(self.path_xy) == 0:
            return 0
        delta = self.path_xy - np.array([[x, y]])
        dists = np.einsum("ij,ij->i", delta, delta)
        return int(np.argmin(dists))

    def _segment_by_distance(self, start_idx: int, lookahead_dist: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect path points forward from start_idx until lookahead_dist is reached.

        Wraps around the path to maintain continuous control on closed loops.
        """
        if self.path_xy is None or len(self.path_xy) == 0:
            return np.empty((0, 2)), np.empty((0,))
        n = len(self.path_xy)
        pts = [self.path_xy[start_idx]]
        kappas = [self.path_kappa[start_idx]] if self.path_kappa is not None else []
        total = 0.0
        idx = start_idx
        # Guard against infinite loops by limiting to n points.
        for _ in range(n - 1):
            next_idx = (idx + 1) % n
            ds = float(np.hypot(*(self.path_xy[next_idx] - self.path_xy[idx])))
            total += ds
            pts.append(self.path_xy[next_idx])
            if self.path_kappa is not None:
                kappas.append(self.path_kappa[next_idx])
            idx = next_idx
            if total >= lookahead_dist or next_idx == start_idx:
                break
        return np.array(pts), np.array(kappas) if self.path_kappa is not None else np.array([])

    def _prepare_visualization_msgs(self) -> None:
        """Precompute static visualization messages for RViz."""
        if self.path_xy is None or self.path_kappa is None:
            return

        self.racing_path_msg = PathMsg()
        self.racing_path_msg.header.frame_id = "map"
        xy = self.path_xy
        headings = self._path_headings(xy)
        now = self.get_clock().now()
        for (px, py), yaw in zip(xy, headings):
            self.racing_path_msg.poses.append(self._pose_stamped(px, py, yaw, now))

        kappa = np.abs(self.path_kappa)
        mask = kappa > self.kappa_th
        decel_points = xy[mask]
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "decel_zones"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.4
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.points = [self._point(px, py) for px, py in decel_points]
        self.decel_markers = MarkerArray(markers=[marker])

    @staticmethod
    def _point(x: float, y: float):
        return Point(x=x, y=y, z=0.0)

    @staticmethod
    def _path_headings(xy: np.ndarray) -> np.ndarray:
        """Approximate heading at each point of the path."""
        if len(xy) < 2:
            return np.zeros(len(xy))
        diffs = np.diff(xy, axis=0, append=xy[-1:])
        return np.arctan2(diffs[:, 1], diffs[:, 0])

    @staticmethod
    def _pose_stamped(x: float, y: float, yaw: float, stamp) -> PoseStamped:
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp.to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        return pose_msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RacingNodeStanley()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
