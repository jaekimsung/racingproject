"""ROS2 node wiring together path management, speed PID, and steering MPC."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Vector3Stamped
from pathlib import Path
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from .speed_pid import SpeedPID
from .steering_mpc import SteeringMPC


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class RacingNode(Node):
    """Main ROS2 node handling perception of vehicle state and control outputs."""

    def __init__(self):
        super().__init__("racing_node")
        self.path_manager = None
        self.declare_parameters(
            "",
            [
                ("path_csv", ""), # 기준 경로 CSV 파일 경로
                ("sample_ds", 0.5), # - sample_ds: 경로 리샘플 간격 [m]
                
                ("max_offset", 3.0), # - max_offset: 곡률 기반 레이싱 라인 최대 횡방향 오프셋 [m]
                ("offset_gain", 1.0), # - offset_gain: 곡률→오프셋 스케일
                ("offset_power", 1.0), # - offset_power: 곡률 정규화 후 거듭제곱 (비선형성)
                ("curvature_smooth_window", 11), # - curvature_smooth_window: 곡률 이동 평균 창 크기
                ("lookahead_points", 30), # - lookahead_points: 제어 시 앞쪽으로 볼 포인트 개수
                
                ("speed_kp", 0.5), # - speed_kp/ki/kd: 속도 PID 게인
                ("speed_ki", 0.1),
                ("speed_kd", 0.01),
                
                ("v_high", 15.0), # - v_high/v_low: 직선/코너에서 목표 속도 [m/s]
                ("v_low", 8.0),
                
                ("kappa_th", 0.05), # - kappa_th: 코너 판단용 곡률 임계값
                
                ("mpc_Np", 10), # - mpc_Np: MPC 예측 지평선 길이
                ("mpc_Nc", 5), # - mpc_Nc: MPC 제어 지평선 길이
                ("control_dt", 0.05), # - control_dt: 제어 주기/샘플 타임 [s]
                ("max_steer_deg", 20.0), # - max_steer_deg: 최대 조향각 [deg]
                ("max_steer_rate_deg", 45.0), # - max_steer_rate_deg: 최대 조향각 속도 [deg/s]
                ("wheelbase", 1.023), # - wheelbase: 휠베이스 길이 [m]
            ],
        )

        path_csv_param = self.get_parameter("path_csv").get_parameter_value().string_value
        path_csv = self._resolve_path_csv(path_csv_param)

        # Speed/curvature thresholds are needed before visualization markers are prepared.
        self.v_high = float(self.get_parameter("v_high").value)
        self.v_low = float(self.get_parameter("v_low").value)
        self.kappa_th = float(self.get_parameter("kappa_th").value)
        self.control_dt = float(self.get_parameter("control_dt").value)
        self.lookahead_points = int(self.get_parameter("lookahead_points").value)

        self.path_xy: Optional[np.ndarray] = None
        self.path_kappa: Optional[np.ndarray] = None
        self.viz_timer = None
        if path_csv:
            try:
                self.path_xy, self.path_kappa = self._load_path(path_csv)
                self.get_logger().info(f"Loaded path from {path_csv} with {len(self.path_xy)} points.")
                self._prepare_visualization_msgs()
            except Exception as exc:
                self.get_logger().error(f"Failed to initialize PathManager: {exc}")
        else:
            self.get_logger().warn("No path CSV could be resolved. Set parameter 'path_csv' to begin.")

        self.speed_pid = SpeedPID(
            kp=float(self.get_parameter("speed_kp").value),
            ki=float(self.get_parameter("speed_ki").value),
            kd=float(self.get_parameter("speed_kd").value),
        )

        self.mpc = SteeringMPC(
            Np=int(self.get_parameter("mpc_Np").value),
            Nc=int(self.get_parameter("mpc_Nc").value),
            dt=float(self.get_parameter("control_dt").value),
            max_steer=math.radians(float(self.get_parameter("max_steer_deg").value)),
            max_steer_rate=math.radians(float(self.get_parameter("max_steer_rate_deg").value)),
            wheelbase=float(self.get_parameter("wheelbase").value),
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
        segment_indices = np.arange(idx, idx + self.lookahead_points) % len(self.path_xy)
        path_segment = self.path_xy[segment_indices]
        kappa_segment = self.path_kappa[segment_indices]

        if len(path_segment) < 2:
            self.get_logger().warn("Path segment too short for control.")
            return

        k_local = float(np.max(np.abs(kappa_segment))) if len(kappa_segment) > 0 else 0.0
        v_ref = self.v_low if k_local > self.kappa_th else self.v_high

        throttle, brake = self.speed_pid.step(v_ref, v_meas, dt)

        e_y, e_psi = self._compute_errors(path_segment, x, y, theta)
        mpc_state = np.array([e_y, e_psi, delta_meas, v_meas], dtype=float)
        steer_cmd = self.mpc.solve(mpc_state, path_segment, v_ref)

        # Log target speed and steering command for each control cycle.
        print(f"[CONTROL] v_ref={v_ref:.2f} m/s, steer_cmd={steer_cmd:.3f} rad")

        msg = Vector3Stamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "Team8"
        msg.vector.x = throttle
        msg.vector.y = steer_cmd
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
        elif self.path_manager is None:
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
        try:
            share_dir = get_package_share_directory("racingproject")
            candidates.append(Path(share_dir) / "data" / "waypoints.csv")
        except (PackageNotFoundError, Exception):
            pass

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
    node = RacingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
