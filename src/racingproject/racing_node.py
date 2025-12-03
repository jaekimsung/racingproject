"""ROS2 node with Data Logging and Continuous Lap Timing."""

from __future__ import annotations

import math
import csv
import datetime
import os
import sys
from typing import Optional, Tuple, List

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
        self.declare_parameters(
            "",
            [
                ("path_csv", ""), 
                ("lookahead_points", 30), 
                ("braking_distance", 20.0), 
                ("speed_kp", 0.5), 
                ("speed_ki", 0.1),
                ("speed_kd", 0.01),
                ("v_high", 15.0), 
                ("v_low", 8.0),
                ("kappa_th", 0.05), 
                ("mpc_Np", 10), 
                ("mpc_Nc", 5), 
                ("control_dt", 0.05), 
                ("max_steer_deg", 20.0), 
                ("wheelbase", 1.023),
                ("slow_x_min", 0.0),
                ("slow_x_max", 0.0),
                ("slow_y_min", 0.0),
                ("slow_y_max", 0.0),
            ],
        )

        path_csv_param = self.get_parameter("path_csv").get_parameter_value().string_value
        path_csv = self._resolve_path_csv(path_csv_param)

        self.v_high = float(self.get_parameter("v_high").value)
        self.v_low = float(self.get_parameter("v_low").value)
        self.kappa_th = float(self.get_parameter("kappa_th").value)
        self.control_dt = float(self.get_parameter("control_dt").value)
        self.lookahead_points = int(self.get_parameter("lookahead_points").value)
        self.braking_distance = float(self.get_parameter("braking_distance").value)
        self.max_steer = math.radians(float(self.get_parameter("max_steer_deg").value))
        self.wheelbase = float(self.get_parameter("wheelbase").value)

        self.slow_x_min = float(self.get_parameter("slow_x_min").value)
        self.slow_x_max = float(self.get_parameter("slow_x_max").value)
        self.slow_y_min = float(self.get_parameter("slow_y_min").value)
        self.slow_y_max = float(self.get_parameter("slow_y_max").value)

        # --- [추가] 랩타임 및 로깅용 변수 ---
        self.log_data: List[List[float]] = [] 
        self.session_start_time = datetime.datetime.now() # 파일명용 절대 시간
        self.lap_start_ros_time = None  # 현재 랩 시작 시간 (ROS Time)
        
        self.max_progress_idx = 0   # 이번 랩에서 도달한 최대 인덱스
        self.lap_count = 1          # 현재 랩 카운트
        self.total_waypoints = 0    # 전체 경로 점 개수

        self.path_xy: Optional[np.ndarray] = None
        self.path_kappa: Optional[np.ndarray] = None
        self.path_step: float = 1.0
        self.viz_timer = None
        if path_csv:
            try:
                self.path_xy, self.path_kappa = self._load_path(path_csv)
                self.path_step = self._compute_path_step(self.path_xy)
                self.total_waypoints = len(self.path_xy) # 전체 웨이포인트 개수 저장
                self.get_logger().info(f"Loaded path from {path_csv} with {len(self.path_xy)} points.")
                self._prepare_visualization_msgs()
            except Exception as exc:
                self.get_logger().error(f"Failed to load path: {exc}")
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
            max_steer=self.max_steer,
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

        if self.path_xy is not None:
            self._publish_visualization()
            self.viz_timer = self.create_timer(0.5, self._publish_visualization)

        self.current_state: Optional[Tuple[float, float, float, float, float]] = None
        self.last_time = self.get_clock().now()
        self.racing_path_msg: Optional[PathMsg] = None
        self.decel_markers: Optional[MarkerArray] = None
        self.pose_path_msg = PathMsg()

    def state_callback(self, msg: Float32MultiArray) -> None:
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
        if self.current_state is None or self.path_xy is None:
            return

        now = self.get_clock().now()
        # 첫 실행 시 랩 타이머 시작
        if self.lap_start_ros_time is None: 
            self.lap_start_ros_time = now

        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 1e-5:
            return
        self.last_time = now

        x, y, theta, v_meas, delta_meas = self.current_state
        if self.path_xy is None or len(self.path_xy) == 0:
            return

        # --- [1] 현재 위치의 경로 인덱스 찾기 ---
        idx = self._find_closest_index(x, y)
        
        # --- [2] 랩 완주 판단 및 처리 로직 ---
        self.max_progress_idx = max(self.max_progress_idx, idx)
        
        # 조건: 전체 경로의 95% 이상을 갔다가 다시 5% 이내(시작점)로 돌아옴
        threshold_finish = int(self.total_waypoints * 0.95)
        threshold_start = int(self.total_waypoints * 0.05)
        
        if self.max_progress_idx > threshold_finish and idx < threshold_start:
            # 1. 랩타임 계산
            lap_time = (now - self.lap_start_ros_time).nanoseconds * 1e-9
            
            print("\n" + "="*50)
            print(f"🏁 LAP {self.lap_count} FINISHED! Time: {lap_time:.3f} sec")
            print("="*50 + "\n")
            
            # 2. 현재 랩의 로그 저장
            self.save_log(suffix=f"_lap{self.lap_count}")
            
            # 3. 다음 랩을 위한 초기화 (차는 멈추지 않음)
            self.log_data = []           # 데이터 비우기
            self.max_progress_idx = 0    # 진행률 초기화
            self.lap_start_ros_time = now # 타이머 리셋
            self.lap_count += 1          # 랩 카운트 증가

        # --- [3] 제어 로직 ---
        step = max(self.path_step, 1e-6)
        num_speed_points = max(self.lookahead_points, int(self.braking_distance / step))
        num_speed_points = max(2, num_speed_points)
        kappa_indices = (np.arange(num_speed_points) + idx) % len(self.path_kappa)
        kappa_segment = self.path_kappa[kappa_indices]
        k_local = float(np.max(np.abs(kappa_segment))) if len(kappa_segment) > 0 else 0.0
        
        is_curved = k_local > self.kappa_th
        in_slow_zone = (self.slow_x_min <= x <= self.slow_x_max) and \
                       (self.slow_y_min <= y <= self.slow_y_max)

        v_ref = self.v_low if (is_curved or in_slow_zone) else self.v_high
        throttle, brake = self.speed_pid.step(v_ref, v_meas, dt)

        T_horizon = self.mpc.Np * self.mpc.dt
        mpc_dist = max(v_meas, 2.0) * T_horizon
        num_mpc_points = int(mpc_dist / step)
        num_mpc_points = max(5, num_mpc_points)
        num_mpc_points = min(len(self.path_xy), num_mpc_points)

        mpc_indices = (np.arange(num_mpc_points) + idx) % len(self.path_xy)
        path_segment = self.path_xy[mpc_indices]
        if len(path_segment) < 2:
            self.get_logger().warn("Path segment too short for control.")
            return

        e_y, e_psi = self._compute_errors(path_segment, x, y, theta)
        mpc_state = np.array([e_y, e_psi, delta_meas, v_meas], dtype=float)
        steer_cmd = - self.mpc.solve(mpc_state, path_segment, v_ref)
        steer_norm = float(np.clip(steer_cmd / max(self.max_steer, 1e-6), -1.0, 1.0))

        # --- [4] 데이터 로깅 (현재 랩 시간 기준) ---
        current_lap_time = (now - self.lap_start_ros_time).nanoseconds * 1e-9
        # [Time, X, Y, V_ref, V_meas, Ey, Epsi, Steer(rad), Throttle, Brake]
        self.log_data.append([current_lap_time, x, y, v_ref, v_meas, e_y, e_psi, steer_cmd, throttle, brake])

        steer_deg = math.degrees(steer_cmd)
        print(
            f"[Lap {self.lap_count}] t={current_lap_time:.1f}s, v={v_meas:.2f}/{v_ref:.2f}, "
            f"str={steer_deg:.1f}deg, e_y={e_y:.3f}m"
        )

        msg = Vector3Stamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "Team8"
        msg.vector.x = throttle
        msg.vector.y = steer_norm
        msg.vector.z = brake
        self.cmd_pub.publish(msg)

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
        if self.racing_path_msg is not None:
            now = self.get_clock().now()
            self.racing_path_msg.header.stamp = now.to_msg()
            self.path_pub.publish(self.racing_path_msg)
        if self.decel_markers is not None:
            now = self.get_clock().now()
            for marker in self.decel_markers.markers:
                marker.header.stamp = now.to_msg()
            self.decel_marker_pub.publish(self.decel_markers)

    def _compute_errors(self, path_segment: np.ndarray, x: float, y: float, theta: float) -> Tuple[float, float]:
        target = path_segment[0] 
        dx = target[0] - x
        dy = target[1] - y
        e_y = -math.sin(theta) * dx + math.cos(theta) * dy
        if len(path_segment) >= 2:
            next_pt = path_segment[1]
            path_yaw = math.atan2(next_pt[1] - target[1], next_pt[0] - target[0])
        else:
            path_yaw = math.atan2(dy, dx)
        e_psi = wrap_angle(path_yaw - theta)
        return e_y, e_psi

    def _resolve_path_csv(self, param_value: str) -> str:
        if param_value: return param_value
        candidates = []
        try:
            share_dir = get_package_share_directory("racingproject")
            candidates.append(Path(share_dir) / "data" / "optimal.csv")
            candidates.append(Path(share_dir) / "data" / "waypoints.csv")
        except (PackageNotFoundError, Exception):
            pass
        base = Path(__file__).resolve().parent / "data"
        candidates.append(base / "optimal.csv")
        candidates.append(base / "waypoints.csv")
        for candidate in candidates:
            if candidate.exists():
                self.get_logger().info(f"Using packaged path CSV at {candidate}")
                return str(candidate)
        return ""

    def _load_path(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            data = np.loadtxt(csv_path, delimiter=",")
        except ValueError:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        data = np.atleast_2d(data)
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
        if len(xy) < 3: return np.zeros(len(xy))
        dx = np.gradient(xy[:, 0], s)
        dy = np.gradient(xy[:, 1], s)
        ddx = np.gradient(dx, s)
        ddy = np.gradient(dy, s)
        return (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-6)

    @staticmethod
    def _compute_path_step(xy: np.ndarray) -> float:
        if len(xy) < 2: return 1.0
        diffs = np.diff(xy, axis=0)
        spacing = np.hypot(diffs[:, 0], diffs[:, 1])
        return float(np.mean(spacing))

    def _find_closest_index(self, x: float, y: float) -> int:
        if self.path_xy is None: return 0
        delta = self.path_xy - np.array([[x, y]])
        dists = np.einsum("ij,ij->i", delta, delta)
        return int(np.argmin(dists))

    def _prepare_visualization_msgs(self) -> None:
        if self.path_xy is None or self.path_kappa is None: return
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
        marker.scale.x = 0.5; marker.scale.y = 0.5; marker.scale.z = 0.1
        marker.color.r = 1.0; marker.color.g = 0.4; marker.color.b = 0.0; marker.color.a = 0.8
        marker.points = [self._point(px, py) for px, py in decel_points]
        self.decel_markers = MarkerArray(markers=[marker])

    @staticmethod
    def _point(x: float, y: float): return Point(x=x, y=y, z=0.0)
    
    @staticmethod
    def _path_headings(xy: np.ndarray) -> np.ndarray:
        if len(xy) < 2: return np.zeros(len(xy))
        diffs = np.diff(xy, axis=0, append=xy[-1:])
        return np.arctan2(diffs[:, 1], diffs[:, 0])

    @staticmethod
    def _pose_stamped(x: float, y: float, yaw: float, stamp) -> PoseStamped:
        qz = math.sin(yaw * 0.5); qw = math.cos(yaw * 0.5)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp.to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = x; pose_msg.pose.position.y = y; pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        return pose_msg

    def save_log(self, suffix=""):
        if not self.log_data: return
        # 파일명: racing_log_YYYYMMDD_HHMMSS_lapN.csv
        filename = f"racing_log_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}{suffix}.csv"
        path = os.path.join(os.getcwd(), filename)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 헤더: 시간, 위치, 속도, 오차, 제어값
            writer.writerow(["Time", "X", "Y", "V_ref", "V_meas", "Ey", "Epsi", "Steer", "Throttle", "Brake"])
            writer.writerows(self.log_data)
        self.get_logger().info(f"✅ Log saved: {filename}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RacingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 종료 시 주행 중이던 랩의 데이터도 저장
        node.save_log(suffix=f"_lap{node.lap_count}_incomplete")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()