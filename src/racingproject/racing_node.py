"""ROS2 node wiring together path management, speed PID, and steering MPC."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .path_manager import PathManager, PathParams
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
                ("sample_ds", 0.5),
                ("max_offset", 3.0),
                ("offset_gain", 1.0),
                ("offset_power", 1.0),
                ("curvature_smooth_window", 11),
                ("lookahead_points", 30),
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
                ("max_steer_rate_deg", 45.0),
                ("wheelbase", 2.7),
            ],
        )

        path_csv = self.get_parameter("path_csv").get_parameter_value().string_value
        if not path_csv:
            self.get_logger().warn("Parameter 'path_csv' is empty. Node will not run without a path.")

        path_params = PathParams(
            sample_ds=float(self.get_parameter("sample_ds").value),
            max_offset=float(self.get_parameter("max_offset").value),
            offset_gain=float(self.get_parameter("offset_gain").value),
            offset_power=float(self.get_parameter("offset_power").value),
            curvature_smooth_window=int(self.get_parameter("curvature_smooth_window").value),
            lookahead_points=int(self.get_parameter("lookahead_points").value),
        )

        self.path_manager: Optional[PathManager] = None
        try:
            self.path_manager = PathManager(path_csv, path_params)
            self.get_logger().info(f"Loaded path from {path_csv} with {len(self.path_manager.racing_xy)} points.")
        except Exception as exc:
            self.get_logger().error(f"Failed to initialize PathManager: {exc}")

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

        self.v_high = float(self.get_parameter("v_high").value)
        self.v_low = float(self.get_parameter("v_low").value)
        self.kappa_th = float(self.get_parameter("kappa_th").value)
        self.control_dt = float(self.get_parameter("control_dt").value)

        self.state_sub = self.create_subscription(
            Float32MultiArray,
            "/mobile_system_control/ego_vehicle",
            self.state_callback,
            10,
        )
        self.cmd_pub = self.create_publisher(Vector3Stamped, "/mobile_system_control/control_msg", 10)
        self.timer = self.create_timer(self.control_dt, self.control_loop)

        self.current_state: Optional[Tuple[float, float, float, float, float]] = None
        self.last_time = self.get_clock().now()

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
        if self.current_state is None or self.path_manager is None:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 1e-5:
            return
        self.last_time = now

        x, y, theta, v_meas, delta_meas = self.current_state
        idx = self.path_manager.find_closest_index(x, y)
        path_segment, kappa_segment = self.path_manager.get_local_segment(idx)

        if len(path_segment) < 2:
            self.get_logger().warn("Path segment too short for control.")
            return

        k_local = float(np.max(np.abs(kappa_segment))) if len(kappa_segment) > 0 else 0.0
        v_ref = self.v_low if k_local > self.kappa_th else self.v_high

        throttle, brake = self.speed_pid.step(v_ref, v_meas, dt)

        e_y, e_psi = self._compute_errors(path_segment, x, y, theta)
        mpc_state = np.array([e_y, e_psi, delta_meas, v_meas], dtype=float)
        steer_cmd = self.mpc.solve(mpc_state, path_segment, v_ref)

        msg = Vector3Stamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "Chadol"
        msg.vector.x = throttle
        msg.vector.y = steer_cmd
        msg.vector.z = brake
        self.cmd_pub.publish(msg)

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
