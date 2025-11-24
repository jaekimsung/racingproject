"""Simple PID controller for longitudinal speed regulation."""

from __future__ import annotations

from typing import Tuple


class SpeedPID:
    """PID controller producing throttle and brake commands from speed error."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        throttle_max: float = 1.0,
        brake_max: float = 1.0,
        integral_limit: float = 10.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.throttle_max = throttle_max
        self.brake_max = brake_max
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        """Reset internal PID state."""
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, v_ref: float, v_meas: float, dt: float) -> Tuple[float, float]:
        """
        Compute throttle and brake commands based on current speed error.

        Returns:
            throttle, brake (both in [0, 1]).
        """
        if dt <= 0.0:
            return 0.0, 0.0

        error = v_ref - v_meas
        self.integral += error * dt
        # Basic anti-windup
        self.integral = max(-self.integral_limit, min(self.integral, self.integral_limit))

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative

        throttle = max(0.0, min(self.throttle_max, u))
        brake = max(0.0, min(self.brake_max, -u))
        return throttle, brake
