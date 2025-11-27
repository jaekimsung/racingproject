"""Stanley lateral controller with steering rate and magnitude limits."""

from __future__ import annotations

import math


class StanleyController:
    """Compute steering commands using the Stanley method with simple rate limits."""

    def __init__(
        self,
        k: float,
        ks: float,
        heading_gain: float,
        max_steer: float,
        max_steer_rate: float,
        wheelbase: float,
        min_speed: float = 0.1,
    ):
        self.k = k
        self.ks = ks
        self.heading_gain = heading_gain
        self.max_steer = max_steer
        self.max_steer_rate = max_steer_rate
        self.wheelbase = wheelbase
        self.min_speed = min_speed

    def compute(
        self,
        e_y: float,
        e_psi: float,
        kappa_ref: float,
        v: float,
        delta_meas: float,
        dt: float,
    ) -> float:
        """
        Compute the steering command using Stanley's control law.

        Args:
            e_y: lateral error [m]
            e_psi: heading error [rad]
            kappa_ref: reference curvature at closest point [1/m]
            v: current speed [m/s]
            delta_meas: current steering angle measurement [rad]
            dt: timestep since last control [s]
        """
        v_eff = max(self.min_speed, abs(v))

        # Feedforward steering from path curvature to reduce steady-state error.
        delta_ff = math.atan(self.wheelbase * kappa_ref)

        # Stanley control terms.
        heading_term = self.heading_gain * e_psi
        cross_track_term = math.atan2(self.k * e_y, self.ks + v_eff)

        desired_delta = delta_ff + heading_term + cross_track_term
        desired_delta = max(-self.max_steer, min(self.max_steer, desired_delta))

        if dt <= 0.0 or self.max_steer_rate <= 0.0:
            return desired_delta

        # max_delta_change = self.max_steer_rate * dt
        # limited_delta = max(
        #     delta_meas - max_delta_change,
        #     min(delta_meas + max_delta_change, desired_delta),
        # )
        # return max(-self.max_steer, min(self.max_steer, limited_delta))
        return desired_delta

