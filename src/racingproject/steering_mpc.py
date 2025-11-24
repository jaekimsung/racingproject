"""Linear MPC for lateral control using a kinematic bicycle approximation."""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - optional dependency guard
    cp = None


class SteeringMPC:
    """Solve a finite-horizon steering problem to minimize lateral/heading error."""

    def __init__(
        self,
        Np: int,
        Nc: int,
        dt: float,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        max_steer: float = np.deg2rad(20.0),
        wheelbase: float = 2.7,
        max_steer_rate: float = np.deg2rad(45.0),
    ):
        if cp is None:
            raise ImportError("cvxpy is required for SteeringMPC.")

        self.Np = Np
        self.Nc = Nc
        self.dt = dt
        self.max_steer = max_steer
        self.max_steer_rate = max_steer_rate
        self.wheelbase = wheelbase

        self.Q = Q if Q is not None else np.diag([5.0, 2.0, 0.5])
        self.R = R if R is not None else np.diag([0.5])

    def _linear_dynamics(self, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Build discrete-time linearized lateral dynamics matrices."""
        A = np.eye(3)
        A[0, 1] = self.dt * v
        A[1, 2] = self.dt * v / max(self.wheelbase, 1e-3)

        B = np.zeros((3, 1))
        B[2, 0] = self.dt  # steering rate affects delta
        return A, B

    def solve(self, state: np.ndarray, path_segment_xy: np.ndarray, v_ref: float) -> float:
        """
        Solve the MPC problem for the next steering command.

        Args:
            state: [e_y, e_psi, delta, v(optional)].
            path_segment_xy: unused placeholder for future enhancements.
            v_ref: nominal speed for linearization.

        Returns:
            Steering command delta [rad] clamped to [-max_steer, max_steer].
        """
        if cp is None:
            raise ImportError("cvxpy is required for SteeringMPC.")

        if state.shape[0] >= 4:
            v_nom = float(state[3])
        else:
            v_nom = float(v_ref)
        v_nom = max(0.1, abs(v_nom))

        x0 = np.array(state[:3]).reshape(-1, 1)
        A, B = self._linear_dynamics(v_nom)

        x = cp.Variable((3, self.Np + 1))
        u = cp.Variable((1, self.Nc))

        cost = 0
        constraints = [x[:, [0]] == x0]
        for k in range(self.Np):
            if k < self.Nc:
                cost += cp.quad_form(x[:, k], self.Q) + cp.quad_form(u[:, k], self.R)
                constraints.append(cp.abs(u[:, k]) <= self.max_steer_rate)
                constraints.append(cp.abs(x[2, k] + self.dt * u[:, k]) <= self.max_steer)
                constraints.append(x[:, [k + 1]] == A @ x[:, [k]] + B @ u[:, [k]])
            else:
                cost += cp.quad_form(x[:, k], self.Q)
                constraints.append(x[:, [k + 1]] == A @ x[:, [k]])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, max_iter=5000)
        except Exception:
            return float(np.clip(x0[2, 0], -self.max_steer, self.max_steer))

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return float(np.clip(x0[2, 0], -self.max_steer, self.max_steer))

        steer_rate_cmd = u.value[0, 0] if u.value is not None else 0.0
        delta_next = x0[2, 0] + self.dt * steer_rate_cmd
        return float(np.clip(delta_next, -self.max_steer, self.max_steer))
