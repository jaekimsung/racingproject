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

    def _path_yaw_and_curvature(self, path_segment_xy: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute reference yaw and curvature sequences from a local path segment."""
        if path_segment_xy.shape[0] < 2:
            # Degenerate path; zero references.
            return np.zeros(horizon), np.zeros(horizon)

        diffs = np.diff(path_segment_xy, axis=0)
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        # Duplicate last heading to match point count.
        headings = np.concatenate([headings, headings[-1:]])

        # Arc length between points.
        ds = np.hypot(diffs[:, 0], diffs[:, 1])
        ds = np.maximum(ds, 1e-3)
        # Approximate curvature as heading change per distance.
        dpsi = np.diff(headings, prepend=headings[0])
        curvature = dpsi / np.concatenate([ds, ds[-1:]])

        # Repeat or trim to horizon length.
        yaw_ref = np.interp(
            np.linspace(0, len(headings) - 1, num=horizon, endpoint=False),
            np.arange(len(headings)),
            headings,
        )
        kappa_ref = np.interp(
            np.linspace(0, len(curvature) - 1, num=horizon, endpoint=False),
            np.arange(len(curvature)),
            curvature,
        )
        return yaw_ref, kappa_ref

    def solve(self, state: np.ndarray, path_segment_xy: np.ndarray, v_ref: float) -> float:
        """
        Solve the MPC problem for the next steering command.

        Args:
            state: [e_y, e_psi, delta, v(optional)].
            path_segment_xy: local path points for reference yaw/curvature.
            v_ref: nominal speed for linearization.

        Returns:
            Steering command delta [rad] clamped to [-max_steer, max_steer].
        """
        if cp is None:
            raise ImportError("cvxpy is required for SteeringMPC.")

        # Choose linearization speed
        if state.shape[0] >= 4:
            v_nom = float(state[3])
        else:
            v_nom = float(v_ref)
        v_nom = max(0.1, abs(v_nom))

        # MPC state: [e_y, e_psi, delta]
        x0 = np.array(state[:3]).reshape(-1, 1)
        A, B = self._linear_dynamics(v_nom)

        # Decision variables
        x = cp.Variable((3, self.Np + 1))
        u = cp.Variable((1, self.Nc))

        # Build reference sequences from path curvature.
        # We only use curvature for a feedforward steering angle, not as a state reference.
        _, kappa_ref = self._path_yaw_and_curvature(path_segment_xy, self.Np)
        delta_ref = np.clip(
            np.arctan(self.wheelbase * kappa_ref),
            -self.max_steer,
            self.max_steer,
        )

        cost = 0
        constraints = [x[:, [0]] == x0]

        for k in range(self.Np):
            # Clamp index when horizon is longer than reference arrays
            idx = min(k, len(delta_ref) - 1)

            if k < self.Nc:
                # Reference state:
                #   e_y   -> 0
                #   e_psi -> 0
                #   delta -> delta_ref
                x_ref_k = cp.Constant(
                    np.array(
                        [
                            0.0,                  # e_y reference
                            0.0,                  # e_psi reference
                            float(delta_ref[idx]) # steering feedforward
                        ]
                    )
                )

                # Stage cost with input penalty
                cost += cp.quad_form(x[:, k] - x_ref_k, self.Q) + cp.quad_form(u[:, k], self.R)

                # Input rate and magnitude constraints
                constraints.append(cp.abs(u[:, k]) <= self.max_steer_rate)
                constraints.append(cp.abs(x[2, k] + self.dt * u[:, k]) <= self.max_steer)

                # Dynamics
                constraints.append(x[:, [k + 1]] == A @ x[:, [k]] + B @ u[:, [k]])
            else:
                # After Nc, we only propagate the dynamics and penalize state deviation
                x_ref_k = cp.Constant(
                    np.array(
                        [
                            0.0,
                            0.0,
                            float(delta_ref[idx]),
                        ]
                    )
                )
                cost += cp.quad_form(x[:, k] - x_ref_k, self.Q)
                constraints.append(x[:, [k + 1]] == A @ x[:, [k]])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, max_iter=5000)
        except Exception:
            # On solver error, fall back to current steering angle
            return float(np.clip(x0[2, 0], -self.max_steer, self.max_steer))

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # If the problem is infeasible or unbounded, also fall back
            return float(np.clip(x0[2, 0], -self.max_steer, self.max_steer))

        # Use the first steering rate command
        steer_rate_cmd = u.value[0, 0] if u.value is not None else 0.0
        delta_next = x0[2, 0] + self.dt * steer_rate_cmd
        return float(np.clip(delta_next, -self.max_steer, self.max_steer))
