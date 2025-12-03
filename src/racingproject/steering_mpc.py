"""Linear MPC for lateral control with DIRECT STEERING (No Rate Limits)."""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except Exception:
    cp = None


class SteeringMPC:
    """
    Solve a finite-horizon steering problem minimizing lateral/heading error.
    
    MODEL: Direct Steering Control (Infinite Steering Rate)
    State: x = [e_y, e_psi]
    Input: u = delta (steering angle)
    """

    def __init__(
        self,
        Np: int,
        Nc: int,
        dt: float,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        max_steer: float = np.deg2rad(20.0),
        wheelbase: float = 2.7,
    ):
        if cp is None:
            raise ImportError("cvxpy is required for SteeringMPC.")

        self.Np = Np
        self.Nc = Nc
        self.dt = dt
        self.max_steer = max_steer
        self.wheelbase = wheelbase

        # State cost: [e_y, e_psi]
        self.Q = Q if Q is not None else np.diag([0.7, 0.5])
        # Input cost: penalize deviation from curvature (steering effort)
        # R 값이 작을수록 핸들을 더 과격하게 꺾을 수 있습니다.
        self.R = R if R is not None else np.diag([4.0]) 

    def _linear_dynamics(self, v: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Build discrete-time linearized lateral dynamics matrices (2-State).
        x_{k+1} = A * x_k + B * u_k
        where x = [e_y, e_psi]^T, u = [delta]
        """
        # A matrix (2x2)
        # e_y_next   = e_y + v * dt * e_psi
        # e_psi_next = e_psi
        A = np.eye(2)
        A[0, 1] = self.dt * v

        # B matrix (2x1)
        # e_psi_next += (-v * dt / L) * delta
        # (Assuming sign convention: positive delta reduces heading error)
        B = np.zeros((2, 1))
        B[1, 0] = -self.dt * v / max(self.wheelbase, 1e-3)
        
        return A, B

    def _path_curvature(self, path_segment_xy: np.ndarray, horizon: int) -> np.ndarray:
        """Compute curvature sequence from a local path segment for feedforward."""
        if path_segment_xy.shape[0] < 2:
            return np.zeros(horizon)

        diffs = np.diff(path_segment_xy, axis=0)
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        ds = np.hypot(diffs[:, 0], diffs[:, 1])
        ds = np.maximum(ds, 1e-3)
        
        dpsi = np.diff(headings, prepend=headings[0])
        # Wrap angle just in case
        dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi
        
        curvature = dpsi / ds

        # Interpolate to horizon length
        kappa_ref = np.interp(
            np.linspace(0, len(curvature) - 1, num=horizon, endpoint=False),
            np.arange(len(curvature)),
            curvature,
        )
        return kappa_ref

    def solve(self, state: np.ndarray, path_segment_xy: np.ndarray, v_ref: float) -> float:
        """
        Solve the MPC problem using Direct Steering (No Rate Limit).

        Args:
            state: [e_y, e_psi, delta_meas, v_meas] 
                   (Note: delta_meas is IGNORED in this mode)
            path_segment_xy: local path points
            v_ref: nominal speed

        Returns:
            Steering command delta [rad]
        """
        if cp is None:
            raise ImportError("cvxpy is required for SteeringMPC.")

        # Extract only relevant states: [e_y, e_psi]
        x0 = np.array(state[:2]).reshape(-1, 1)
        
        # Velocity for linearization
        if state.shape[0] >= 4:
            v_nom = float(state[3])
        else:
            v_nom = float(v_ref)
        v_nom = max(0.1, abs(v_nom))

        A, B = self._linear_dynamics(v_nom)

        # Decision variables
        # x: State [e_y, e_psi] over horizon
        # u: Input [delta] over control horizon (Direct Steering Angle)
        x = cp.Variable((2, self.Np + 1))
        u = cp.Variable((1, self.Nc))

        # Feedforward steering from curvature
        kappa_ref = self._path_curvature(path_segment_xy, self.Np)
        delta_ff = np.clip(
            np.arctan(self.wheelbase * kappa_ref),
            -self.max_steer,
            self.max_steer,
        )

        cost = 0
        constraints = [x[:, [0]] == x0]

        for k in range(self.Np):
            idx = min(k, len(delta_ff) - 1)
            
            # Reference State (Always 0 error)
            x_ref = np.zeros((2, 1))

            if k < self.Nc:
                # Reference Input (Curvature feedforward)
                u_ref = delta_ff[idx]
                
                # Cost: Minimize error + Minimize deviation from feedforward steering
                cost += cp.quad_form(x[:, k] - x_ref[:, 0], self.Q)
                cost += cp.quad_form(u[:, k] - u_ref, self.R)

                # Constraints: Absolute Steering Limit ONLY (No Rate Limit)
                constraints.append(cp.abs(u[:, k]) <= self.max_steer)

                # Dynamics update
                constraints.append(x[:, [k + 1]] == A @ x[:, [k]] + B @ u[:, [k]])
            else:
                # Prediction beyond control horizon
                cost += cp.quad_form(x[:, k] - x_ref[:, 0], self.Q)
                constraints.append(x[:, [k + 1]] == A @ x[:, [k]]) # Zero order hold assumption or just coast

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, max_iter=5000)
        except Exception:
            # Fallback: simple P-control if solver fails
            return float(np.clip(-0.5 * x0[1, 0], -self.max_steer, self.max_steer))

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
             return float(np.clip(-0.5 * x0[1, 0], -self.max_steer, self.max_steer))

        # Direct result is the steering angle
        delta_cmd = u.value[0, 0] if u.value is not None else 0.0
        return float(np.clip(delta_cmd, -self.max_steer, self.max_steer))