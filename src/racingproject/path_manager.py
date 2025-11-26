"""Path management utilities for generating and querying a smoothed racing line."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    from scipy import interpolate
except Exception:  # pragma: no cover - optional dependency guard
    interpolate = None


@dataclass
class PathParams:
    """Configuration parameters for the racing line generation."""

    sample_ds: float = 0.5
    max_offset: float = 3.0
    offset_gain: float = 1.0
    offset_power: float = 1.0
    curvature_smooth_window: int = 11
    lookahead_points: int = 30


class PathManager:
    """Generate and serve a racing line directly from a CSV centerline (no resampling)."""

    def __init__(self, csv_path: str, params: PathParams):
        self.params = params

        # 1) Load centerline from CSV: expected shape (N, 2) = [x, y]
        centerline = self.load_centerline(csv_path)

        # 2) Compute cumulative arc length along the raw CSV points
        s_center = self._compute_arclength(centerline)

        # 3) Store as centerline
        self.s_center = s_center           # arc length array
        self.xy_center = centerline        # raw CSV coordinates

        # 4) For now, use the centerline directly as the racing line (no offset / optimization)
        self.s_racing = self.s_center
        self.racing_xy = self.xy_center
        self.kappa_racing = self.compute_curvature(self.racing_xy, self.s_racing)

    @staticmethod
    def _compute_arclength(xy: np.ndarray) -> np.ndarray:
        """
        Compute cumulative arc length along a polyline defined by xy points.

        Args:
            xy: np.ndarray of shape (N, 2), columns are [x, y].

        Returns:
            s: np.ndarray of shape (N,), cumulative arc length with s[0] = 0.
        """
        if xy.shape[0] < 2:
            # Degenerate path: just return zero-length
            return np.zeros(xy.shape[0], dtype=float)

        dx = np.diff(xy[:, 0])
        dy = np.diff(xy[:, 1])
        ds = np.hypot(dx, dy)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        return s

    @staticmethod
    def load_centerline(csv_path: str) -> np.ndarray:
        """
        Load a reference path from CSV. Assumes columns [x, y, ...].

        Returns:
            np.ndarray of shape (N, 2).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV path does not exist: {csv_path}")

        try:
            data = np.loadtxt(csv_path, delimiter=",")
        except ValueError:
            # Fallback when a header row (e.g., "x,y") is present.
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError("CSV must contain at least two numeric columns for x and y.")
        return data[:, :2]

    @staticmethod
    def resample_by_arclength(xy: np.ndarray, ds: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample the path to approximately uniform arc-length spacing.

        Returns:
            s_uniform: cumulative arc length samples.
            xy_uniform: resampled coordinates.
        """
        dx = np.diff(xy[:, 0])
        dy = np.diff(xy[:, 1])
        ds_raw = np.hypot(dx, dy)
        s_raw = np.concatenate([[0.0], np.cumsum(ds_raw)])

        s_end = s_raw[-1]
        s_uniform = np.arange(0.0, s_end, ds)
        x_uniform = np.interp(s_uniform, s_raw, xy[:, 0])
        y_uniform = np.interp(s_uniform, s_raw, xy[:, 1])
        xy_uniform = np.stack([x_uniform, y_uniform], axis=1)
        return s_uniform, xy_uniform

    @staticmethod
    def compute_tangent_normal(xy: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute unit tangent and normal vectors along the path."""
        dx_ds = np.gradient(xy[:, 0], s)
        dy_ds = np.gradient(xy[:, 1], s)
        norm = np.hypot(dx_ds, dy_ds) + 1e-6
        t_x = dx_ds / norm
        t_y = dy_ds / norm
        tangent = np.stack([t_x, t_y], axis=1)
        normal = np.stack([-t_y, t_x], axis=1)
        return tangent, normal

    def compute_curvature(self, xy: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Compute signed curvature along the path with optional smoothing."""
        dx_ds = np.gradient(xy[:, 0], s)
        dy_ds = np.gradient(xy[:, 1], s)
        ddx_ds = np.gradient(dx_ds, s)
        ddy_ds = np.gradient(dy_ds, s)

        cross = dx_ds * ddy_ds - dy_ds * ddx_ds
        denom = (dx_ds**2 + dy_ds**2) ** 1.5 + 1e-6
        kappa = cross / denom
        return self._moving_average(kappa, self.params.curvature_smooth_window)

    @staticmethod
    def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
        """Simple moving average with reflection padding for smooth curvature."""
        if window < 2:
            return x
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(x, pad_width=pad, mode="reflect")
        kernel = np.ones(window) / window
        smoothed = np.convolve(padded, kernel, mode="same")[pad:-pad]
        return smoothed

    def compute_lateral_offset(self, kappa: np.ndarray) -> np.ndarray:
        """
        Map curvature to lateral offset toward the track's outside.

        Positive curvature => turn left => offset to the right (negative normal).
        """
        k_abs = np.abs(kappa)
        if np.max(k_abs) < 1e-6:
            return np.zeros_like(kappa)

        k_norm = k_abs / np.max(k_abs)
        k_scaled = k_norm ** self.params.offset_power
        d_mag = self.params.max_offset * self.params.offset_gain * k_scaled
        direction = -np.sign(kappa)
        return direction * d_mag

    def spline_smooth_and_resample(self, xy: np.ndarray, ds: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a cubic B-spline to the path and resample with uniform spacing.

        Raises:
            ImportError if scipy is not available.
        """
        if interpolate is None:
            raise ImportError("scipy is required for spline smoothing.")

        dx = np.diff(xy[:, 0])
        dy = np.diff(xy[:, 1])
        ds_raw = np.hypot(dx, dy)
        s_raw = np.concatenate([[0.0], np.cumsum(ds_raw)])
        s_end = s_raw[-1]
        t = s_raw / s_end if s_end > 0 else s_raw

        tck, _ = interpolate.splprep([xy[:, 0], xy[:, 1]], s=0.0, k=3, u=t)

        s_uniform = np.arange(0.0, s_end, ds)
        t_uniform = s_uniform / s_end if s_end > 0 else s_uniform
        x_smooth, y_smooth = interpolate.splev(t_uniform, tck)
        xy_smooth = np.stack([x_smooth, y_smooth], axis=1)
        return s_uniform, xy_smooth

    def find_closest_index(self, x: float, y: float) -> int:
        """Return the index of the racing line closest to (x, y)."""
        delta = self.racing_xy - np.array([[x, y]])
        dists = np.einsum("ij,ij->i", delta, delta)
        return int(np.argmin(dists))

    def get_local_segment(self, idx: int, horizon: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a forward segment of the racing line and curvature.

        Args:
            idx: starting index on the racing line.
            horizon: number of points to include (defaults to params.lookahead_points).
        """
        if horizon is None:
            horizon = self.params.lookahead_points
        end = min(idx + horizon, len(self.racing_xy))
        return self.racing_xy[idx:end], self.kappa_racing[idx:end]
