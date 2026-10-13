"""In-place rotation dance moves (turn left / turn right).

Same time-stretched S-curve as the linear steps (see scurve.py), but applied to
the heading angle: only angular.z is used and linear.x is always 0.

    angle     theta(t) = Theta * s(t/T)
    ang. vel  w(t)     = (Theta/T)   * s'(u)
    ang. acc  alpha(t) = (Theta/T^2) * s''(u)
    ang. jerk          = (Theta/T^3) * s'''(u)

Sign convention (ROS REP-103): positive angle / angular.z = counter-clockwise =
turning LEFT. Pure Python, compatible with Python 3.10.
"""

from __future__ import annotations

import math

from .dance_step import DanceStep, RobotLimits, Velocity
from .scurve import ACCEL_FRACTION, CRUISE_FRACTION, UnitSCurve

# Default turn of the dance step (handover: 30 degrees), stored in radians.
TURN_ANGLE = math.radians(30.0)  # ~0.5236 rad


class TurnStep(DanceStep):
    """In-place rotation by a SIGNED angle in radians (positive = left / CCW)."""

    def __init__(
        self,
        duration: float,
        angle: float,
        limits: RobotLimits | None = None,
        accel_fraction: float = ACCEL_FRACTION,
        cruise_fraction: float = CRUISE_FRACTION,
    ) -> None:
        super().__init__(duration, limits)  # validates duration, stores limits

        # A zero (or NaN/inf) angle is not a move; fail loudly.
        if not math.isfinite(angle) or angle == 0.0:
            raise ValueError(f"angle must be finite and non-zero, got {angle!r}")

        self._angle = float(angle)
        self._profile = UnitSCurve(accel_fraction, cruise_fraction)

        # Must come LAST: min_duration() needs _angle and _profile.
        self._check_feasibility()

    # -- Peak magnitudes at the actual duration ------------------------------
    @property
    def angle(self) -> float:
        """Signed rotation of the move [rad]."""
        return self._angle

    @property
    def peak_angular_velocity(self) -> float:
        """Peak |w| [rad/s]: unit peak scaled by Theta/T."""
        return self._profile.v_peak * abs(self._angle) / self._duration

    @property
    def peak_angular_acceleration(self) -> float:
        """Peak |alpha| [rad/s^2]: unit peak scaled by Theta/T^2."""
        return self._profile.a_peak * abs(self._angle) / self._duration**2

    @property
    def peak_angular_jerk(self) -> float:
        """Peak |angular jerk| [rad/s^3]: unit peak scaled by Theta/T^3."""
        return self._profile.j_peak * abs(self._angle) / self._duration**3

    def min_duration(self) -> float:
        """Smallest T that keeps w, alpha and angular jerk under their limits.

        Same idea as LinearStep: solve each peak formula for T at its limit and
        keep the largest. The alpha / jerk limits are the ESTIMATES derived in
        RobotLimits from the wheel-level linear limits and the track width.
        """
        th = abs(self._angle)
        p, lim = self._profile, self._limits
        t_from_velocity = p.v_peak * th / lim.w_max
        t_from_accel = math.sqrt(p.a_peak * th / lim.alpha_max)
        t_from_jerk = (p.j_peak * th / lim.jerk_ang_max) ** (1.0 / 3.0)
        return max(t_from_velocity, t_from_accel, t_from_jerk)

    # -- Kinematics ----------------------------------------------------------
    def get_kinematics(self, t: float) -> tuple[float, float, float, float]:
        """(angle, ang. velocity, ang. acceleration, ang. jerk) at local time t.

        Used by the offline tests/plots; the node only needs get_velocity().
        """
        th_u, w_u, a_u, j_u = self._profile.state(t / self._duration)
        d, T = self._angle, self._duration
        return (d * th_u, d / T * w_u, d / T**2 * a_u, d / T**3 * j_u)

    def _compute_velocity(self, t: float) -> Velocity:
        """Command for 0 <= t < T (boundaries handled by the base class)."""
        _, w, _, _ = self.get_kinematics(t)
        return (0.0, w)  # (linear.x, angular.z)


class TurnLeftStep(TurnStep):
    """Rotate in place to the LEFT (counter-clockwise) by `angle` radians
    (default 30 degrees). Give a positive magnitude."""

    def __init__(self, duration: float, angle: float = TURN_ANGLE,
                 limits: RobotLimits | None = None) -> None:
        if not angle > 0.0:
            raise ValueError(f"TurnLeftStep angle must be positive, got {angle!r}")
        super().__init__(duration, +angle, limits)


class TurnRightStep(TurnStep):
    """Rotate in place to the RIGHT (clockwise) by `angle` radians
    (default 30 degrees). Give a positive magnitude; the class applies the sign."""

    def __init__(self, duration: float, angle: float = TURN_ANGLE,
                 limits: RobotLimits | None = None) -> None:
        if not angle > 0.0:
            raise ValueError(f"TurnRightStep angle must be positive, got {angle!r}")
        super().__init__(duration, -angle, limits)