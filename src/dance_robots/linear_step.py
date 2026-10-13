"""Straight-line dance moves (forward / backward) using a time-stretched S-curve.

Approach ("Option A" from the design discussion):
  1. Use the ONE normalized 7-segment jerk-limited profile from scurve.py (unit
     time u = t/T, unit distance). Its shape (segment proportions) is fixed.
  2. Scale it to the real move:  x(t) = D * s(t/T)
         velocity  v(t) = (D/T)   * s'(u)
         accel     a(t) = (D/T^2) * s''(u)
         jerk      j(t) = (D/T^3) * s'''(u)
  3. The robot limits become three simple feasibility checks on the peak values.

There is no solver and no CSV: everything is closed-form and evaluated online.
Pure Python (no ROS, no numpy). Compatible with Python 3.10.
"""

from __future__ import annotations

import math

from .dance_step import DanceStep, RobotLimits, Velocity
from .scurve import ACCEL_FRACTION, CRUISE_FRACTION, UnitSCurve

# Default displacement of the forward / backward dance step (handover: 30 cm).
FORWARD_DISTANCE = 0.3  # [m]
# Default displacement of the backward step (handover: same 30 cm, opposite way).
BACKWARD_DISTANCE = 0.3  # [m] (magnitude; the sign is applied by BackwardStep)


# ----------------------------------------------------------------------------
# Dance steps
# ----------------------------------------------------------------------------
class LinearStep(DanceStep):
    """Straight-line move of a SIGNED distance (positive = forward).

    Only linear.x is used; angular.z is always 0.
    """

    def __init__(
        self,
        duration: float,
        distance: float,
        limits: RobotLimits | None = None,
        accel_fraction: float = ACCEL_FRACTION,
        cruise_fraction: float = CRUISE_FRACTION,
    ) -> None:
        super().__init__(duration, limits)  # validates duration, stores limits

        # A zero (or NaN/inf) distance is not a move; fail loudly.
        if not math.isfinite(distance) or distance == 0.0:
            raise ValueError(f"distance must be finite and non-zero, got {distance!r}")

        self._distance = float(distance)
        self._profile = UnitSCurve(accel_fraction, cruise_fraction)

        # Must come LAST: min_duration() needs _distance and _profile.
        self._check_feasibility()

    # -- Peak values at the actual duration (handy for tests, plots, logging) --
    @property
    def distance(self) -> float:
        """Signed displacement of the move [m]."""
        return self._distance

    @property
    def peak_velocity(self) -> float:
        """Peak |velocity| [m/s]: unit peak scaled by D/T."""
        return self._profile.v_peak * abs(self._distance) / self._duration

    @property
    def peak_acceleration(self) -> float:
        """Peak |acceleration| [m/s^2]: unit peak scaled by D/T^2."""
        return self._profile.a_peak * abs(self._distance) / self._duration**2

    @property
    def peak_jerk(self) -> float:
        """Peak |jerk| [m/s^3]: unit peak scaled by D/T^3."""
        return self._profile.j_peak * abs(self._distance) / self._duration**3

    def min_duration(self) -> float:
        """Smallest T that keeps v, a and j under the limits.

        From peak_v = V*D/T, peak_a = A*D/T^2, peak_j = J*D/T^3 we solve each
        one for T at its limit. The move needs the LARGEST of the three.
        """
        d = abs(self._distance)
        p, lim = self._profile, self._limits
        t_from_velocity = p.v_peak * d / lim.v_max
        t_from_accel = math.sqrt(p.a_peak * d / lim.a_max)
        t_from_jerk = (p.j_peak * d / lim.j_max) ** (1.0 / 3.0)
        return max(t_from_velocity, t_from_accel, t_from_jerk)

    # -- Kinematics ----------------------------------------------------------
    def get_kinematics(self, t: float) -> tuple[float, float, float, float]:
        """(position, velocity, acceleration, jerk) at local time t, signed.

        Not needed by the ROS node (it only uses get_velocity); it exists so
        the offline tests and plots can check position, accel and jerk too.
        """
        x_u, v_u, a_u, j_u = self._profile.state(t / self._duration)
        d, T = self._distance, self._duration
        # Undo the normalization: each derivative picks up another 1/T.
        return (d * x_u, d / T * v_u, d / T**2 * a_u, d / T**3 * j_u)

    def _compute_velocity(self, t: float) -> Velocity:
        """Velocity command for 0 <= t < T (boundaries handled by the base class)."""
        _, v, _, _ = self.get_kinematics(t)
        return (v, 0.0)  # (linear.x, angular.z)


class ForwardStep(LinearStep):
    """Move forward by `distance` metres (default 0.3 m) in `duration` seconds."""

    def __init__(
        self,
        duration: float,
        distance: float = FORWARD_DISTANCE,
        limits: RobotLimits | None = None,
    ) -> None:
        # "Forward" must be positive; the backward step will use a negative sign.
        if not distance > 0.0:
            raise ValueError(f"ForwardStep distance must be positive, got {distance!r}")
        super().__init__(duration, distance, limits)


class BackwardStep(LinearStep):
    """Move backward by `distance` metres (default 0.3 m) in `duration` seconds.

    The caller gives a POSITIVE magnitude ("back 0.3 m"); the class applies the
    minus sign. The velocity profile is the exact mirror image of ForwardStep:
    same shape, same peaks, same minimum duration, but linear.x <= 0.
    """

    def __init__(
        self,
        duration: float,
        distance: float = BACKWARD_DISTANCE,
        limits: RobotLimits | None = None,
    ) -> None:
        # Reject non-positive input so "BackwardStep(-0.3)" can't silently
        # turn into a forward move (a double negative).
        if not distance > 0.0:
            raise ValueError(
                f"BackwardStep distance must be a positive magnitude, got {distance!r}"
            )
        # LinearStep already supports signed distances: negative = backward.
        super().__init__(duration, -distance, limits)