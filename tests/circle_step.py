"""Arc dance moves (circle left / circle right).

The robot drives FORWARD along a circular arc of radius R, turning through
`arc_angle`. Linear and angular velocity are synchronised so peaks and endpoints
align exactly:

    s(t) = R * |arc_angle| * S(t/T)        (distance along the arc, S = unit S-curve)
    v(t) = ds/dt                            (linear.x, always >= 0)
    w(t) = +-v(t) / R                       (angular.z, + = left / CCW)

Wheel-level limits
------------------
On an arc the outer wheel moves faster than the body:
    v_outer = v + w*b/2 = v * (1 + b/(2R))        (b = track width)
and its acceleration and jerk are scaled by the same factor. Because the limits
protect the motors and avoid slip, we hold the OUTER WHEEL to v_max, a_max and
j_max. This is more conservative than limiting only the body, and it also implies
the estimated angular acceleration / jerk limits (wheel accel >= alpha*b/2).
The firmware angular velocity limit w_max is checked separately.

Pure Python, compatible with Python 3.10.
"""

from __future__ import annotations

import math

from .dance_step import DanceStep, RobotLimits, Velocity
from .scurve import ACCEL_FRACTION, CRUISE_FRACTION, UnitSCurve

# Defaults of the circle dance step (30 degrees; R = 0.4 m, changed from the
# handover's 0.5 m to gain time buffer at 70 BPM).
CIRCLE_RADIUS = 0.4                     # [m]  (0.4 gives more time buffer than 0.5)
CIRCLE_ARC_ANGLE = math.radians(30.0)   # [rad]


class CircleStep(DanceStep):
    """Forward arc of radius `radius` through a SIGNED angle (positive = left)."""

    def __init__(
        self,
        duration: float,
        arc_angle: float,
        radius: float = CIRCLE_RADIUS,
        limits: RobotLimits | None = None,
        accel_fraction: float = ACCEL_FRACTION,
        cruise_fraction: float = CRUISE_FRACTION,
    ) -> None:
        super().__init__(duration, limits)  # validates duration, stores limits

        if not math.isfinite(arc_angle) or arc_angle == 0.0:
            raise ValueError(f"arc_angle must be finite and non-zero, got {arc_angle!r}")
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"radius must be positive and finite, got {radius!r}")

        self._arc_angle = float(arc_angle)
        self._radius = float(radius)
        # +1 = left (CCW, angular.z > 0), -1 = right (CW, angular.z < 0).
        self._sign = 1.0 if arc_angle > 0.0 else -1.0
        # Length of the arc the body travels [m].
        self._arc_length = self._radius * abs(self._arc_angle)
        self._profile = UnitSCurve(accel_fraction, cruise_fraction)

        # Must come LAST: min_duration() needs everything above.
        self._check_feasibility()

    # -- Geometry ------------------------------------------------------------
    @property
    def arc_angle(self) -> float:
        """Signed angle swept by the heading [rad] (positive = left)."""
        return self._arc_angle

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def arc_length(self) -> float:
        """Distance travelled by the body centre along the arc [m]."""
        return self._arc_length

    @property
    def wheel_factor(self) -> float:
        """Outer-wheel speed / body speed = 1 + b/(2R). Also scales a and jerk."""
        return 1.0 + self._limits.track_width / (2.0 * self._radius)

    # -- Peak magnitudes at the actual duration --------------------------------
    @property
    def peak_velocity(self) -> float:
        """Peak body linear velocity [m/s]."""
        return self._profile.v_peak * self._arc_length / self._duration

    @property
    def peak_angular_velocity(self) -> float:
        """Peak |angular.z| [rad/s] = peak_velocity / R."""
        return self.peak_velocity / self._radius

    @property
    def peak_acceleration(self) -> float:
        """Peak body linear acceleration [m/s^2]."""
        return self._profile.a_peak * self._arc_length / self._duration**2

    @property
    def peak_jerk(self) -> float:
        """Peak body linear jerk [m/s^3]."""
        return self._profile.j_peak * self._arc_length / self._duration**3

    @property
    def peak_outer_wheel_speed(self) -> float:
        return self.peak_velocity * self.wheel_factor

    @property
    def peak_outer_wheel_acceleration(self) -> float:
        return self.peak_acceleration * self.wheel_factor

    def min_duration(self) -> float:
        """Smallest T keeping the OUTER WHEEL and the body heading in limits."""
        s, th = self._arc_length, abs(self._arc_angle)
        p, lim, f = self._profile, self._limits, self.wheel_factor
        t_from_wheel_speed = p.v_peak * s * f / lim.v_max
        t_from_wheel_accel = math.sqrt(p.a_peak * s * f / lim.a_max)
        t_from_wheel_jerk = (p.j_peak * s * f / lim.j_max) ** (1.0 / 3.0)
        # Firmware cap on angular velocity: w_peak = v_peak_unit * theta / T.
        t_from_ang_velocity = p.v_peak * th / lim.w_max
        return max(t_from_wheel_speed, t_from_wheel_accel,
                   t_from_wheel_jerk, t_from_ang_velocity)

    # -- Kinematics ----------------------------------------------------------
    def get_kinematics(self, t: float) -> tuple[float, float, float, float]:
        """(arc distance, speed, acceleration, jerk) of the BODY along the arc.

        All values are >= 0 in magnitude terms (the robot only drives forward);
        the heading is  sign * distance / R.  For tests and plots only.
        """
        s_u, v_u, a_u, j_u = self._profile.state(t / self._duration)
        d, T = self._arc_length, self._duration
        return (d * s_u, d / T * v_u, d / T**2 * a_u, d / T**3 * j_u)

    def _compute_velocity(self, t: float) -> Velocity:
        """Command for 0 <= t < T (boundaries handled by the base class)."""
        _, v, _, _ = self.get_kinematics(t)
        return (v, self._sign * v / self._radius)  # (linear.x, angular.z)


class CircleLeftStep(CircleStep):
    """Forward arc turning LEFT (counter-clockwise). Give positive magnitudes."""

    def __init__(self, duration: float, arc_angle: float = CIRCLE_ARC_ANGLE,
                 radius: float = CIRCLE_RADIUS, limits: RobotLimits | None = None) -> None:
        if not arc_angle > 0.0:
            raise ValueError(f"CircleLeftStep arc_angle must be positive, got {arc_angle!r}")
        super().__init__(duration, +arc_angle, radius, limits)


class CircleRightStep(CircleStep):
    """Forward arc turning RIGHT (clockwise). Give positive magnitudes."""

    def __init__(self, duration: float, arc_angle: float = CIRCLE_ARC_ANGLE,
                 radius: float = CIRCLE_RADIUS, limits: RobotLimits | None = None) -> None:
        if not arc_angle > 0.0:
            raise ValueError(f"CircleRightStep arc_angle must be positive, got {arc_angle!r}")
        super().__init__(duration, -arc_angle, radius, limits)
