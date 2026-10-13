"""StepSpec: one choreography move described by plain values (name, angle, radius).

This is the piece shared by the sequencer and the plain-language checker, so they
can never disagree. It knows how to:
  * validate the values against the choreography rules (rules.py),
  * build the right DanceStep for a given time window,
  * answer feasibility questions ("what is the shortest window?",
    "what is the biggest angle that fits this window?").

Pure Python, compatible with Python 3.10.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from . import rules
from .circle_step import CircleLeftStep, CircleRightStep
from .dance_step import DanceStep, RobotLimits
from .linear_step import BackwardStep, ForwardStep
from .turn_step import TurnLeftStep, TurnRightStep

STEP_KINDS = ("forward", "backward", "turn_left", "turn_right",
              "circle_left", "circle_right")

_LINEAR = ("forward", "backward")
_TURNS = ("turn_left", "turn_right")
_CIRCLES = ("circle_left", "circle_right")

# A step's min_duration() does not depend on its duration, so to ask for it we
# build the step with a huge (always feasible) duration.
_HUGE_DURATION = 1.0e6

_BISECT_ITERATIONS = 60


@dataclass(frozen=True)
class StepSpec:
    """A move as written in the choreography file.

    `angle_deg` and `radius_m` are None when the CSV cell was blank; the rules
    supply the defaults (see the `angle` and `radius` properties).
    """

    kind: str
    angle_deg: float | None = None
    radius_m: float | None = None

    # -- Classification and defaults -----------------------------------------
    @property
    def family(self) -> str:
        """'linear', 'turn', 'circle' or 'unknown'."""
        if self.kind in _LINEAR:
            return "linear"
        if self.kind in _TURNS:
            return "turn"
        if self.kind in _CIRCLES:
            return "circle"
        return "unknown"

    @property
    def angle(self) -> float | None:
        """Angle in degrees with the default applied (None for linear moves)."""
        if self.family == "turn":
            return self.angle_deg if self.angle_deg is not None else rules.TURN_ANGLE_DEFAULT_DEG
        if self.family == "circle":
            return self.angle_deg if self.angle_deg is not None else rules.CIRCLE_ANGLE_DEFAULT_DEG
        return None

    @property
    def radius(self) -> float | None:
        """Radius in metres with the default applied (None unless a circle)."""
        if self.family == "circle":
            return self.radius_m if self.radius_m is not None else rules.CIRCLE_RADIUS_DEFAULT_M
        return None

    # -- Validation against the choreography rules -------------------------------
    def validate(self) -> list[str]:
        """Return plain-language error messages (empty list = valid)."""
        errors: list[str] = []
        fam = self.family

        if fam == "unknown":
            return [f"unknown step '{self.kind}'. Valid steps: {', '.join(STEP_KINDS)}"]

        if fam == "linear":
            if self.angle_deg is not None:
                errors.append(f"'{self.kind}' has a fixed distance of "
                              f"{rules.FORWARD_DISTANCE_M:g} m; leave angle_deg blank")
            if self.radius_m is not None:
                errors.append(f"'{self.kind}' does not use radius_m; leave it blank")

        elif fam == "turn":
            if self.radius_m is not None:
                errors.append(f"'{self.kind}' turns in place and does not use radius_m; leave it blank")
            a = self.angle
            if not (rules.TURN_ANGLE_MIN_DEG <= a <= rules.TURN_ANGLE_MAX_DEG):
                errors.append(f"turn angle {a:g}° is not allowed "
                              f"(allowed: {rules.TURN_ANGLE_MIN_DEG:g}° to {rules.TURN_ANGLE_MAX_DEG:g}°)")

        else:  # circle
            a, r = self.angle, self.radius
            if not (0.0 < a <= rules.CIRCLE_ANGLE_MAX_DEG):
                errors.append(f"circle angle {a:g}° is not allowed "
                              f"(allowed: more than 0° and up to {rules.CIRCLE_ANGLE_MAX_DEG:g}°)")
            if not (rules.CIRCLE_RADIUS_MIN_M <= r <= rules.CIRCLE_RADIUS_MAX_M):
                errors.append(f"circle radius {r:g} m is not allowed "
                              f"(allowed: {rules.CIRCLE_RADIUS_MIN_M:g} m to {rules.CIRCLE_RADIUS_MAX_M:g} m)")
        return errors

    # -- Building ---------------------------------------------------------------------
    def build(self, duration: float, limits: RobotLimits | None = None) -> DanceStep:
        """Create the DanceStep for a window of `duration` seconds.

        Assumes validate() passed. Raises InfeasibleStepError if the window is
        shorter than min_duration().
        """
        k = self.kind
        if k == "forward":
            return ForwardStep(duration, rules.FORWARD_DISTANCE_M, limits)
        if k == "backward":
            return BackwardStep(duration, rules.BACKWARD_DISTANCE_M, limits)
        if k == "turn_left":
            return TurnLeftStep(duration, math.radians(self.angle), limits)
        if k == "turn_right":
            return TurnRightStep(duration, math.radians(self.angle), limits)
        if k == "circle_left":
            return CircleLeftStep(duration, math.radians(self.angle), self.radius, limits)
        if k == "circle_right":
            return CircleRightStep(duration, math.radians(self.angle), self.radius, limits)
        raise ValueError(f"unknown step '{k}'")

    # -- Feasibility questions ---------------------------------------------------------
    def min_duration(self, limits: RobotLimits | None = None) -> float:
        """Shortest window [s] in which this move respects all robot limits."""
        return self.build(_HUGE_DURATION, limits).min_duration()

    def fits(self, duration: float, limits: RobotLimits | None = None) -> bool:
        return self.min_duration(limits) <= duration

    def max_angle_deg_for(self, duration: float, limits: RobotLimits | None = None) -> float | None:
        """Largest allowed angle (deg) that fits `duration`, or None if even the
        smallest angle doesn't fit. Only for turns and circles.

        min_duration grows with the angle, so a bisection search finds the edge.
        """
        if self.family not in ("turn", "circle"):
            return None
        hi = rules.TURN_ANGLE_MAX_DEG if self.family == "turn" else rules.CIRCLE_ANGLE_MAX_DEG
        lo = rules.TURN_ANGLE_MIN_DEG if self.family == "turn" else 0.1
        at = lambda a: dataclasses.replace(self, angle_deg=a)
        if at(hi).fits(duration, limits):
            return hi
        if not at(lo).fits(duration, limits):
            return None
        for _ in range(_BISECT_ITERATIONS):  # invariant: lo fits, hi does not
            mid = (lo + hi) / 2.0
            lo, hi = (mid, hi) if at(mid).fits(duration, limits) else (lo, mid)
        return lo

    def max_radius_m_for(self, duration: float, limits: RobotLimits | None = None) -> float | None:
        """Largest allowed radius (m) <= the current one that fits `duration`
        (a smaller radius means a shorter arc), or None if none fits."""
        if self.family != "circle":
            return None
        hi, lo = self.radius, rules.CIRCLE_RADIUS_MIN_M
        at = lambda r: dataclasses.replace(self, radius_m=r)
        if at(hi).fits(duration, limits):
            return hi
        if not at(lo).fits(duration, limits):
            return None
        for _ in range(_BISECT_ITERATIONS):
            mid = (lo + hi) / 2.0
            lo, hi = (mid, hi) if at(mid).fits(duration, limits) else (lo, mid)
        return lo

    def peak_speed(self, duration: float, limits: RobotLimits | None = None) -> tuple[float, str]:
        """Peak speed over the window: (value, unit). rad/s for turns, m/s otherwise."""
        step = self.build(duration, limits)
        if self.family == "turn":
            return step.peak_angular_velocity, "rad/s"
        return step.peak_velocity, "m/s"

    def describe(self) -> str:
        """Short human-readable label, e.g. 'turn_left 90°'."""
        if self.family == "turn":
            return f"{self.kind} {self.angle:g}°"
        if self.family == "circle":
            return f"{self.kind} {self.angle:g}° r={self.radius:g} m"
        return self.kind