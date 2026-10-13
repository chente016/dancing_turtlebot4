"""Base abstractions for the dancing-robot choreography.

This module is PURE PYTHON: it must never import ROS 2 (rclpy, geometry_msgs...).
Keeping the math free of ROS lets us test every step offline with plain pytest,
and the ROS node later only has to ask a step "what velocity at time t?".

Compatible with Python 3.10 (the version shipped with ROS 2 Humble).
"""

from __future__ import annotations  # allows modern type hints on Python 3.10

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

# A velocity command is (linear.x [m/s], angular.z [rad/s]).
# This maps 1:1 to the fields of geometry_msgs/msg/Twist that we use.
Velocity = tuple[float, float]


@dataclass(frozen=True)
class RobotLimits:
    """Conservative kinematic limits used for feasibility checks.

    Values come from the engineering handover. They are deliberately below the
    raw hardware limits (0.460 m/s and 0.9 m/s^2) to protect the motors and
    avoid wheel slip during open-loop execution.

    frozen=True makes the object immutable, so a step can't accidentally
    change the limits that other steps are also using.

    NOTE: angular limits are intentionally not defined yet. The handover does
    not give them, so we will add them when we implement the turn steps.
    """

    v_max: float = 0.306  # max linear velocity [m/s]
    a_max: float = 0.6    # max linear acceleration [m/s^2]
    j_max: float = 3.0    # max linear jerk [m/s^3] (= a_max / t1, t1 = 0.2 s)
    w_max: float = 1.9    # max angular velocity [rad/s]
    track_width: float = 0.235  # [m]

    @property
    def alpha_max(self) -> float:
        """Estimated max angular acceleration [rad/s^2] = 2 * a_max / track."""
        return 2.0 * self.a_max / self.track_width

    @property
    def jerk_ang_max(self) -> float:
        """Estimated max angular jerk [rad/s^3] = 2 * j_max / track."""
        return 2.0 * self.j_max / self.track_width


class InfeasibleStepError(ValueError):
    """Raised when a step cannot be performed in the requested duration
    without exceeding the robot limits (e.g. the measure is too short)."""


class DanceStep(ABC):
    """Abstract dance move: starts at rest, ends at rest, lasts `duration` s.

    Design rules (agreed in the architecture discussion):
      * A step is a pure function of LOCAL time `t`, measured from the moment
        THAT step starts. It knows nothing about the clock, the music or the
        other steps; the sequencer handles the time offsets.
      * The step's duration comes from the music timetable (one measure, or a
        range of measures) and is given to the constructor.
      * Outside [0, duration) the robot must be stopped, so we return zeros.
    """

    def __init__(self, duration: float, limits: RobotLimits | None = None) -> None:
        # Reject nonsense early: NaN/inf or non-positive durations would
        # silently produce garbage profiles later.
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"duration must be a positive finite number, got {duration!r}")

        self._duration = float(duration)
        # Use the default limits unless the caller provides their own.
        self._limits = limits if limits is not None else RobotLimits()

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def duration(self) -> float:
        """Total time window of the step in seconds (T)."""
        return self._duration

    @property
    def limits(self) -> RobotLimits:
        """Kinematic limits this step is checked against."""
        return self._limits

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------
    @abstractmethod
    def min_duration(self) -> float:
        """Smallest duration (s) in which this step respects ALL limits.

        Each subclass computes this from its own distance/angle and shape.
        """

    def _check_feasibility(self) -> None:
        """Raise InfeasibleStepError if `duration` is below `min_duration()`.

        Subclasses must call this at the END of their own __init__, once the
        attributes needed by min_duration() (distance, angle...) are set.
        We can't call it from this base __init__ because those attributes
        don't exist yet at that point.
        """
        t_min = self.min_duration()
        if self._duration < t_min:
            raise InfeasibleStepError(
                f"{type(self).__name__}: duration {self._duration:.3f} s is shorter "
                f"than the minimum feasible {t_min:.3f} s"
            )

    # ------------------------------------------------------------------
    # Velocity generation
    # ------------------------------------------------------------------
    def get_velocity(self, t: float) -> Velocity:
        """Velocity command at local time `t` (seconds since this step began).

        This is the ONLY method the sequencer / ROS node needs to call.
        It handles the boundaries so subclasses don't have to:
          * t < 0       -> step hasn't started: stay still.
          * t >= T      -> step is over: stay still. Using ">=" matters at
                           50 Hz because the last timer tick often lands a
                           little past T, and we must not command motion then.
        """
        if t < 0.0 or t >= self._duration:
            return (0.0, 0.0)
        return self._compute_velocity(t)

    @abstractmethod
    def _compute_velocity(self, t: float) -> Velocity:
        """Subclass hook: velocity for 0 <= t < duration.

        Only called with an in-range `t`, so implementations can focus on the
        profile math and ignore boundary handling.
        """