"""Normalized 7-segment S-curve shared by ALL dance steps.

One fixed, jerk-limited profile on unit time u = t/T covering unit distance
(or unit angle). Each step scales it to its own displacement and duration:
    velocity  ~ D/T      acceleration ~ D/T^2      jerk ~ D/T^3

Keeping it here means the "feel" of the whole choreography (the two shape
constants below) is tuned in ONE place for forward, turn and circle moves.

Pure Python (no ROS, no numpy). Compatible with Python 3.10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# Tunable shape constants (fractions of the total duration T).
# These are the knobs to adjust after looking at the plots.
#
# The 7 segments are: J+ , A , J- , CRUISE , J- , A , J+
#   J = jerk segments (4 of them, all the same length)
#   A = constant-acceleration segments (2 of them, same length)
# so:  4*fj + 2*ACCEL_FRACTION + CRUISE_FRACTION = 1
# We choose ACCEL and CRUISE and DERIVE fj from that equation.
#
# With both at 0.0 the move is "pure jerk": acceleration is a triangle and
# velocity is one smooth bell, which feels the most dancer-like and has the
# lowest peak jerk for a given T. Raising them gives a flatter velocity plateau.
# ----------------------------------------------------------------------------
ACCEL_FRACTION = 0.0   # length of EACH constant-acceleration segment, in units of T
CRUISE_FRACTION = 0.0  # length of the constant-velocity segment, in units of T


# ----------------------------------------------------------------------------
# Normalized profile
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class _Segment:
    """One constant-jerk piece of the unit profile.

    Stores the state at the START of the segment so we can evaluate any point
    inside it with the constant-jerk polynomial equations.
    """

    start: float  # segment start in unit time u
    end: float    # segment end in unit time u
    jerk: float   # constant jerk during the segment (unit profile)
    x0: float     # position at segment start
    v0: float     # velocity at segment start
    a0: float     # acceleration at segment start


class UnitSCurve:
    """Symmetric 7-segment S-curve covering unit distance in unit time.

    Built once in the constructor; afterwards `state(u)` is cheap (it just
    finds the active segment and evaluates a polynomial).
    """

    def __init__(self, accel_fraction: float, cruise_fraction: float) -> None:
        fa, fc = accel_fraction, cruise_fraction

        # Validate the shape parameters before using them.
        if fa < 0.0 or fc < 0.0 or (2.0 * fa + fc) >= 1.0:
            raise ValueError(
                "Need accel_fraction >= 0, cruise_fraction >= 0 and "
                f"2*accel + cruise < 1 (got accel={fa}, cruise={fc})"
            )

        # Length of each jerk segment, from 4*fj + 2*fa + fc = 1.
        fj = (1.0 - 2.0 * fa - fc) / 4.0

        # --- Closed-form peak values of the UNIT profile -------------------
        # Velocity profile = accel phase (duration 2fj+fa) + cruise (fc)
        # + mirrored decel phase. The accel phase is an "S" that is point-
        # symmetric, so its average velocity is v_peak/2. Therefore:
        #     distance = v_peak * (2fj + fa + fc)  = 1   (unit distance)
        self.v_peak = 1.0 / (2.0 * fj + fa + fc)
        # During the first two phases the acceleration reaches
        #     a_peak = jerk * fj,   and   v_peak = a_peak * (fj + fa)
        self.a_peak = self.v_peak / (fj + fa)
        # Jerk magnitude that produces that a_peak in time fj.
        self.j_peak = self.a_peak / fj

        # --- Build the list of constant-jerk segments ----------------------
        # (duration, jerk) for J+, A, J-, CRUISE, J-, A, J+
        j = self.j_peak
        plan = [(fj, +j), (fa, 0.0), (fj, -j), (fc, 0.0),
                (fj, -j), (fa, 0.0), (fj, +j)]

        self._segments: list[_Segment] = []
        u = x = v = a = 0.0  # running state at the start of the next segment
        for duration, jerk in plan:
            if duration <= 0.0:
                continue  # skip empty segments (e.g. no cruise, no const-accel)
            self._segments.append(_Segment(u, u + duration, jerk, x, v, a))
            # Advance the state to the END of this segment. Order matters:
            # position uses the OLD v and a, velocity uses the OLD a.
            x += v * duration + a * duration**2 / 2.0 + jerk * duration**3 / 6.0
            v += a * duration + jerk * duration**2 / 2.0
            a += jerk * duration
            u += duration

        # --- Self-check: the profile must land at rest on unit distance ----
        if not (math.isclose(u, 1.0, abs_tol=1e-9)
                and math.isclose(x, 1.0, abs_tol=1e-9)
                and abs(v) < 1e-9 and abs(a) < 1e-9):
            raise RuntimeError(
                f"Unit S-curve is inconsistent: u={u}, x={x}, v={v}, a={a}"
            )

    def state(self, u: float) -> tuple[float, float, float, float]:
        """Return (position, velocity, acceleration, jerk) at unit time u."""
        if u <= 0.0:
            return (0.0, 0.0, 0.0, 0.0)  # before the move: at the start, at rest
        if u >= 1.0:
            return (1.0, 0.0, 0.0, 0.0)  # after the move: at the goal, at rest

        # Find the segment containing u (at most 7, so a linear scan is fine).
        seg = self._segments[-1]
        for candidate in self._segments:
            if u < candidate.end:
                seg = candidate
                break

        # Constant-jerk polynomial equations from the segment start.
        dt = u - seg.start
        a = seg.a0 + seg.jerk * dt
        v = seg.v0 + seg.a0 * dt + seg.jerk * dt**2 / 2.0
        x = seg.x0 + seg.v0 * dt + seg.a0 * dt**2 / 2.0 + seg.jerk * dt**3 / 6.0
        return (x, v, a, seg.jerk)