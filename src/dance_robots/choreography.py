"""The Choreography sequencer plus the choreography CSV loader and validator.

Who does what
-------------
* A DanceStep computes a velocity from its LOCAL time (seconds since it began).
* The Choreography (sequencer) owns the schedule. Given the GLOBAL time it finds
  the active step, converts to local time (t_local = t_song - step_start) and asks
  that step. During rests, and before / after the dance, it returns (0, 0).
  It does no profile math itself.
* The ROS node only calls `get_velocity(t)` every timer tick and publishes it.

CSV format (one row per move; measures are whole measures only)
-----------------------------------------------------------------
    first_measure,last_measure,step,angle_deg,radius_m
    40,,forward,,
    41,43,turn_left,90,
    45,,circle_right,45,0.3

* last_measure blank = a single measure. Measures not listed are rests.
* angle_deg / radius_m are optional; blanks use the defaults from rules.py.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence

from . import rules
from .csvio import parse_float, parse_int, read_table
from .dance_step import InfeasibleStepError, RobotLimits
from .issues import ERROR, WARNING, Issue, has_errors
from .step_spec import StepSpec
from .timetable import Timetable


# ----------------------------------------------------------------------------
# Data objects
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Move:
    """One row of the choreography file (before checking it against the timetable)."""

    line: int
    first_measure: int
    last_measure: int
    spec: StepSpec


@dataclass(frozen=True)
class ScheduledStep:
    """A step placed on the song clock."""

    first_measure: int
    last_measure: int
    start_s: float      # song time at which the step begins
    duration_s: float   # window length = end of last measure - start of first
    spec: StepSpec
    step: object        # the DanceStep (kept loose to avoid a heavy import here)

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


# ----------------------------------------------------------------------------
# The sequencer
# ----------------------------------------------------------------------------
class Choreography:
    """Ordered, non-overlapping steps on the song clock.

    `music_start_offset` is the NODE time at which the song's t = 0 happens:
        t_song = t_node - music_start_offset
    A positive value starts the dance later. To compensate a robot that reacts
    ~0.1 s late, use -0.1 so commands go out 0.1 s early. It is one number, tuned
    on the real robot.
    """

    def __init__(self, scheduled: Sequence[ScheduledStep], music_start_offset: float = 0.0) -> None:
        steps = sorted(scheduled, key=lambda s: s.start_s)
        for a, b in zip(steps, steps[1:]):
            if b.start_s < a.end_s - rules.TIME_TOLERANCE_S:
                raise ValueError(f"steps overlap: measures {a.first_measure}-{a.last_measure} "
                                 f"and {b.first_measure}-{b.last_measure}")
        self._steps = steps
        self._starts = [s.start_s for s in steps]  # for the binary search
        self.music_start_offset = float(music_start_offset)

    @property
    def schedule(self) -> list[ScheduledStep]:
        return list(self._steps)

    @property
    def song_end_s(self) -> float:
        """Song time at which the last step ends (0 if there are no steps)."""
        return self._steps[-1].end_s if self._steps else 0.0

    @property
    def node_end_s(self) -> float:
        """Node time at which the dance is over."""
        return self.song_end_s + self.music_start_offset

    def active_step(self, t_song: float) -> ScheduledStep | None:
        """The step whose window contains `t_song`, or None (rest / outside)."""
        i = bisect_right(self._starts, t_song) - 1  # last step starting at or before t
        if i < 0:
            return None
        step = self._steps[i]
        return step if t_song < step.end_s else None

    def get_velocity(self, t_node: float) -> tuple[float, float]:
        """(linear.x, angular.z) to publish at node time `t_node`."""
        t_song = t_node - self.music_start_offset
        active = self.active_step(t_song)
        if active is None:
            return (0.0, 0.0)  # rest, before the dance, or after it
        return active.step.get_velocity(t_song - active.start_s)


# ----------------------------------------------------------------------------
# Reading the choreography CSV
# ----------------------------------------------------------------------------
def parse_moves(path) -> tuple[list[Move], list[Issue]]:
    """Read the CSV into Move objects (syntax only; no timetable needed yet)."""
    table, issues = read_table(
        path, required=("first_measure", "step"),
        optional=("last_measure", "angle_deg", "radius_m"))
    if table is None:
        return [], issues

    moves: list[Move] = []
    for row in table.rows:
        where = f"{table.name}, line {row.line}"
        cells, dc = row.cells, table.decimal_comma
        try:
            first = parse_int(cells["first_measure"], dc) if cells["first_measure"] else None
            if first is None:
                raise ValueError("first_measure is empty")
            last_text = cells.get("last_measure", "")
            last = parse_int(last_text, dc) if last_text else first
            angle = parse_float(cells["angle_deg"], dc) if cells.get("angle_deg") else None
            radius = parse_float(cells["radius_m"], dc) if cells.get("radius_m") else None
        except ValueError as exc:
            issues.append(Issue(ERROR, where, str(exc)))
            continue

        kind = cells["step"].strip().lower().replace(" ", "_").replace("-", "_")
        if kind == "":
            issues.append(Issue(ERROR, where, "step is empty"))
            continue
        moves.append(Move(row.line, first, last, StepSpec(kind, angle, radius)))
    return moves, issues


# ----------------------------------------------------------------------------
# Checking moves against the timetable and the robot limits
# ----------------------------------------------------------------------------
def _measures_label(first: int, last: int) -> str:
    return f"measure {first}" if first == last else f"measures {first}-{last}"


def _suggestions(spec: StepSpec, move: Move, window: float, timetable: Timetable,
                 limits: RobotLimits | None) -> list[str]:
    """Concrete ways to fix a window that is too short."""
    options: list[str] = []
    tmin = spec.min_duration(limits)

    # Option 1: extend over more measures.
    for last in range(move.last_measure + 1, timetable.last_number + 1):
        _, longer = timetable.window(move.first_measure, last)
        if longer >= tmin:
            options.append(f"use {_measures_label(move.first_measure, last)} ({longer:.2f} s)")
            break

    # Option 2: a smaller angle (turns and circles).
    max_angle = spec.max_angle_deg_for(window, limits)
    if max_angle is not None:
        floored = math.floor(max_angle * 10.0) / 10.0  # round DOWN so it surely fits
        if 0.0 < floored < spec.angle:
            options.append(f"reduce the angle to at most {floored:g}°")

    # Option 3: a smaller radius (circles).
    max_radius = spec.max_radius_m_for(window, limits)
    if max_radius is not None and max_radius < spec.radius:
        floored = math.floor(max_radius * 100.0) / 100.0
        if floored >= rules.CIRCLE_RADIUS_MIN_M:
            options.append(f"use a radius of at most {floored:g} m")
    return options


def _check_move(move: Move, timetable: Timetable, limits: RobotLimits | None,
                source: str) -> tuple[ScheduledStep | None, list[Issue]]:
    where = f"{source}, line {move.line}"
    spec = move.spec
    issues: list[Issue] = []

    # 1) Measures must exist in the timetable and be in order.
    ok = True
    if move.last_measure < move.first_measure:
        issues.append(Issue(ERROR, where, f"last_measure {move.last_measure} is before "
                                          f"first_measure {move.first_measure}"))
        ok = False
    for m in sorted({move.first_measure, move.last_measure}):  # a set: report each once
        if m not in timetable:
            issues.append(Issue(ERROR, where, f"measure {m} is not in the timetable "
                                              f"(it covers measures {timetable.first_number}-{timetable.last_number})"))
            ok = False

    # 2) The step name and its parameters must follow the choreography rules.
    for message in spec.validate():
        issues.append(Issue(ERROR, where, message))
        ok = False
    if not ok:
        return None, issues

    # 3) The window must be long enough for the move.
    start, window = timetable.window(move.first_measure, move.last_measure)
    label = _measures_label(move.first_measure, move.last_measure)
    tmin = spec.min_duration(limits)
    if window < tmin:
        text = (f"{spec.describe()} needs at least {tmin:.2f} s but {label} "
                f"last{'s' if move.first_measure == move.last_measure else ''} only {window:.2f} s.")
        options = _suggestions(spec, move, window, timetable, limits)
        if options:
            text += " Options: " + "; or ".join(options) + "."
        issues.append(Issue(ERROR, where, text))
        return None, issues

    # 4) Feasible. Warn if so slow that the real robot may not move properly.
    try:
        step = spec.build(window, limits)
    except InfeasibleStepError as exc:  # should not happen after the check above
        issues.append(Issue(ERROR, where, str(exc)))
        return None, issues

    peak, unit = spec.peak_speed(window, limits)
    floor = rules.MIN_RELIABLE_ANGULAR_SPEED if unit == "rad/s" else rules.MIN_RELIABLE_LINEAR_SPEED
    if peak < floor:
        issues.append(Issue(WARNING, where,
                            f"{spec.describe()} over {label} peaks at only {peak:.3f} {unit}; "
                            f"it may be too slow to move reliably on the real robot "
                            f"(consider fewer measures or a bigger move)"))
    return ScheduledStep(move.first_measure, move.last_measure, start, window, spec, step), issues


def build_choreography(moves: Sequence[Move], timetable: Timetable,
                       limits: RobotLimits | None = None,
                       music_start_offset: float = 0.0,
                       source: str = "choreography.csv"):
    """Check every move and assemble the sequencer.

    Returns (Choreography | None, issues). The choreography is None when any
    error was found; warnings alone do not block it.
    """
    issues: list[Issue] = []
    scheduled: list[ScheduledStep] = []

    for move in moves:
        item, move_issues = _check_move(move, timetable, limits, source)
        issues.extend(move_issues)
        if item is not None:
            scheduled.append(item)

    # Two moves must not claim the same measure.
    ranged = sorted((m for m in moves if m.first_measure <= m.last_measure),
                    key=lambda m: (m.first_measure, m.line))
    previous: Move | None = None
    for m in ranged:
        if previous is not None and m.first_measure <= previous.last_measure:
            issues.append(Issue(
                ERROR, f"{source}, line {m.line}",
                f"{_measures_label(m.first_measure, m.last_measure)} overlaps with line "
                f"{previous.line} ({_measures_label(previous.first_measure, previous.last_measure)}); "
                f"a measure can hold only one move"))
        if previous is None or m.last_measure > previous.last_measure:
            previous = m

    if not moves:
        issues.append(Issue(WARNING, source, "the choreography has no moves"))
    if has_errors(issues):
        return None, issues
    return Choreography(scheduled, music_start_offset), issues


def load_choreography(path, timetable: Timetable, limits: RobotLimits | None = None,
                      music_start_offset: float = 0.0):
    """Read + check a choreography CSV against a timetable. Returns (Choreography | None, issues)."""
    from pathlib import Path
    moves, parse_issues = parse_moves(path)
    choreography, build_issues = build_choreography(
        moves, timetable, limits, music_start_offset, source=Path(path).name)
    all_issues = parse_issues + build_issues
    if has_errors(all_issues):
        return None, all_issues
    return choreography, all_issues