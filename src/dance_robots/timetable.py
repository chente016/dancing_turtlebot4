"""Timetable: where each measure (compás) sits on the song clock.

One row per measure: its number, its start time and its duration, all in seconds.
It comes from the music-analysis team, so time signature, tempo and tempo changes
are already baked in; this code never needs to know the BPM or the meter.

A move that spans several measures gets the window
    start(first measure)  ->  end(last measure)
computed as end minus start, so rounding in the table cannot accumulate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import rules
from .csvio import parse_float, parse_int, read_table
from .issues import ERROR, WARNING, Issue, has_errors


@dataclass(frozen=True)
class Measure:
    number: int
    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class Timetable:
    """Validated, consecutive list of measures."""

    def __init__(self, measures: Sequence[Measure]) -> None:
        if not measures:
            raise ValueError("a timetable needs at least one measure")
        ordered = sorted(measures, key=lambda m: m.number)
        for a, b in zip(ordered, ordered[1:]):
            if b.number != a.number + 1:
                raise ValueError(f"measure numbers must be consecutive: {a.number} -> {b.number}")
            if b.start_s < a.end_s - rules.TIME_TOLERANCE_S:
                raise ValueError(f"measure {b.number} starts before measure {a.number} ends")
        if any(m.duration_s <= 0.0 for m in ordered):
            raise ValueError("every measure needs a positive duration")
        self._measures = {m.number: m for m in ordered}
        self._ordered = ordered

    # -- Lookup ------------------------------------------------------------------
    @property
    def measures(self) -> list[Measure]:
        return list(self._ordered)

    @property
    def first_number(self) -> int:
        return self._ordered[0].number

    @property
    def last_number(self) -> int:
        return self._ordered[-1].number

    def __contains__(self, number: int) -> bool:
        return number in self._measures

    def __getitem__(self, number: int) -> Measure:
        return self._measures[number]

    def window(self, first: int, last: int) -> tuple[float, float]:
        """(start_s, duration_s) from the start of `first` to the end of `last`."""
        start = self._measures[first].start_s
        return start, self._measures[last].end_s - start

    # -- Convenience builders / writers ----------------------------------------------
    @classmethod
    def from_tempo_map(cls, segments: Sequence[tuple[int, float]],
                       beats_per_measure: int = 3, first_measure: int = 1) -> "Timetable":
        """Build a timetable from [(number_of_measures, bpm), ...].

        Handy for tests and demos. In production the table comes from the
        analysis team, and this class never needs the meter or the tempo.
        """
        measures: list[Measure] = []
        number, start = first_measure, 0.0
        for count, bpm in segments:
            duration = beats_per_measure * 60.0 / bpm
            for _ in range(count):
                measures.append(Measure(number, start, duration))
                number, start = number + 1, start + duration
        return cls(measures)

    def to_csv(self, path) -> None:
        """Write the timetable in the CSV format the loader expects."""
        with open(Path(path), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["measure", "start_s", "duration_s"])
            for m in self._ordered:
                writer.writerow([m.number, f"{m.start_s:.6f}", f"{m.duration_s:.6f}"])


def load_timetable(path) -> tuple[Timetable | None, list[Issue]]:
    """Read and validate a timetable CSV (columns: measure, start_s, duration_s).

    Returns (timetable, issues). The timetable is None if any error was found.
    """
    table, issues = read_table(path, required=("measure", "start_s", "duration_s"))
    if table is None:
        return None, issues

    measures: list[Measure] = []
    seen: dict[int, int] = {}  # measure number -> line where it first appeared
    for row in table.rows:
        where = f"{table.name}, line {row.line}"
        try:
            number = parse_int(row.cells["measure"], table.decimal_comma)
            start = parse_float(row.cells["start_s"], table.decimal_comma)
            duration = parse_float(row.cells["duration_s"], table.decimal_comma)
        except ValueError as exc:
            issues.append(Issue(ERROR, where, str(exc)))
            continue
        if number < 0:
            issues.append(Issue(ERROR, where, f"measure number {number} must not be negative"))
            continue
        if start < 0.0:
            issues.append(Issue(ERROR, where, f"start_s {start:g} must not be negative"))
            continue
        if duration <= 0.0:
            issues.append(Issue(ERROR, where, f"duration_s {duration:g} must be greater than 0"))
            continue
        if number in seen:
            issues.append(Issue(ERROR, where, f"measure {number} is repeated (first seen on line {seen[number]})"))
            continue
        seen[number] = row.line
        measures.append(Measure(number, start, duration))

    if has_errors(issues):
        return None, issues
    if not measures:
        return None, issues + [Issue(ERROR, table.name, "the timetable has no measures")]

    measures.sort(key=lambda m: m.number)
    for a, b in zip(measures, measures[1:]):
        if b.number != a.number + 1:
            issues.append(Issue(ERROR, table.name,
                                f"measures must be consecutive but {a.number} is followed by {b.number}"))
        gap = b.start_s - a.end_s
        if gap < -rules.TIME_TOLERANCE_S:
            issues.append(Issue(ERROR, table.name,
                                f"measure {b.number} starts {-gap:.3f} s before measure {a.number} ends"))
        elif gap > rules.TIME_TOLERANCE_S:
            issues.append(Issue(WARNING, table.name,
                                f"there is a {gap:.3f} s gap between measure {a.number} and "
                                f"measure {b.number} (a pause in the score?)"))
    if has_errors(issues):
        return None, issues
    return Timetable(measures), issues