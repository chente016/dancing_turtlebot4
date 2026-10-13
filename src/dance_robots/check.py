"""Plain-language checker for a choreography file.

Usage (from the project root):
    uv run python -m dance_robots.check examples/choreography.csv examples/timetable.csv

It runs EXACTLY the same loading and validation code the robot node will use, so
a file that passes here cannot be rejected later. It prints:
  1. problems (errors must be fixed; warnings are worth a look), each with the
     file line and, when possible, concrete ways to fix it;
  2. one line per move with its time window and how much margin it has;
  3. a measure-by-measure map of the song, to show choreographers where each
     move sits in the score.
Exit code: 0 = usable (maybe with warnings), 1 = errors found.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .choreography import Choreography, load_choreography
from .issues import ERROR, WARNING, Issue, has_errors
from .timetable import Timetable, load_timetable


def _measures_text(first: int, last: int) -> str:
    return f"{first}" if first == last else f"{first}-{last}"


def format_moves(choreography: Choreography) -> list[str]:
    """One line per move: measures, time window, move, margin."""
    lines = ["Moves"]
    for s in choreography.schedule:
        needed = s.spec.min_duration()
        lines.append(
            f"  measures {_measures_text(s.first_measure, s.last_measure):<7} "
            f"{s.start_s:7.2f} s to {s.end_s:7.2f} s ({s.duration_s:5.2f} s)   "
            f"{s.spec.describe():<26} OK, needs at least {needed:.2f} s "
            f"(margin {s.duration_s - needed:.2f} s)")
    return lines


def format_song_map(timetable: Timetable, choreography: Choreography) -> list[str]:
    """Measure-by-measure table: which move sits in each measure."""
    by_measure: dict[int, tuple[str, bool]] = {}
    for s in choreography.schedule:
        for m in range(s.first_measure, s.last_measure + 1):
            continues = m != s.first_measure
            label = s.spec.describe()
            if continues:
                label = "(continues)"
            elif s.last_measure != s.first_measure:
                label += f"  [measures {s.first_measure}-{s.last_measure}]"
            by_measure[m] = (label, continues)

    lines = ["Song map", "  measure   starts at   length   move",
             "  -------   ---------   ------   ----"]
    for m in timetable.measures:
        label = by_measure.get(m.number, ("-  (rest)", False))[0]
        lines.append(f"  {m.number:>7}   {m.start_s:7.2f} s   {m.duration_s:4.2f} s   {label}")
    return lines


def format_issues(issues: Sequence[Issue]) -> list[str]:
    errors = [i for i in issues if i.severity == ERROR]
    warnings = [i for i in issues if i.severity == WARNING]
    lines: list[str] = []
    if errors:
        lines.append(f"Errors ({len(errors)}) - these must be fixed")
        lines += [f"  * {i.where}: {i.message}" for i in errors]
    if warnings:
        lines.append(f"Warnings ({len(warnings)}) - worth a look")
        lines += [f"  * {i.where}: {i.message}" for i in warnings]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dance_robots.check",
        description="Check a choreography CSV against a timetable CSV.")
    parser.add_argument("choreography", help="choreography CSV (first_measure,last_measure,step,...)")
    parser.add_argument("timetable", help="timetable CSV (measure,start_s,duration_s)")
    parser.add_argument("--no-map", action="store_true", help="do not print the measure-by-measure song map")
    args = parser.parse_args(argv)

    print(f"Choreography: {args.choreography}")
    print(f"Timetable:    {args.timetable}\n")

    timetable, issues = load_timetable(args.timetable)
    choreography = None
    if timetable is not None:
        choreography, choreo_issues = load_choreography(args.choreography, timetable)
        issues = issues + choreo_issues
    else:
        issues.append(Issue(WARNING, "checker", "the choreography was not checked because the timetable has errors"))

    problems = format_issues(issues)
    if problems:
        print("\n".join(problems))
        print()

    if has_errors(issues) or choreography is None or timetable is None:
        print("RESULT: NOT OK - fix the errors above and run the check again.")
        return 1

    print("\n".join(format_moves(choreography)))
    print()
    if not args.no_map:
        print("\n".join(format_song_map(timetable, choreography)))
        print()
    n_moves = len(choreography.schedule)
    print(f"RESULT: OK - {n_moves} move{'s' if n_moves != 1 else ''}, "
          f"the dance ends at {choreography.song_end_s:.2f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())