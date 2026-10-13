"""Tests for the Choreography sequencer and the choreography CSV loader."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dance_robots.choreography import (
    Choreography, Move, ScheduledStep, build_choreography, load_choreography, parse_moves)
from dance_robots.issues import ERROR, WARNING
from dance_robots.step_spec import StepSpec
from dance_robots.timetable import Timetable

EXAMPLES = Path(__file__).parent.parent / "examples"
HEADER = "first_measure,last_measure,step,angle_deg,radius_m\n"

# Measures 1-6 at 60 BPM (3.000 s), measures 7-10 at 70 BPM (2.571 s), 3/4 time.
TT = Timetable.from_tempo_map([(6, 60), (4, 70)])


def write(tmp_path, body, header=HEADER, name="choreography.csv"):
    path = tmp_path / name
    path.write_text(header + body, encoding="utf-8")
    return path


def load(tmp_path, body, **kwargs):
    return load_choreography(write(tmp_path, body), TT, **kwargs)


def errors(issues):
    return [i for i in issues if i.severity == ERROR]


def warnings(issues):
    return [i for i in issues if i.severity == WARNING]


# ---- Sequencer: dispatching velocities ---------------------------------------------------
def test_sequencer_delegates_with_local_time(tmp_path):
    """The velocity at global time t equals the step's velocity at (t - start)."""
    ch, issues = load(tmp_path, "3,,forward,,\n")
    step = ch.schedule[0]
    assert step.start_s == pytest.approx(6.0)
    for t_local in (0.5, 1.2, 2.9):
        assert ch.get_velocity(6.0 + t_local) == pytest.approx(step.step.get_velocity(t_local), abs=1e-12)
    assert ch.get_velocity(6.0 + 1.5)[0] > 0.0


def test_rests_and_outside_the_dance_are_zero(tmp_path):
    ch, _ = load(tmp_path, "2,,forward,,\n5,,forward,,\n")
    assert ch.get_velocity(0.0) == (0.0, 0.0)       # measure 1: rest, before the first step
    assert ch.get_velocity(7.5) == (0.0, 0.0)       # measures 3-4: a rest between steps
    assert ch.get_velocity(100.0) == (0.0, 0.0)     # after the dance
    assert ch.get_velocity(-1.0) == (0.0, 0.0)


def test_step_boundaries_are_at_rest(tmp_path):
    """Back-to-back steps: velocity is exactly 0 at the boundary and near it."""
    ch, _ = load(tmp_path, "1,,forward,,\n2,,backward,,\n")
    assert ch.get_velocity(3.0) == (0.0, 0.0)
    assert abs(ch.get_velocity(3.0 - 1e-6)[0]) < 1e-9
    assert abs(ch.get_velocity(3.0 + 1e-6)[0]) < 1e-9


def test_all_six_steps_in_one_dance(tmp_path):
    body = ("1,,forward,,\n2,,backward,,\n3,,turn_left,45,\n4,,turn_right,45,\n"
            "5,,circle_left,30,\n6,,circle_right,30,\n")
    ch, issues = load(tmp_path, body)
    assert errors(issues) == [] and len(ch.schedule) == 6
    v, w = ch.get_velocity(1.5)           # forward
    assert v > 0 and w == 0
    v, w = ch.get_velocity(4.5)           # backward
    assert v < 0 and w == 0
    v, w = ch.get_velocity(7.5)           # turn left
    assert v == 0 and w > 0
    v, w = ch.get_velocity(10.5)          # turn right
    assert v == 0 and w < 0
    v, w = ch.get_velocity(13.5)          # circle left
    assert v > 0 and w > 0
    v, w = ch.get_velocity(16.5)          # circle right
    assert v > 0 and w < 0


def test_multi_measure_window_spans_the_tempo_change(tmp_path):
    """Measures 6-8 = end(8) - start(6): 3.000 + 2 * 2.571 s."""
    ch, issues = load(tmp_path, "6,8,turn_left,90,\n")
    s = ch.schedule[0]
    assert s.start_s == pytest.approx(15.0)
    assert s.duration_s == pytest.approx(3.0 + 2 * 180 / 70)
    assert s.step.duration == pytest.approx(s.duration_s)


def test_a_step_covers_its_whole_window(tmp_path):
    """A long move is spread over the entire window (slow and continuous)."""
    ch, _ = load(tmp_path, "2,3,turn_left,90,\n")
    s = ch.schedule[0]
    assert ch.get_velocity(s.start_s + 0.01)[1] > 0
    assert ch.get_velocity(s.end_s - 0.01)[1] > 0
    assert ch.get_velocity(s.start_s + s.duration_s / 2)[1] == pytest.approx(
        s.step.peak_angular_velocity, rel=1e-6)


def test_music_start_offset_shifts_the_dance(tmp_path):
    ch0, _ = load(tmp_path, "1,,forward,,\n")
    ch1, _ = load(tmp_path, "1,,forward,,\n", music_start_offset=0.4)
    assert ch1.get_velocity(0.4 + 1.5) == pytest.approx(ch0.get_velocity(1.5), abs=1e-12)
    assert ch1.node_end_s == pytest.approx(ch0.node_end_s + 0.4)
    assert ch1.get_velocity(0.2) == (0.0, 0.0)     # the dance has not started yet


def test_negative_offset_sends_commands_early(tmp_path):
    ch0, _ = load(tmp_path, "1,,forward,,\n")
    ch1, _ = load(tmp_path, "1,,forward,,\n", music_start_offset=-0.1)
    assert ch1.get_velocity(1.5 - 0.1) == pytest.approx(ch0.get_velocity(1.5), abs=1e-12)


def test_sequencer_rejects_overlapping_steps():
    spec = StepSpec("forward")
    a = ScheduledStep(1, 1, 0.0, 3.0, spec, spec.build(3.0))
    b = ScheduledStep(2, 2, 2.0, 3.0, spec, spec.build(3.0))
    with pytest.raises(ValueError):
        Choreography([a, b])


def test_empty_choreography_is_all_rest():
    ch = Choreography([])
    assert ch.get_velocity(1.0) == (0.0, 0.0) and ch.song_end_s == 0.0


def test_schedule_order_does_not_depend_on_file_order(tmp_path):
    ch, issues = load(tmp_path, "4,,forward,,\n1,,backward,,\n")
    assert errors(issues) == []
    assert [s.first_measure for s in ch.schedule] == [1, 4]
    assert ch.get_velocity(1.5)[0] < 0 and ch.get_velocity(10.5)[0] > 0


# ---- Loader: parsing -------------------------------------------------------------------------------
def test_blank_last_measure_means_a_single_measure(tmp_path):
    moves, issues = parse_moves(write(tmp_path, "40,,forward,,\n41,43,turn_left,90,\n"))
    assert issues == []
    assert (moves[0].first_measure, moves[0].last_measure) == (40, 40)
    assert (moves[1].first_measure, moves[1].last_measure) == (41, 43)
    assert moves[1].spec == StepSpec("turn_left", 90.0, None)


def test_optional_columns_may_be_missing_entirely(tmp_path):
    path = write(tmp_path, "1,forward\n2,turn_left\n", header="first_measure,step\n")
    ch, issues = load_choreography(path, TT)
    assert errors(issues) == [] and len(ch.schedule) == 2
    assert ch.schedule[1].spec.angle == 30.0          # default angle


def test_step_names_are_forgiving(tmp_path):
    moves, _ = parse_moves(write(tmp_path, "1,,Turn Left,,\n2,,CIRCLE-RIGHT,,\n"))
    assert [m.spec.kind for m in moves] == ["turn_left", "circle_right"]


def test_comments_and_semicolons_with_decimal_comma(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("# comment\nfirst_measure;last_measure;step;angle_deg;radius_m\n"
                    "1;;circle_left;22,5;0,3\n", encoding="utf-8")
    moves, issues = parse_moves(path)
    assert issues == [] and moves[0].spec == StepSpec("circle_left", 22.5, 0.3)


def test_parse_errors_report_the_line(tmp_path):
    path = write(tmp_path, "1,,forward,,\nabc,,forward,,\n,,forward,,\n5,,,,\n6,,turn_left,xx,\n")
    moves, issues = parse_moves(path)
    assert [m.line for m in moves] == [2]
    by_line = {i.where.split("line ")[1]: i.message for i in errors(issues)}
    assert "'abc' is not a number" in by_line["3"]
    assert "first_measure is empty" in by_line["4"]
    assert "step is empty" in by_line["5"]
    assert "'xx' is not a number" in by_line["6"]


def test_missing_required_column(tmp_path):
    path = write(tmp_path, "1,forward\n", header="first_measure,action\n")
    moves, issues = parse_moves(path)
    assert moves == [] and "missing column(s): step" in errors(issues)[0].message


# ---- Loader: plain-language checking ---------------------------------------------------------------------
def test_too_short_circle_gives_concrete_options(tmp_path):
    ch, issues = load(tmp_path, "7,,circle_left,45,0.4\n")
    assert ch is None
    msg = errors(issues)[0].message
    assert "needs at least 2.66 s" in msg and "measure 7 lasts only 2.57 s" in msg
    assert "use measures 7-8 (5.14 s)" in msg
    assert "reduce the angle to at most 43.5°" in msg
    assert "use a radius of at most 0.38 m" in msg


def test_suggested_fixes_really_work(tmp_path):
    """Each option from the message, when applied, passes the checker."""
    for body in ["7,8,circle_left,45,0.4\n", "7,,circle_left,43.5,0.4\n", "7,,circle_left,45,0.38\n"]:
        ch, issues = load(tmp_path, body)
        assert ch is not None, (body, [str(i) for i in issues])


def test_too_short_forward_suggests_more_measures_only(tmp_path):
    tt = Timetable.from_tempo_map([(4, 120)], beats_per_measure=2)    # 1 s measures
    path = write(tmp_path, "1,,forward,,\n")
    ch, issues = load_choreography(path, tt)
    msg = errors(issues)[0].message
    assert "forward needs at least 2.00 s but measure 1 lasts only 1.00 s" in msg
    assert "use measures 1-2 (2.00 s)" in msg and "angle" not in msg


def test_plural_wording_for_multi_measure_windows(tmp_path):
    tt = Timetable.from_tempo_map([(4, 120)], beats_per_measure=2)
    ch, issues = load_choreography(write(tmp_path, "1,2,turn_left,90,\n3,4,circle_left,45,\n"), tt)
    # 2 s is enough for the 90 deg turn but not for the 45 deg circle (2.66 s).
    assert [i.where.split("line ")[1] for i in errors(issues)] == ["3"]
    assert "measures 3-4 last only 2.00 s" in errors(issues)[0].message


@pytest.mark.parametrize("row, fragment", [
    ("1,,turn_left,120,", "turn angle 120° is not allowed"),
    ("1,,turn_left,0.5,", "turn angle 0.5° is not allowed"),
    ("1,,circle_left,60,", "circle angle 60° is not allowed"),
    ("1,,circle_left,30,0.5", "circle radius 0.5 m is not allowed"),
    ("1,,forward,45,", "leave angle_deg blank"),
    ("1,,turn_left,45,0.3", "does not use radius_m"),
    ("1,,hop,,", "unknown step 'hop'"),
    ("99,,forward,,", "measure 99 is not in the timetable (it covers measures 1-10)"),
    ("5,3,forward,,", "last_measure 3 is before first_measure 5"),
])
def test_rule_violations_are_reported_in_plain_language(tmp_path, row, fragment):
    ch, issues = load(tmp_path, row + "\n")
    assert ch is None and fragment in errors(issues)[0].message


def test_each_missing_measure_is_reported_once(tmp_path):
    ch, issues = load(tmp_path, "12,,forward,,\n")
    assert len(errors(issues)) == 1


def test_overlapping_moves_are_reported_with_both_lines(tmp_path):
    ch, issues = load(tmp_path, "2,4,turn_left,90,\n4,,forward,,\n")
    assert ch is None
    msg = errors(issues)[0].message
    assert "measure 4 overlaps with line 2" in msg and "only one move" in msg


def test_adjacent_moves_are_not_overlapping(tmp_path):
    ch, issues = load(tmp_path, "2,3,turn_left,90,\n4,,forward,,\n")
    assert errors(issues) == [] and ch is not None


def test_all_problems_are_reported_at_once(tmp_path):
    ch, issues = load(tmp_path, "1,,turn_left,120,\n2,,hop,,\n7,,circle_left,45,\n")
    assert ch is None and len(errors(issues)) == 3


def test_very_slow_move_gives_a_warning_not_an_error(tmp_path):
    """A 1 degree turn over one measure peaks at ~0.012 rad/s: usable but risky."""
    ch, issues = load(tmp_path, "1,,turn_left,1,\n")
    assert ch is not None and errors(issues) == []
    assert "too slow to move reliably" in warnings(issues)[0].message


def test_empty_choreography_is_a_warning(tmp_path):
    ch, issues = load(tmp_path, "")
    assert ch is not None and "no moves" in warnings(issues)[0].message


def test_build_choreography_accepts_moves_directly():
    ch, issues = build_choreography([Move(1, 2, 3, StepSpec("turn_left", 90.0))], TT)
    assert issues == [] and ch.schedule[0].first_measure == 2


# ---- The shipped example files ---------------------------------------------------------------------------------
def test_example_choreography_is_valid():
    from dance_robots.timetable import load_timetable
    tt, tt_issues = load_timetable(EXAMPLES / "timetable.csv")
    ch, issues = load_choreography(EXAMPLES / "choreography.csv", tt)
    assert tt_issues == [] and issues == [] and len(ch.schedule) == 8
