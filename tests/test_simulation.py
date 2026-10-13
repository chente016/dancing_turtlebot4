"""Tests for the offline timer simulation (and, through it, whole dances)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dance_robots.choreography import load_choreography
from dance_robots.simulation import simulate
from dance_robots.timetable import Timetable, load_timetable

EXAMPLES = Path(__file__).parent.parent / "examples"
HEADER = "first_measure,last_measure,step,angle_deg,radius_m\n"
TT = Timetable.from_tempo_map([(6, 60), (4, 70)])
RATES = [0.02, 0.05]  # 50 Hz and 20 Hz


def dance(tmp_path, body, **kwargs):
    path = tmp_path / "c.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    ch, issues = load_choreography(path, TT, **kwargs)
    assert ch is not None, [str(i) for i in issues]
    return ch


@pytest.mark.parametrize("period", RATES)
def test_forward_moves_30_cm_along_x(tmp_path, period):
    r = simulate(dance(tmp_path, "1,,forward,,\n"), period)
    x, y, th = r.pose_at(r.t[-1])
    assert (x, y, th) == pytest.approx((0.3, 0.0, 0.0), abs=1e-3)


@pytest.mark.parametrize("period", RATES)
def test_backward_returns_behind_the_start(tmp_path, period):
    r = simulate(dance(tmp_path, "1,,backward,,\n"), period)
    assert r.pose_at(r.t[-1])[0] == pytest.approx(-0.3, abs=1e-3)


@pytest.mark.parametrize("period", RATES)
def test_forward_then_backward_ends_where_it_started(tmp_path, period):
    r = simulate(dance(tmp_path, "1,,forward,,\n2,,backward,,\n"), period)
    assert r.pose_at(r.t[-1])[:2] == pytest.approx((0.0, 0.0), abs=2e-3)


@pytest.mark.parametrize("period", RATES)
@pytest.mark.parametrize("angle", [1.0, 30.0, 90.0])
def test_turns_end_at_the_requested_heading(tmp_path, period, angle):
    r = simulate(dance(tmp_path, f"1,,turn_left,{angle},\n"), period)
    x, y, th = r.pose_at(r.t[-1])
    assert th == pytest.approx(math.radians(angle), abs=2e-3)
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)        # in place


@pytest.mark.parametrize("period", RATES)
def test_right_turn_is_negative_heading(tmp_path, period):
    r = simulate(dance(tmp_path, "1,,turn_right,45,\n"), period)
    assert r.pose_at(r.t[-1])[2] == pytest.approx(-math.pi / 4, abs=2e-3)


@pytest.mark.parametrize("period", RATES)
@pytest.mark.parametrize("radius", [0.1, 0.25, 0.4])
def test_circle_ends_at_the_expected_pose(tmp_path, period, radius):
    ang = 30.0
    r = simulate(dance(tmp_path, f"1,,circle_left,{ang},{radius}\n"), period)
    a = math.radians(ang)
    x, y, th = r.pose_at(r.t[-1])
    assert (x, y, th) == pytest.approx(
        (radius * math.sin(a), radius * (1 - math.cos(a)), a), abs=2e-3)


@pytest.mark.parametrize("period", RATES)
def test_the_robot_is_stopped_after_the_dance(tmp_path, period):
    ch = dance(tmp_path, "1,,forward,,\n2,,turn_left,90,\n")
    r = simulate(ch, period, tail=1.0)
    after = r.tick_t >= ch.node_end_s
    assert after.any() and np.all(r.tick_v[after] == 0.0) and np.all(r.tick_w[after] == 0.0)


def test_example_dance_final_pose_matches_hand_calculation():
    """Worked out by hand from the arc geometry (see README): the example dance
    ends at x = 0.3172 m, y = 0.6364 m with heading 120 degrees."""
    tt, _ = load_timetable(EXAMPLES / "timetable.csv")
    ch, _ = load_choreography(EXAMPLES / "choreography.csv", tt)
    # forward (0.3, 0) -> turn 90 -> forward (0.3, 0.3) -> circle_right 45 deg, R=0.4:
    #   centre at (0.7, 0.3); end = centre + 0.4 * (-cos45, +sin45) = (0.4172, 0.5828)
    # -> turn_right 45 (heading 0) -> backward 0.3 -> (0.1172, 0.5828)
    # -> circle_left 30 deg, R=0.4: + (0.4 sin30, 0.4 (1 - cos30)) = (0.3172, 0.6364), heading 30
    # -> turn_left 90 -> heading 120
    for period in RATES:
        r = simulate(ch, period)
        x, y, th = r.pose_at(r.t[-1])
        assert x == pytest.approx(0.1172 + 0.2, abs=3e-3)
        assert y == pytest.approx(0.5828 + 0.4 * (1 - math.cos(math.radians(30))), abs=3e-3)
        assert th == pytest.approx(math.radians(120), abs=3e-3)


def test_20_and_50_hz_give_the_same_dance(tmp_path):
    """The node rate hardly matters: both timers end within 2 mm of each other."""
    tt, _ = load_timetable(EXAMPLES / "timetable.csv")
    ch, _ = load_choreography(EXAMPLES / "choreography.csv", tt)
    p20 = np.array(simulate(ch, 0.05).pose_at(1e9))
    p50 = np.array(simulate(ch, 0.02).pose_at(1e9))
    assert np.max(np.abs(p20 - p50)) < 2e-3


def test_commands_never_exceed_the_robot_limits():
    tt, _ = load_timetable(EXAMPLES / "timetable.csv")
    ch, _ = load_choreography(EXAMPLES / "choreography.csv", tt)
    r = simulate(ch, 0.02)
    b = 0.235
    assert np.max(np.abs(r.tick_v)) <= 0.306
    assert np.max(np.abs(r.tick_w)) <= 1.9
    wheels = np.concatenate([r.tick_v + r.tick_w * b / 2, r.tick_v - r.tick_w * b / 2])
    assert np.max(np.abs(wheels)) <= 0.306


def test_timer_offset_shifts_the_whole_dance(tmp_path):
    a = simulate(dance(tmp_path, "1,,forward,,\n"), 0.02)
    b = simulate(dance(tmp_path, "1,,forward,,\n", music_start_offset=1.0), 0.02)
    assert b.pose_at(1.0 + 1.5)[0] == pytest.approx(a.pose_at(1.5)[0], abs=1e-3)
    assert b.pose_at(0.9)[0] == 0.0
