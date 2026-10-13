"""Offline tests for TurnLeftStep / TurnRightStep / TurnStep (no ROS needed).

Run from the project root with:   uv run pytest -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dance_robots.dance_step import InfeasibleStepError, RobotLimits
from dance_robots.turn_step import TURN_ANGLE, TurnLeftStep, TurnRightStep, TurnStep

ANGLE = math.radians(30.0)
DT = 5e-4
T_60BPM = 3.0
T_70BPM = 3 * 60.0 / 70.0
DURATIONS = [1.0, 2.1, T_70BPM, T_60BPM, 6.0]
SHAPES = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.2), (0.1, 0.2)]


def sample(step, dt: float = DT):
    """Fine grid including t=0 and t=T. Returns (t, theta, w, alpha, jerk)."""
    n = int(round(step.duration / dt))
    t = np.linspace(0.0, step.duration, n + 1)
    k = np.array([step.get_kinematics(ti) for ti in t])
    return t, k[:, 0], k[:, 1], k[:, 2], k[:, 3]


def trapezoid(y, t) -> float:
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(t)))


def make(T, shape=(0.0, 0.0), angle=ANGLE):
    return TurnStep(T, angle, accel_fraction=shape[0], cruise_fraction=shape[1])


# ---- Defaults and conventions ----------------------------------------------
def test_default_angle_is_30_degrees():
    assert TURN_ANGLE == pytest.approx(0.5235987756)
    assert TurnLeftStep(T_60BPM).angle == pytest.approx(ANGLE)
    assert TurnRightStep(T_60BPM).angle == pytest.approx(-ANGLE)


def test_left_is_ccw_and_right_is_cw():
    """ROS REP-103: left turn = positive angular.z, right turn = negative."""
    assert all(TurnLeftStep(T_60BPM).get_velocity(t)[1] >= 0.0
               for t in np.linspace(0, T_60BPM, 200))
    assert all(TurnRightStep(T_60BPM).get_velocity(t)[1] <= 0.0
               for t in np.linspace(0, T_60BPM, 200))


def test_turn_in_place_has_no_linear_velocity():
    step = TurnLeftStep(T_60BPM)
    assert all(step.get_velocity(t)[0] == 0.0 for t in np.linspace(0, T_60BPM, 200))


# ---- Correctness of the motion ------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_angular_velocity_integrates_to_target_angle(T, shape):
    """Integrating the commanded angular.z must give exactly 30 degrees."""
    step = make(T, shape)
    n = int(round(T / DT))
    t = np.linspace(0.0, T, n + 1)
    w = np.array([step.get_velocity(ti)[1] for ti in t])
    assert trapezoid(w, t) == pytest.approx(ANGLE, abs=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_angle_is_consistent_with_angular_velocity(T, shape):
    step = make(T, shape)
    t, th, w, _, _ = sample(step)
    th_from_w = np.concatenate(([0.0], np.cumsum((w[1:] + w[:-1]) / 2.0 * np.diff(t))))
    assert np.max(np.abs(th - th_from_w)) < 1e-5


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_starts_and_ends_at_rest(T, shape):
    step = make(T, shape)
    th0, w0, a0, _ = step.get_kinematics(0.0)
    th1, w1, a1, _ = step.get_kinematics(T - 1e-9)
    assert (th0, w0, a0) == (0.0, 0.0, 0.0)
    assert th1 == pytest.approx(ANGLE, abs=1e-6)
    assert w1 == pytest.approx(0.0, abs=1e-6)
    assert a1 == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_no_jumps_and_symmetric(T, shape):
    step = make(T, shape)
    _, _, w, a, _ = sample(step)
    assert np.max(np.abs(np.diff(w))) <= step.peak_angular_acceleration * DT * 1.01
    assert np.max(np.abs(np.diff(a))) <= step.peak_angular_jerk * DT * 1.01
    assert np.allclose(w, w[::-1], atol=1e-9)


# ---- Limits ------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_limits_respected_and_closed_form_peaks_match(T, shape):
    step = make(T, shape)
    lim = step.limits
    _, _, w, a, j = sample(step)
    assert np.max(np.abs(w)) <= lim.w_max
    assert np.max(np.abs(a)) <= lim.alpha_max
    assert np.max(np.abs(j)) <= lim.jerk_ang_max
    assert np.max(np.abs(w)) == pytest.approx(step.peak_angular_velocity, rel=2e-3)
    assert np.max(np.abs(a)) == pytest.approx(step.peak_angular_acceleration, rel=2e-3)
    assert np.max(np.abs(j)) == pytest.approx(step.peak_angular_jerk, rel=2e-3)


@pytest.mark.parametrize("T", DURATIONS)
def test_wheel_speeds_stay_within_linear_limit(T):
    """Independent hardware-level check: in a pure rotation the wheels move at
    +-w*b/2. Computed from the actual commands, that must stay below v_max."""
    step = TurnLeftStep(T)
    b = step.limits.track_width
    t = np.linspace(0.0, T, int(T / DT) + 1)
    w = np.array([step.get_velocity(ti)[1] for ti in t])
    assert np.max(np.abs(w) * b / 2.0) <= step.limits.v_max


# ---- Feasibility -----------------------------------------------------------------------
def test_min_duration_of_default_turn():
    """Default shape is limited by angular acceleration: sqrt(8*theta/alpha_max)."""
    expected = math.sqrt(8 * ANGLE / RobotLimits().alpha_max)  # ~0.906 s
    assert TurnLeftStep(T_60BPM).min_duration() == pytest.approx(expected, rel=1e-9)
    assert expected == pytest.approx(0.906, abs=1e-3)


def test_waltz_tempos_are_feasible_and_too_short_is_rejected():
    TurnLeftStep(T_60BPM)
    TurnRightStep(T_70BPM)
    with pytest.raises(InfeasibleStepError):
        TurnLeftStep(0.5)


def test_exactly_min_duration_is_accepted_and_hits_the_limit():
    t_min = TurnLeftStep(T_60BPM).min_duration()
    step = TurnLeftStep(t_min)
    assert step.peak_angular_acceleration == pytest.approx(step.limits.alpha_max, rel=1e-9)


def test_stricter_limits_raise_min_duration():
    slow = RobotLimits(w_max=0.2, a_max=0.05, j_max=0.2)
    with pytest.raises(InfeasibleStepError):
        TurnLeftStep(T_60BPM, limits=slow)
    TurnLeftStep(10.0, limits=slow)


# ---- Boundaries, timer, validation -------------------------------------------------------
@pytest.mark.parametrize("t", [-5.0, -1e-9, T_60BPM, T_60BPM + 1e-9, 100.0])
def test_zero_command_outside_window(t):
    assert TurnLeftStep(T_60BPM).get_velocity(t) == (0.0, 0.0)


@pytest.mark.parametrize("period", [0.02, 0.05])   # 50 Hz and 20 Hz timers
@pytest.mark.parametrize("T", [3.0, T_70BPM, 2.61, 2.99])
def test_timer_sampling_at_20_and_50_hz(T, period):
    """Zero command from T onwards and ~30 degrees of discrete angle, for both
    candidate node rates."""
    step = TurnLeftStep(T)
    ticks = np.arange(int(math.ceil(T / period)) + 5) * period
    w = np.array([step.get_velocity(t)[1] for t in ticks])
    assert np.all(w[ticks >= T] == 0.0)
    assert np.sum(w) * period == pytest.approx(ANGLE, abs=2e-3)  # 2 mrad ~ 0.11 deg


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_invalid_duration_is_rejected(bad):
    with pytest.raises(ValueError):
        TurnLeftStep(bad)


@pytest.mark.parametrize("bad", [0.0, -0.5, math.nan, math.inf])
def test_invalid_angle_is_rejected(bad):
    with pytest.raises(ValueError):
        TurnLeftStep(T_60BPM, angle=bad)
    with pytest.raises(ValueError):
        TurnRightStep(T_60BPM, angle=bad)


def test_right_turn_is_exact_mirror_of_left_turn():
    _, thl, wl, al, jl = sample(TurnLeftStep(T_60BPM))
    _, thr, wr, ar, jr = sample(TurnRightStep(T_60BPM))
    assert np.allclose(thr, -thl) and np.allclose(wr, -wl)
    assert np.allclose(ar, -al) and np.allclose(jr, -jl)
