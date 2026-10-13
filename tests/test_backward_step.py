"""Offline tests for BackwardStep (pure Python, no ROS needed).

The heavy math is already covered in test_linear_step.py (BackwardStep reuses
it with a negative sign). Here we check what is specific to going backwards:
sign, mirror symmetry with ForwardStep, limits, and forward+backward combos.

Run from the project root with:   uv run pytest -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dance_robots.dance_step import InfeasibleStepError, RobotLimits
from dance_robots.linear_step import BackwardStep, ForwardStep

DISTANCE = 0.3
DT = 5e-4
T_60BPM = 3.0
T_70BPM = 3 * 60.0 / 70.0
DURATIONS = [2.1, T_70BPM, T_60BPM, 4.0, 6.0]


def sample(step, dt: float = DT):
    """Sample on a fine grid including t=0 and t=T. Returns (t, x, v, a, j)."""
    n = int(round(step.duration / dt))
    t = np.linspace(0.0, step.duration, n + 1)
    k = np.array([step.get_kinematics(ti) for ti in t])
    return t, k[:, 0], k[:, 1], k[:, 2], k[:, 3]


@pytest.mark.parametrize("T", DURATIONS)
def test_backward_integrates_to_minus_target_distance(T):
    """The commanded velocity must integrate to -0.3 m (not +0.3)."""
    step = BackwardStep(T)
    n = int(round(T / DT))
    t = np.linspace(0.0, T, n + 1)
    v = np.array([step.get_velocity(ti)[0] for ti in t])
    total = float(np.sum((v[1:] + v[:-1]) / 2.0 * np.diff(t)))
    assert total == pytest.approx(-DISTANCE, abs=1e-5)


@pytest.mark.parametrize("T", DURATIONS)
def test_backward_never_goes_forward(T):
    """linear.x must be <= 0 for the whole move, and angular.z always 0."""
    step = BackwardStep(T)
    _, _, v, _, _ = sample(step)
    assert np.all(v <= 1e-12)
    assert all(step.get_velocity(t)[1] == 0.0 for t in np.linspace(0, T, 100))


@pytest.mark.parametrize("T", DURATIONS)
def test_backward_is_exact_mirror_of_forward(T):
    """Same shape as ForwardStep with the opposite sign, at every sample:
    x, v, a and jerk are all negated."""
    _, xf, vf, af, jf = sample(ForwardStep(T))
    _, xb, vb, ab, jb = sample(BackwardStep(T))
    assert np.allclose(xb, -xf)
    assert np.allclose(vb, -vf)
    assert np.allclose(ab, -af)
    assert np.allclose(jb, -jf)


@pytest.mark.parametrize("T", DURATIONS)
def test_backward_starts_and_ends_at_rest(T):
    step = BackwardStep(T)
    eps = 1e-9
    assert step.get_kinematics(0.0) == (0.0, 0.0, 0.0, 0.0)
    x, v, a, _ = step.get_kinematics(T - eps)
    assert x == pytest.approx(-DISTANCE, abs=1e-6)
    assert v == pytest.approx(0.0, abs=1e-6)
    assert a == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("T", DURATIONS)
def test_backward_limits_use_absolute_values(T):
    """Limits apply to |v|, |a|, |j|: a negative move must not slip past them."""
    step = BackwardStep(T)
    lim = step.limits
    _, _, v, a, j = sample(step)
    assert np.max(np.abs(v)) <= lim.v_max
    assert np.max(np.abs(a)) <= lim.a_max
    assert np.max(np.abs(j)) <= lim.j_max
    # The peak_* properties are magnitudes, so they must be positive here too.
    assert step.peak_velocity > 0.0
    assert step.peak_acceleration > 0.0
    assert step.peak_jerk > 0.0


def test_backward_has_same_min_duration_as_forward():
    assert BackwardStep(T_60BPM).min_duration() == pytest.approx(
        ForwardStep(T_60BPM).min_duration()
    )


def test_backward_feasibility():
    BackwardStep(T_60BPM)
    BackwardStep(T_70BPM)
    with pytest.raises(InfeasibleStepError):
        BackwardStep(1.5)
    with pytest.raises(InfeasibleStepError):
        BackwardStep(T_60BPM, limits=RobotLimits(v_max=0.1, a_max=0.2, j_max=1.0))


@pytest.mark.parametrize("t", [-5.0, -1e-9, T_60BPM, T_60BPM + 1e-9, 100.0])
def test_backward_zero_velocity_outside_window(t):
    assert BackwardStep(T_60BPM).get_velocity(t) == (0.0, 0.0)


@pytest.mark.parametrize("bad", [0.0, -0.3, math.nan, math.inf])
def test_backward_rejects_invalid_distance(bad):
    """The distance is a POSITIVE magnitude; a negative value is an error."""
    with pytest.raises(ValueError):
        BackwardStep(T_60BPM, distance=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_backward_rejects_invalid_duration(bad):
    with pytest.raises(ValueError):
        BackwardStep(bad)


@pytest.mark.parametrize("T_f, T_b", [(T_60BPM, T_60BPM), (T_70BPM, T_60BPM), (2.5, 4.0)])
def test_forward_then_backward_returns_to_start(T_f, T_b):
    """Chain forward + backward as the sequencer will (each step uses its own
    LOCAL time) and simulate the 50 Hz timer. The robot must come back to
    where it started: net displacement ~ 0, and it is stopped between moves."""
    period = 0.02
    fwd, bwd = ForwardStep(T_f), BackwardStep(T_b)

    # Absolute start time of each step (here: back-to-back, no gap).
    schedule = [(0.0, fwd), (T_f, bwd)]
    total = T_f + T_b

    position = 0.0
    for k in range(int(math.ceil(total / period)) + 5):
        t_abs = k * period
        # Pick the step whose window contains t_abs (the sequencer's job).
        v = 0.0
        for start, step in schedule:
            if start <= t_abs < start + step.duration:
                v = step.get_velocity(t_abs - start)[0]
        position += v * period

    # Open-loop sum of discrete commands: allow 2 mm of discretisation error.
    assert position == pytest.approx(0.0, abs=2e-3)
