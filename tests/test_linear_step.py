"""Offline tests for ForwardStep / LinearStep (pure Python, no ROS needed).

Run from the project root with:   uv run pytest -v

What we verify (see the test-plan from the design discussion):
  * the velocity profile integrates to the target distance,
  * the move starts and ends at rest (v = 0 and a = 0, no jumps),
  * v, a and jerk stay under the robot limits,
  * 50 Hz timer sampling behaves, even when T is not a multiple of 0.02 s,
  * infeasible / invalid inputs are rejected,
  * plots of x, v, a, j are saved so a human can eyeball the shape.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: lets us save PNGs without a display
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dance_robots.dance_step import InfeasibleStepError, RobotLimits  # noqa: E402
from dance_robots.linear_step import ForwardStep, LinearStep  # noqa: E402

DISTANCE = 0.3   # [m] handover target for forward/backward steps
DT = 5e-4        # [s] fine time grid for the numerical checks

# Measure durations of the 3/4 waltz: 3 beats per measure.
T_60BPM = 3 * 60.0 / 60.0   # 3.000 s
T_70BPM = 3 * 60.0 / 70.0   # 2.571 s

# A few feasible durations (all above the ~2.0 s minimum of the default shape).
DURATIONS = [2.1, T_70BPM, T_60BPM, 4.0, 6.0]

# Profile shapes as (accel_fraction, cruise_fraction). The first one is the
# default pure-jerk shape; the others have a constant-accel plateau and/or a
# cruise phase, so every one of the 7 segments gets exercised.
SHAPES = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.2), (0.1, 0.2)]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def sample(step: LinearStep, dt: float = DT):
    """Sample the step on a fine grid that includes BOTH t=0 and t=T exactly.

    Returns (t, x, v, a, j) as numpy arrays.
    """
    n = int(round(step.duration / dt))
    t = np.linspace(0.0, step.duration, n + 1)
    k = np.array([step.get_kinematics(ti) for ti in t])  # shape (n+1, 4)
    return t, k[:, 0], k[:, 1], k[:, 2], k[:, 3]


def trapezoid(y: np.ndarray, t: np.ndarray) -> float:
    """Trapezoidal integral (written by hand to avoid numpy-version differences)."""
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(t)))


# ----------------------------------------------------------------------------
# Correctness of the motion
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_velocity_integrates_to_target_distance(T, shape):
    """Integrating the commanded velocity over [0, T] must give exactly 0.3 m.

    This uses get_velocity (what the robot actually receives), not the
    internal position, so it checks the real output.
    """
    step = LinearStep(T, DISTANCE, accel_fraction=shape[0], cruise_fraction=shape[1])
    n = int(round(T / DT))
    t = np.linspace(0.0, T, n + 1)
    v = np.array([step.get_velocity(ti)[0] for ti in t])
    assert trapezoid(v, t) == pytest.approx(DISTANCE, abs=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_position_is_consistent_with_velocity(T, shape):
    """The position from get_kinematics must equal the running integral of v.

    This cross-checks the segment-by-segment polynomial math: if any segment's
    starting state were wrong, position and integrated velocity would diverge.
    """
    step = LinearStep(T, DISTANCE, accel_fraction=shape[0], cruise_fraction=shape[1])
    t, x, v, _, _ = sample(step)
    # Cumulative trapezoid: running integral of v at every grid point.
    x_from_v = np.concatenate(([0.0], np.cumsum((v[1:] + v[:-1]) / 2.0 * np.diff(t))))
    assert np.max(np.abs(x - x_from_v)) < 1e-5


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_starts_and_ends_at_rest(T, shape):
    """v = 0 and a = 0 at both ends, and the final position is the target."""
    step = LinearStep(T, DISTANCE, accel_fraction=shape[0], cruise_fraction=shape[1])
    eps = 1e-9  # just inside the window: get_velocity is clamped to 0 at t >= T

    x0, v0, a0, _ = step.get_kinematics(0.0)
    x1, v1, a1, _ = step.get_kinematics(T - eps)

    assert (x0, v0, a0) == (0.0, 0.0, 0.0)
    assert x1 == pytest.approx(DISTANCE, abs=1e-6)
    assert v1 == pytest.approx(0.0, abs=1e-6)
    assert a1 == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_no_jumps_in_velocity_or_acceleration(T, shape):
    """Velocity AND acceleration must be continuous (smooth S-curve).

    With bounded jerk, between two samples dt apart the acceleration can change
    by at most j_peak*dt, and the velocity by at most a_peak*dt. A jump (for
    example a mis-built segment) would exceed those bounds by a wide margin.
    """
    step = LinearStep(T, DISTANCE, accel_fraction=shape[0], cruise_fraction=shape[1])
    _, _, v, a, _ = sample(step)
    assert np.max(np.abs(np.diff(v))) <= step.peak_acceleration * DT * 1.01
    assert np.max(np.abs(np.diff(a))) <= step.peak_jerk * DT * 1.01


@pytest.mark.parametrize("T", DURATIONS)
def test_forward_step_never_reverses(T):
    """A forward move must never command negative velocity."""
    _, _, v, _, _ = sample(ForwardStep(T))
    assert np.all(v >= -1e-12)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_velocity_profile_is_symmetric(T, shape):
    """Accelerating and decelerating halves mirror each other: v(t) = v(T - t)."""
    step = LinearStep(T, DISTANCE, accel_fraction=shape[0], cruise_fraction=shape[1])
    _, _, v, _, _ = sample(step)
    assert np.allclose(v, v[::-1], atol=1e-9)


# ----------------------------------------------------------------------------
# Limits
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_limits_are_respected(T, shape):
    """Numerical peaks of v, a, jerk must stay under the RobotLimits."""
    step = LinearStep(T, DISTANCE, accel_fraction=shape[0], cruise_fraction=shape[1])
    lim = step.limits
    _, _, v, a, j = sample(step)

    assert np.max(np.abs(v)) <= lim.v_max
    assert np.max(np.abs(a)) <= lim.a_max
    assert np.max(np.abs(j)) <= lim.j_max


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_closed_form_peaks_match_numerical_peaks(T, shape):
    """peak_velocity / peak_acceleration / peak_jerk (closed form) must agree
    with the maxima found by sampling. They are what min_duration() relies on."""
    step = LinearStep(T, DISTANCE, accel_fraction=shape[0], cruise_fraction=shape[1])
    _, _, v, a, j = sample(step)

    # Small tolerance: the sampled grid may miss the exact peak by a hair.
    assert np.max(np.abs(v)) == pytest.approx(step.peak_velocity, rel=2e-3)
    assert np.max(np.abs(a)) == pytest.approx(step.peak_acceleration, rel=2e-3)
    assert np.max(np.abs(j)) == pytest.approx(step.peak_jerk, rel=2e-3)


# ----------------------------------------------------------------------------
# Feasibility
# ----------------------------------------------------------------------------
def test_min_duration_of_default_forward_step():
    """Default shape is limited by acceleration: T_min = sqrt(8*D/a_max) = 2.0 s."""
    assert ForwardStep(T_60BPM).min_duration() == pytest.approx(2.0, abs=1e-9)


def test_waltz_tempos_are_feasible():
    """60-70 BPM waltz measures (3.0 s and 2.57 s) must be accepted."""
    ForwardStep(T_60BPM)
    ForwardStep(T_70BPM)


def test_too_short_duration_is_rejected():
    with pytest.raises(InfeasibleStepError):
        ForwardStep(1.5)


def test_exactly_min_duration_is_accepted_and_hits_the_limit():
    """At T = T_min the binding limit is reached exactly, but not exceeded."""
    t_min = ForwardStep(T_60BPM).min_duration()
    step = ForwardStep(t_min)  # must NOT raise
    assert step.peak_acceleration == pytest.approx(step.limits.a_max, rel=1e-9)
    assert step.peak_velocity <= step.limits.v_max
    assert step.peak_jerk <= step.limits.j_max


def test_stricter_limits_raise_min_duration():
    """Custom RobotLimits are honoured: a slower robot needs a longer window."""
    slow = RobotLimits(v_max=0.1, a_max=0.2, j_max=1.0)
    # With these limits the 60 BPM measure (3.0 s) is no longer enough...
    with pytest.raises(InfeasibleStepError):
        ForwardStep(T_60BPM, limits=slow)
    # ...but a long enough window is.
    ForwardStep(10.0, limits=slow)


def test_accel_plateau_lowers_min_duration():
    """A constant-acceleration segment makes the acceleration phase more
    efficient (flat top instead of a triangle), so for the same T the peak
    acceleration drops and a shorter window becomes feasible."""
    pure = LinearStep(T_60BPM, DISTANCE).min_duration()
    plateau = LinearStep(T_60BPM, DISTANCE, accel_fraction=0.1).min_duration()
    assert plateau < pure


def test_cruise_phase_raises_min_duration_but_lowers_peak_velocity():
    """A cruise segment takes time away from the acceleration phases, so they
    must be steeper: the minimum duration goes UP, while the peak velocity goes
    DOWN (the velocity plateau is flatter). It is a smoothness/style knob, not
    a way to fit shorter measures."""
    pure = LinearStep(T_60BPM, DISTANCE)
    cruise = LinearStep(T_60BPM, DISTANCE, cruise_fraction=0.3)
    assert cruise.min_duration() > pure.min_duration()
    assert cruise.peak_velocity < pure.peak_velocity


# ----------------------------------------------------------------------------
# Boundaries and the 50 Hz timer
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("t", [-5.0, -1e-9, T_60BPM, T_60BPM + 1e-9, 100.0])
def test_zero_velocity_outside_window(t):
    """Before the start and at/after T the robot must be commanded to stop."""
    assert ForwardStep(T_60BPM).get_velocity(t) == (0.0, 0.0)


def test_angular_velocity_is_always_zero():
    step = ForwardStep(T_60BPM)
    assert all(step.get_velocity(t)[1] == 0.0 for t in np.linspace(0, T_60BPM, 200))


@pytest.mark.parametrize("T", [3.0, T_70BPM, 2.61, 2.99])
def test_50hz_timer_sampling(T):
    """Simulate the ROS timer: one command every 0.02 s, T NOT always a multiple
    of 0.02. After T every command must be zero, and the discrete sum of
    v * 0.02 (what the robot would roughly travel) must be close to 0.3 m."""
    period = 0.02
    step = ForwardStep(T)

    # Run a few ticks beyond the end, as the real node would.
    ticks = np.arange(int(math.ceil(T / period)) + 5) * period
    cmds = np.array([step.get_velocity(t)[0] for t in ticks])

    assert np.all(cmds[ticks >= T] == 0.0)   # stopped from T onwards
    assert np.any(cmds[ticks < T] > 0.0)     # and it did move before that
    assert np.sum(cmds) * period == pytest.approx(DISTANCE, abs=1e-3)


# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_invalid_duration_is_rejected(bad):
    with pytest.raises(ValueError):
        ForwardStep(bad)


@pytest.mark.parametrize("bad", [0.0, -0.3, math.nan])
def test_invalid_forward_distance_is_rejected(bad):
    with pytest.raises(ValueError):
        ForwardStep(T_60BPM, distance=bad)


@pytest.mark.parametrize("shape", [(-0.1, 0.0), (0.0, -0.1), (0.5, 0.0), (0.2, 0.6)])
def test_invalid_shape_is_rejected(shape):
    """Fractions must be >= 0 and satisfy 2*accel + cruise < 1."""
    with pytest.raises(ValueError):
        LinearStep(T_60BPM, DISTANCE, accel_fraction=shape[0], cruise_fraction=shape[1])


def test_negative_distance_mirrors_the_profile():
    """Preview of the backward step: a negative signed distance should give the
    exact mirror image of the forward profile and end at -0.3 m."""
    fwd = LinearStep(T_60BPM, +DISTANCE)
    bwd = LinearStep(T_60BPM, -DISTANCE)
    _, x_f, v_f, _, _ = sample(fwd)
    _, x_b, v_b, _, _ = sample(bwd)
    assert np.allclose(v_b, -v_f)
    assert x_b[-1] == pytest.approx(-DISTANCE, abs=1e-9)


# ----------------------------------------------------------------------------
# Plots (for humans)
# ----------------------------------------------------------------------------
def test_save_profile_plots():
    """Save position / velocity / acceleration / jerk plots to tests/output/.

    Dashed red lines mark the robot limits. Open the PNG and check that the
    curves look smooth and symmetric. This test only fails if saving fails.
    """
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    labels = ["Position [m]", "Velocity [m/s]", "Acceleration [m/s²]", "Jerk [m/s³]"]

    for T, name in [(T_60BPM, "60 BPM (3.00 s)"), (T_70BPM, "70 BPM (2.57 s)")]:
        step = ForwardStep(T)
        t, x, v, a, j = sample(step)
        for ax, y in zip(axes, (x, v, a, j)):
            ax.plot(t, y, label=name)

    lim = RobotLimits()
    for ax, limit in zip(axes[1:], (lim.v_max, lim.a_max, lim.j_max)):
        ax.axhline(limit, color="red", linestyle="--", linewidth=0.8)  # limit line

    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("ForwardStep (0.3 m) - S-curve profiles")
    axes[0].legend()
    axes[-1].set_xlabel("Time [s]")

    fig.tight_layout()
    path = out_dir / "forward_step_profiles.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    assert path.exists()
