"""Offline tests for CircleLeftStep / CircleRightStep / CircleStep (no ROS needed).

Run from the project root with:   uv run pytest -v
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: save PNGs without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dance_robots.circle_step import (  # noqa: E402
    CIRCLE_RADIUS, CircleLeftStep, CircleRightStep, CircleStep)
from dance_robots.dance_step import InfeasibleStepError, RobotLimits  # noqa: E402

ANGLE = math.radians(30.0)
DT = 5e-4
T_60BPM = 3.0
T_70BPM = 3 * 60.0 / 70.0
DURATIONS = [2.2, T_70BPM, T_60BPM, 6.0]
RADII = [0.3, 0.4, 0.5]
SHAPES = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.2), (0.1, 0.2)]
B = RobotLimits().track_width


def sample(step, dt: float = DT):
    """Fine grid including t=0 and t=T. Returns (t, s, v, a, j) along the arc."""
    n = int(round(step.duration / dt))
    t = np.linspace(0.0, step.duration, n + 1)
    k = np.array([step.get_kinematics(ti) for ti in t])
    return t, k[:, 0], k[:, 1], k[:, 2], k[:, 3]


def commands(step, dt: float = DT):
    """Fine grid of the actual (linear.x, angular.z) commands."""
    n = int(round(step.duration / dt))
    t = np.linspace(0.0, step.duration, n + 1)
    c = np.array([step.get_velocity(ti) for ti in t])
    return t, c[:, 0], c[:, 1]


def dead_reckon(step, n: int = 6000):
    """Integrate the commands with a unicycle model, as the robot would move
    in an ideal world (no slip, no latency). Midpoint rule for accuracy.
    Returns arrays x, y, theta of length n+1."""
    dt = step.duration / n
    x = np.zeros(n + 1); y = np.zeros(n + 1); th = np.zeros(n + 1)
    for i in range(n):
        v, w = step.get_velocity((i + 0.5) * dt)
        mid = th[i] + w * dt / 2.0
        x[i + 1] = x[i] + v * math.cos(mid) * dt
        y[i + 1] = y[i] + v * math.sin(mid) * dt
        th[i + 1] = th[i] + w * dt
    return x, y, th


# ---- Geometry: does the robot actually drive the right arc? -------------------------
@pytest.mark.parametrize("R", RADII)
@pytest.mark.parametrize("T", DURATIONS)
def test_left_arc_ends_at_the_expected_pose(T, R):
    """Starting at the origin facing +x, a left arc of radius R through 30 degrees
    must end at (R sin(a), R(1 - cos(a))) with heading +a."""
    x, y, th = dead_reckon(CircleLeftStep(T, radius=R))
    assert x[-1] == pytest.approx(R * math.sin(ANGLE), abs=1e-4)
    assert y[-1] == pytest.approx(R * (1 - math.cos(ANGLE)), abs=1e-4)
    assert th[-1] == pytest.approx(ANGLE, abs=1e-4)


@pytest.mark.parametrize("R", RADII)
def test_right_arc_is_mirror_image_of_left_arc(R):
    xl, yl, thl = dead_reckon(CircleLeftStep(T_60BPM, radius=R))
    xr, yr, thr = dead_reckon(CircleRightStep(T_60BPM, radius=R))
    assert np.allclose(xr, xl, atol=1e-9)
    assert np.allclose(yr, -yl, atol=1e-9)
    assert np.allclose(thr, -thl, atol=1e-9)


@pytest.mark.parametrize("R", RADII)
def test_path_stays_on_the_circle(R):
    """Every point of the path is at distance R from the circle centre (0, R)."""
    x, y, _ = dead_reckon(CircleLeftStep(T_60BPM, radius=R))
    assert np.allclose(np.hypot(x, y - R), R, atol=1e-4)


@pytest.mark.parametrize("T", DURATIONS)
def test_linear_and_angular_are_synchronised(T):
    """w(t) = v(t) / R at every instant, so peaks and endpoints align."""
    step = CircleLeftStep(T)
    _, v, w = commands(step)
    assert np.allclose(w, v / CIRCLE_RADIUS, atol=1e-12)
    assert np.argmax(v) == np.argmax(w)


def test_signs_left_and_right():
    left, right = CircleLeftStep(T_60BPM), CircleRightStep(T_60BPM)
    _, vl, wl = commands(left)
    _, vr, wr = commands(right)
    assert np.all(vl >= 0) and np.all(vr >= 0)          # always drives forward
    assert np.all(wl >= 0) and np.all(wr <= 0)          # left = CCW, right = CW


# ---- Correctness of the profile --------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_arc_length_and_rest_conditions(T, shape):
    step = CircleStep(T, ANGLE, accel_fraction=shape[0], cruise_fraction=shape[1])
    s_end, v_end, a_end, _ = step.get_kinematics(T - 1e-9)
    assert step.get_kinematics(0.0) == (0.0, 0.0, 0.0, 0.0)
    assert s_end == pytest.approx(CIRCLE_RADIUS * ANGLE, abs=1e-6)       # R * theta (default radius)
    assert v_end == pytest.approx(0.0, abs=1e-6)
    assert a_end == pytest.approx(0.0, abs=1e-6)

    t, v = commands(step)[0], commands(step)[1]
    n = int(round(T / DT))
    assert float(np.sum((v[1:] + v[:-1]) / 2 * np.diff(t))) == pytest.approx(CIRCLE_RADIUS * ANGLE, abs=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("T", DURATIONS)
def test_no_jumps_and_symmetric(T, shape):
    step = CircleStep(T, ANGLE, accel_fraction=shape[0], cruise_fraction=shape[1])
    _, _, v, a, _ = sample(step)
    assert np.max(np.abs(np.diff(v))) <= step.peak_acceleration * DT * 1.01
    assert np.max(np.abs(np.diff(a))) <= step.peak_jerk * DT * 1.01
    assert np.allclose(v, v[::-1], atol=1e-9)


# ---- Wheel-level limits --------------------------------------------------------------------
@pytest.mark.parametrize("R", RADII)
@pytest.mark.parametrize("T", DURATIONS)
def test_both_wheels_respect_limits(T, R):
    """Independent check from the actual commands: v_right/left = v +- w*b/2.
    Each wheel's speed and acceleration must stay within v_max / a_max, and the
    body angular velocity within the firmware cap w_max."""
    step = CircleLeftStep(T, radius=R)
    lim = step.limits
    t, v, w = commands(step)
    v_right, v_left = v + w * B / 2.0, v - w * B / 2.0
    for wheel in (v_right, v_left):
        assert np.max(np.abs(wheel)) <= lim.v_max
        assert np.max(np.abs(np.diff(wheel) / np.diff(t))) <= lim.a_max * 1.001
    assert np.max(np.abs(w)) <= lim.w_max


@pytest.mark.parametrize("R", RADII)
def test_closed_form_peaks_match_numerical(R):
    step = CircleLeftStep(T_70BPM, radius=R)
    _, _, v, a, j = sample(step)
    assert np.max(v) == pytest.approx(step.peak_velocity, rel=2e-3)
    assert np.max(np.abs(a)) == pytest.approx(step.peak_acceleration, rel=2e-3)
    assert np.max(np.abs(j)) == pytest.approx(step.peak_jerk, rel=2e-3)
    assert step.peak_angular_velocity == pytest.approx(step.peak_velocity / R)
    assert step.wheel_factor == pytest.approx(1.0 + B / (2.0 * R))


# ---- Feasibility -----------------------------------------------------------------------------
def test_min_duration_values():
    """Reference values (track 0.235 m, wheel-level limits)."""
    assert CircleLeftStep(T_60BPM, radius=0.5).min_duration() == pytest.approx(2.113, abs=2e-3)
    assert CircleLeftStep(T_60BPM, radius=0.4).min_duration() == pytest.approx(1.901, abs=2e-3)


def test_smaller_radius_gives_a_bigger_time_buffer():
    """A shorter arc (R * theta) outweighs the larger wheel-speed factor."""
    assert (CircleLeftStep(T_60BPM, radius=0.4).min_duration()
            < CircleLeftStep(T_60BPM, radius=0.5).min_duration())


def test_large_radius_needs_a_longer_window():
    """R = 0.8 m means a 0.42 m arc: it does NOT fit a 2.57 s measure (min ~3.14 s),
    so the constructor must reject it at 70 BPM but accept it at 4 s."""
    with pytest.raises(InfeasibleStepError):
        CircleLeftStep(T_70BPM, radius=0.8)
    CircleLeftStep(4.0, radius=0.8)


def test_waltz_tempos_feasible_and_too_short_rejected():
    CircleLeftStep(T_60BPM)
    CircleRightStep(T_70BPM)
    with pytest.raises(InfeasibleStepError):
        CircleLeftStep(1.5)


def test_exactly_min_duration_is_accepted():
    t_min = CircleLeftStep(T_60BPM).min_duration()
    step = CircleLeftStep(t_min)  # must NOT raise
    assert step.peak_outer_wheel_speed <= step.limits.v_max * (1 + 1e-9)
    assert step.peak_outer_wheel_acceleration <= step.limits.a_max * (1 + 1e-9)


def test_angular_velocity_cap_binds_for_tiny_radius():
    """For a very tight circle the firmware cap w_max (not the wheels) can be the
    binding limit; check the step respects it or is rejected."""
    tight = RobotLimits(w_max=0.3)
    with pytest.raises(InfeasibleStepError):
        CircleLeftStep(T_70BPM, limits=tight)  # needs w_peak ~0.41 rad/s > 0.3


# ---- Boundaries, timer, validation -------------------------------------------------------------
@pytest.mark.parametrize("t", [-5.0, -1e-9, T_60BPM, T_60BPM + 1e-9, 100.0])
def test_zero_command_outside_window(t):
    assert CircleLeftStep(T_60BPM).get_velocity(t) == (0.0, 0.0)


@pytest.mark.parametrize("period", [0.02, 0.05])   # 50 Hz and 20 Hz timers
@pytest.mark.parametrize("T", [3.0, T_70BPM, 2.61])
def test_timer_sampling_at_20_and_50_hz(T, period):
    """Discrete commands at both candidate node rates: stopped from T onwards,
    and the summed arc length and heading are close to the targets."""
    step = CircleLeftStep(T)
    ticks = np.arange(int(math.ceil(T / period)) + 5) * period
    c = np.array([step.get_velocity(t) for t in ticks])
    assert np.all(c[ticks >= T] == 0.0)
    assert np.sum(c[:, 0]) * period == pytest.approx(CIRCLE_RADIUS * ANGLE, abs=1e-3)
    assert np.sum(c[:, 1]) * period == pytest.approx(ANGLE, abs=2e-3)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_invalid_duration_is_rejected(bad):
    with pytest.raises(ValueError):
        CircleLeftStep(bad)


@pytest.mark.parametrize("bad", [0.0, -0.5, math.nan, math.inf])
def test_invalid_angle_and_radius_are_rejected(bad):
    for cls in (CircleLeftStep, CircleRightStep):
        with pytest.raises(ValueError):
            cls(T_60BPM, arc_angle=bad)
        with pytest.raises(ValueError):
            cls(T_60BPM, radius=bad)


# ---- Plot (for humans) --------------------------------------------------------------------------
def test_save_circle_path_plot():
    """Save the dead-reckoned x-y paths (left/right, R = 0.5 and 0.4) and the
    commanded v / w to tests/output/circle_paths.png for visual inspection."""
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    fig, (ax_path, ax_v, ax_w) = plt.subplots(1, 3, figsize=(15, 4.5))
    for R, color in [(0.5, "tab:blue"), (0.4, "tab:orange")]:
        for cls, style in [(CircleLeftStep, "-"), (CircleRightStep, "--")]:
            step = cls(T_70BPM, radius=R)
            x, y, _ = dead_reckon(step)
            ax_path.plot(x, y, style, color=color, label=f"{cls.__name__[6:-4]} R={R}")
            if cls is CircleLeftStep:
                t, v, w = commands(step)
                ax_v.plot(t, v, color=color, label=f"R={R}")
                ax_w.plot(t, w, color=color, label=f"R={R}")
    ax_path.set_aspect("equal"); ax_path.set_title("Dead-reckoned path (start at origin, facing +x)")
    ax_path.set_xlabel("x [m]"); ax_path.set_ylabel("y [m]")
    ax_v.set_title("linear.x [m/s]"); ax_w.set_title("angular.z [rad/s]")
    for ax in (ax_path, ax_v, ax_w):
        ax.grid(True, alpha=0.3); ax.legend()
    ax_v.set_xlabel("Time [s]"); ax_w.set_xlabel("Time [s]")
    fig.tight_layout()
    path = out_dir / "circle_paths.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    assert path.exists()
