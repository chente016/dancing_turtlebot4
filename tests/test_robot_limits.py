"""Tests for RobotLimits: defaults and the derived angular limits."""

from __future__ import annotations

import dataclasses

import pytest

from dance_robots.dance_step import RobotLimits


def test_default_values():
    lim = RobotLimits()
    assert lim.v_max == 0.306
    assert lim.a_max == 0.6
    assert lim.j_max == 3.0
    assert lim.w_max == 1.9          # firmware default, NOT the 2.64 rad/s override
    assert lim.track_width == 0.235  # Create 3 documentation


def test_angular_limits_are_derived_from_wheel_geometry():
    """alpha_max = 2 * a_max / track,  jerk_ang_max = 2 * j_max / track."""
    lim = RobotLimits()
    assert lim.alpha_max == pytest.approx(2 * 0.6 / 0.235)   # ~5.106 rad/s^2
    assert lim.jerk_ang_max == pytest.approx(2 * 3.0 / 0.235)  # ~25.53 rad/s^3


def test_derived_limits_follow_overrides():
    """They are properties, so changing a_max / track keeps them consistent."""
    lim = RobotLimits(a_max=0.3, j_max=1.5, track_width=0.2)
    assert lim.alpha_max == pytest.approx(3.0)
    assert lim.jerk_ang_max == pytest.approx(15.0)


def test_wheel_speed_at_max_rotation_is_within_linear_limit():
    """Sanity check: spinning at w_max moves each wheel at w_max*b/2 ~ 0.223 m/s,
    which is below the 0.306 m/s linear limit."""
    lim = RobotLimits()
    assert lim.w_max * lim.track_width / 2.0 < lim.v_max


def test_limits_are_immutable():
    lim = RobotLimits()
    with pytest.raises(dataclasses.FrozenInstanceError):
        lim.v_max = 1.0  # type: ignore[misc]
