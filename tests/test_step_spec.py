"""Tests for StepSpec: validation rules, defaults, building and feasibility queries."""

from __future__ import annotations

import math

import pytest

from dance_robots import rules
from dance_robots.circle_step import CircleLeftStep, CircleRightStep
from dance_robots.linear_step import BackwardStep, ForwardStep
from dance_robots.step_spec import STEP_KINDS, StepSpec
from dance_robots.turn_step import TurnLeftStep, TurnRightStep

T_70BPM = 3 * 60.0 / 70.0


# ---- Defaults ------------------------------------------------------------------
def test_defaults_are_applied_when_cells_are_blank():
    assert StepSpec("turn_left").angle == 30.0
    assert StepSpec("circle_left").angle == 30.0
    assert StepSpec("circle_left").radius == 0.4
    assert StepSpec("forward").angle is None and StepSpec("forward").radius is None


def test_default_circle_radius_is_the_maximum_allowed():
    assert rules.CIRCLE_RADIUS_DEFAULT_M == rules.CIRCLE_RADIUS_MAX_M == 0.4


# ---- Validation -------------------------------------------------------------------
@pytest.mark.parametrize("kind", STEP_KINDS)
def test_all_six_steps_are_valid_with_defaults(kind):
    assert StepSpec(kind).validate() == []


@pytest.mark.parametrize("angle", [1.0, 1.5, 45.0, 90.0])
def test_turn_angles_inside_range_are_valid(angle):
    assert StepSpec("turn_left", angle).validate() == []


@pytest.mark.parametrize("angle", [0.0, 0.99, 90.01, 120.0, -30.0, 180.0])
def test_turn_angles_outside_range_are_rejected(angle):
    errors = StepSpec("turn_right", angle).validate()
    assert len(errors) == 1 and "not allowed" in errors[0] and "1° to 90°" in errors[0]


@pytest.mark.parametrize("angle", [0.1, 15.0, 45.0])
def test_circle_angles_inside_range_are_valid(angle):
    assert StepSpec("circle_left", angle).validate() == []


@pytest.mark.parametrize("angle", [0.0, -5.0, 45.01, 90.0])
def test_circle_angles_outside_range_are_rejected(angle):
    errors = StepSpec("circle_right", angle).validate()
    assert len(errors) == 1 and "not allowed" in errors[0]


@pytest.mark.parametrize("radius", [0.1, 0.25, 0.4])
def test_circle_radii_inside_range_are_valid(radius):
    assert StepSpec("circle_left", 30.0, radius).validate() == []


@pytest.mark.parametrize("radius", [0.05, 0.401, 0.5, 1.0, 0.0, -0.4])
def test_circle_radii_outside_range_are_rejected(radius):
    errors = StepSpec("circle_left", 30.0, radius).validate()
    assert len(errors) == 1 and "radius" in errors[0]


def test_bigger_radius_than_0_4_is_rejected():
    """The team constraint: radii may get smaller, never bigger."""
    assert StepSpec("circle_left", 30.0, 0.5).validate() != []


def test_unknown_step_lists_the_valid_ones():
    (error,) = StepSpec("hop").validate()
    assert "unknown step 'hop'" in error and "turn_left" in error


@pytest.mark.parametrize("kind", ["forward", "backward"])
def test_linear_steps_reject_angle_and_radius(kind):
    assert "fixed distance" in StepSpec(kind, angle_deg=10.0).validate()[0]
    assert "radius_m" in StepSpec(kind, radius_m=0.3).validate()[0]


@pytest.mark.parametrize("kind", ["turn_left", "turn_right"])
def test_turns_reject_radius(kind):
    assert "radius_m" in StepSpec(kind, 30.0, 0.3).validate()[0]


# ---- Building ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind, cls", [
    ("forward", ForwardStep), ("backward", BackwardStep),
    ("turn_left", TurnLeftStep), ("turn_right", TurnRightStep),
    ("circle_left", CircleLeftStep), ("circle_right", CircleRightStep)])
def test_build_creates_the_right_class(kind, cls):
    assert isinstance(StepSpec(kind).build(3.0), cls)


def test_build_uses_the_given_angle_and_radius():
    turn = StepSpec("turn_left", 90.0).build(3.0)
    assert turn.angle == pytest.approx(math.pi / 2)
    circle = StepSpec("circle_right", 45.0, 0.3).build(3.0)
    assert circle.radius == 0.3
    assert circle.arc_angle == pytest.approx(-math.pi / 4)


# ---- Feasibility queries ---------------------------------------------------------------------
@pytest.mark.parametrize("spec, expected", [
    (StepSpec("forward"), 2.000),
    (StepSpec("backward"), 2.000),
    (StepSpec("turn_left", 30.0), 0.906),
    (StepSpec("turn_left", 90.0), 1.653),
    (StepSpec("circle_left", 30.0), 1.901),
    (StepSpec("circle_left", 45.0), 2.656),
    (StepSpec("circle_left", 45.0, 0.3), 2.14),
])
def test_min_duration_reference_values(spec, expected):
    assert spec.min_duration() == pytest.approx(expected, abs=5e-3)


def test_min_duration_does_not_depend_on_the_window():
    spec = StepSpec("circle_left", 45.0)
    assert spec.min_duration() == spec.build(10.0).min_duration()


def test_big_circle_does_not_fit_70_bpm_but_fits_60_bpm():
    spec = StepSpec("circle_left", 45.0)
    assert not spec.fits(T_70BPM)
    assert spec.fits(3.0)


def test_max_angle_for_window_is_the_edge_of_feasibility():
    spec = StepSpec("circle_left", 45.0)
    edge = spec.max_angle_deg_for(T_70BPM)
    assert 40.0 < edge < 45.0
    from dataclasses import replace
    assert replace(spec, angle_deg=edge).fits(T_70BPM)
    assert not replace(spec, angle_deg=edge + 0.01).fits(T_70BPM)


def test_max_angle_returns_the_maximum_when_everything_fits():
    assert StepSpec("circle_left", 45.0).max_angle_deg_for(10.0) == 45.0
    assert StepSpec("turn_left", 90.0).max_angle_deg_for(10.0) == 90.0


def test_max_angle_is_none_when_nothing_fits_or_not_applicable():
    assert StepSpec("turn_left", 90.0).max_angle_deg_for(0.1) is None
    assert StepSpec("forward").max_angle_deg_for(5.0) is None


def test_max_radius_for_window():
    spec = StepSpec("circle_left", 45.0, 0.4)
    r = spec.max_radius_m_for(T_70BPM)
    assert 0.3 < r < 0.4
    from dataclasses import replace
    assert replace(spec, radius_m=r).fits(T_70BPM)
    assert spec.max_radius_m_for(10.0) == 0.4
    assert StepSpec("turn_left").max_radius_m_for(5.0) is None


def test_peak_speed_units():
    value, unit = StepSpec("turn_left", 90.0).peak_speed(3.0)
    assert unit == "rad/s" and value == pytest.approx(2 * math.pi / 2 / 3.0)
    value, unit = StepSpec("forward").peak_speed(3.0)
    assert unit == "m/s" and value == pytest.approx(0.2)


def test_describe():
    assert StepSpec("forward").describe() == "forward"
    assert StepSpec("turn_left", 90.0).describe() == "turn_left 90°"
    assert StepSpec("circle_right", 45.0, 0.3).describe() == "circle_right 45° r=0.3 m"
