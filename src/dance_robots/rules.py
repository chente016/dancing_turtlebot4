"""Choreography rules: the restrictions the choreography team works within.

These are POLICY, not physics. The step classes (ForwardStep, TurnStep, ...) are
general and can do more than this; the rules below decide what the choreography
CSV is ALLOWED to ask for. Changing a restriction later is a one-line edit here.

Feasibility (is the measure long enough?) is a separate question, answered by the
step classes from the robot limits and the timetable. See step_spec.py.
"""

from __future__ import annotations

from .circle_step import CIRCLE_RADIUS
from .linear_step import BACKWARD_DISTANCE, FORWARD_DISTANCE

# --- Fixed moves (not configurable from the CSV) ------------------------------
FORWARD_DISTANCE_M = FORWARD_DISTANCE    # 0.3 m
BACKWARD_DISTANCE_M = BACKWARD_DISTANCE  # 0.3 m

# --- Turns in place: free angle between these values (inclusive) ------------------
TURN_ANGLE_MIN_DEG = 1.0
TURN_ANGLE_MAX_DEG = 90.0
TURN_ANGLE_DEFAULT_DEG = 30.0   # used when angle_deg is blank

# --- Circles (arcs): angle in (0, MAX], radius in [MIN, MAX] ----------------------
CIRCLE_ANGLE_MAX_DEG = 45.0     # the minimum is "greater than 0"
CIRCLE_ANGLE_DEFAULT_DEG = 30.0
CIRCLE_RADIUS_DEFAULT_M = CIRCLE_RADIUS  # 0.4 m, used when radius_m is blank
CIRCLE_RADIUS_MAX_M = 0.4       # bigger radii are NOT allowed
CIRCLE_RADIUS_MIN_M = 0.1       # assumption: smaller arcs are basically a spin

# --- Warning thresholds (NOT errors) -------------------------------------------------
# A very slow move may not overcome friction / motor deadband on the real robot.
# These values are first guesses: tune them after the first robot tests.
MIN_RELIABLE_LINEAR_SPEED = 0.03    # [m/s]   peak speed of forward/backward/circle
MIN_RELIABLE_ANGULAR_SPEED = 0.05   # [rad/s] peak speed of turns

# --- Timetable consistency -------------------------------------------------------------
# Measures whose start/end differ by less than this are considered contiguous.
TIME_TOLERANCE_S = 1e-3