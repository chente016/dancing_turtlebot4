"""Offline simulation of a choreography: mimics the ROS timer, no ROS needed.

Every `period` seconds the "node" asks the choreography for a command and holds it
until the next tick (zero-order hold, like publishing on /cmd_vel). The held
command is integrated with the unicycle model to get the path the robot would
follow in an IDEAL world (no wheel slip, no latency, no firmware smoothing).
It validates our logic, not the real robot.

Pure Python + numpy. Compatible with Python 3.10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .choreography import Choreography


@dataclass(frozen=True)
class SimResult:
    """Everything recorded by `simulate`. Times are NODE times."""

    t: np.ndarray        # fine time grid
    x: np.ndarray        # position [m] on that grid
    y: np.ndarray
    theta: np.ndarray    # heading [rad], 0 = facing +x, counter-clockwise positive
    tick_t: np.ndarray   # time of every timer tick
    tick_v: np.ndarray   # linear.x published at every tick
    tick_w: np.ndarray   # angular.z published at every tick

    def pose_at(self, t: float) -> tuple[float, float, float]:
        """Interpolated (x, y, theta) at node time t."""
        return (float(np.interp(t, self.t, self.x)),
                float(np.interp(t, self.t, self.y)),
                float(np.interp(t, self.t, self.theta)))


def simulate(choreography: Choreography, period: float = 0.02,
             tail: float = 0.5, substeps: int = 10) -> SimResult:
    """Run the choreography against a timer of the given period (0.02 = 50 Hz,
    0.05 = 20 Hz), starting from the origin facing +x.

    `tail` extra seconds are simulated after the dance to confirm the robot
    stays stopped. `substeps` controls the integration accuracy per tick.
    """
    n_ticks = int(math.ceil((choreography.node_end_s + tail) / period))
    dt = period / substeps

    tick_t = np.arange(n_ticks) * period
    tick_v = np.zeros(n_ticks)
    tick_w = np.zeros(n_ticks)

    n = n_ticks * substeps
    t = np.arange(n + 1) * dt
    x = np.zeros(n + 1); y = np.zeros(n + 1); th = np.zeros(n + 1)

    i = 0
    for k in range(n_ticks):
        v, w = choreography.get_velocity(tick_t[k])  # one command per tick...
        tick_v[k], tick_w[k] = v, w
        for _ in range(substeps):                    # ...held until the next tick
            mid = th[i] + w * dt / 2.0               # midpoint heading: better accuracy
            x[i + 1] = x[i] + v * math.cos(mid) * dt
            y[i + 1] = y[i] + v * math.sin(mid) * dt
            th[i + 1] = th[i] + w * dt
            i += 1
    return SimResult(t, x, y, th, tick_t, tick_v, tick_w)