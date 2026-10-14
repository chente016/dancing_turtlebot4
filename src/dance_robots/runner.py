"""DanceRunner: the ROS-free state machine behind the director node.

Where it sits
-------------
    Choreography (one per robot)  ->  DanceRunner  ->  DanceNode (thin, rclpy)

The runner knows NOTHING about ROS. Time is passed in as plain seconds (floats)
read from the node's ROS clock, so the whole thing can be tested with a fake clock.

States
------
    IDLE     nothing loaded to move; waiting for arm()
    ARMED    pre-flight passed; waiting for schedule_start()
    WAITING  start time known and in the future; nothing is published yet
    RUNNING  publishing the choreography commands
    DONE     publishing a short burst of zero velocities, then back to IDLE

Time convention
---------------
`start_s` is the absolute (node clock) time at which song t = 0 happens.
Each dancer gets  t = now - start_s  and calls choreography.get_velocity(t).
The per-robot `music_start_offset` is already applied INSIDE Choreography, so the
runner must never add it a second time.

Note: absolute ROS time is ~1.7e9 s. A float64 resolves that to ~0.2 microseconds,
which is plenty for a 20 Hz timer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, NamedTuple, Protocol, Sequence


class State(Enum):
    IDLE = "idle"
    ARMED = "armed"
    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"


class Dancer(Protocol):
    """What the runner needs from a Choreography (kept as a Protocol so tests can fake it)."""

    node_end_s: float  # node time at which this robot's dance is over

    def get_velocity(self, t_node: float) -> tuple[float, float]: ...


class Result(NamedTuple):
    """Answer to arm / schedule_start / abort. `message` goes straight into the service reply."""

    ok: bool
    message: str


@dataclass(frozen=True)
class TickOutput:
    """What the node must do after one timer tick."""

    commands: dict[str, tuple[float, float]] | None  # {robot: (linear.x, angular.z)}; None = publish nothing
    warnings: tuple[str, ...] = ()                    # to be logged by the node


class DanceRunner:
    def __init__(self, dancers: Mapping[str, Dancer], zero_burst_ticks: int = 5,
                 start_tolerance_s: float = 0.5, gap_warn_s: float = 0.25) -> None:
        """
        dancers            {robot name: Choreography}, e.g. {"donatello": ..., "leonardo": ...}
        zero_burst_ticks   how many ticks of zero velocity to send at the end or on abort
                           (BEST_EFFORT may drop a packet, so one zero is not enough)
        start_tolerance_s  how late a start time may already be when schedule_start arrives
        gap_warn_s         warn if two RUNNING ticks are further apart than this
                           (the Create 3 stops by itself after 0.5 s without commands)
        """
        if not dancers:
            raise ValueError("DanceRunner needs at least one dancer")
        if zero_burst_ticks < 1:
            raise ValueError("zero_burst_ticks must be at least 1")
        self._dancers = dict(dancers)
        self._zero_burst_ticks = zero_burst_ticks
        self._start_tolerance_s = start_tolerance_s
        self._gap_warn_s = gap_warn_s

        self.state = State.IDLE
        self._start_s: float | None = None      # absolute start of the song
        self._last_tick_s: float | None = None  # for the gap warning
        self._burst_left = 0                    # zero commands still to send

    # ------------------------------------------------------------------ queries
    @property
    def names(self) -> Sequence[str]:
        return list(self._dancers)

    @property
    def start_s(self) -> float | None:
        return self._start_s

    @property
    def needs_timer(self) -> bool:
        """True while the node's periodic timer must keep running."""
        return self.state in (State.WAITING, State.RUNNING, State.DONE)

    @property
    def dance_length_s(self) -> float:
        """Seconds from the start until the LAST robot has finished."""
        return max(d.node_end_s for d in self._dancers.values())

    # ------------------------------------------------------------------ commands
    def arm(self, problems: Sequence[str] = ()) -> Result:
        """Go IDLE -> ARMED.

        `problems` are the pre-flight failures found by the node (e.g. 'no subscriber
        on /donatello/cmd_vel'). The runner cannot check those itself: they need ROS.
        """
        if self.state in (State.WAITING, State.RUNNING, State.DONE):
            return Result(False, f"cannot arm while {self.state.value}")
        if problems:
            self.state = State.IDLE
            return Result(False, "pre-flight failed: " + "; ".join(problems))
        self.state = State.ARMED
        return Result(True, "armed")

    def schedule_start(self, start_s: float, now_s: float) -> Result:
        """Go ARMED -> WAITING. Repeating the SAME start time is acknowledged again,
        so a client that lost our reply can safely retry."""
        if self.state in (State.WAITING, State.RUNNING) and self._start_s is not None:
            if abs(start_s - self._start_s) < 1e-6:
                return Result(True, "already scheduled for this start time")
            return Result(False, "a different start time is already scheduled; abort first")
        if self.state is not State.ARMED:
            return Result(False, f"not armed (state is {self.state.value})")
        if start_s < now_s - self._start_tolerance_s:
            return Result(False, f"start time is {now_s - start_s:.2f} s in the past")
        self._start_s = start_s
        self.state = State.WAITING
        return Result(True, f"start in {start_s - now_s:.2f} s")

    def abort(self) -> Result:
        """Stop everything, from ANY state. Safe to call twice (it just restarts the zero burst)."""
        self.state = State.DONE
        self._burst_left = self._zero_burst_ticks
        self._start_s = None
        self._last_tick_s = None
        return Result(True, "aborted")

    # ------------------------------------------------------------------ the timer tick
    def tick(self, now_s: float) -> TickOutput:
        """Call from every timer callback. Returns what to publish (if anything)."""
        # WAITING -> RUNNING happens by TIME, so an early or late tick can't break it.
        if self.state is State.WAITING and self._start_s is not None and now_s >= self._start_s:
            self.state = State.RUNNING
            self._last_tick_s = None
        if self.state is State.RUNNING:
            return self._tick_running(now_s)
        if self.state is State.DONE:
            return self._tick_done()
        return TickOutput(None)  # IDLE / ARMED / WAITING: stay silent (don't fight teleop)

    def _tick_running(self, now_s: float) -> TickOutput:
        warnings: list[str] = []
        if self._last_tick_s is not None and now_s - self._last_tick_s > self._gap_warn_s:
            warnings.append(f"timer gap of {now_s - self._last_tick_s:.2f} s "
                            f"(robots stop on their own after 0.5 s without commands)")
        self._last_tick_s = now_s

        t = now_s - self._start_s            # time since song t = 0 (offset applied inside)
        if t >= self.dance_length_s:         # everybody is done: start the zero burst
            self.state = State.DONE
            self._burst_left = self._zero_burst_ticks
            return TickOutput(self._tick_done().commands, tuple(warnings))
        commands = {name: d.get_velocity(t) for name, d in self._dancers.items()}
        return TickOutput(commands, tuple(warnings))

    def _tick_done(self) -> TickOutput:
        if self._burst_left <= 0:
            self.state = State.IDLE
            return TickOutput(None)
        self._burst_left -= 1
        if self._burst_left == 0:            # that was the last zero: back to IDLE
            self.state = State.IDLE
            self._start_s = None
        return TickOutput({name: (0.0, 0.0) for name in self._dancers})