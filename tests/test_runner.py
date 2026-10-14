"""Tests for DanceRunner. No ROS and no real choreography: dancers and the clock are fakes."""

import pytest

from dance_robots.runner import DanceRunner, State

START = 1_700_000_000.0  # a realistic absolute ROS time in seconds


class FakeDancer:
    """Moves while 0 <= t < length. Returns (t, -t) so tests can see which t it was given."""

    def __init__(self, length: float) -> None:
        self.node_end_s = length

    def get_velocity(self, t: float):
        return (t, -t) if 0.0 <= t < self.node_end_s else (0.0, 0.0)


def make(burst: int = 3, **kw) -> DanceRunner:
    return DanceRunner({"a": FakeDancer(10.0), "b": FakeDancer(4.0)}, zero_burst_ticks=burst, **kw)


def armed_and_scheduled(runner: DanceRunner, lead: float = 5.0) -> None:
    assert runner.arm().ok
    assert runner.schedule_start(START + lead, now_s=START).ok


def test_starts_idle_and_silent():
    r = make()
    assert r.state is State.IDLE and not r.needs_timer
    assert r.tick(START).commands is None


def test_needs_at_least_one_dancer():
    with pytest.raises(ValueError):
        DanceRunner({})


def test_arm_with_problems_stays_idle_and_names_them():
    r = make()
    res = r.arm(["no subscriber on /donatello/cmd_vel"])
    assert not res.ok and "donatello" in res.message and r.state is State.IDLE


def test_cannot_schedule_before_arming():
    r = make()
    assert not r.schedule_start(START + 5, now_s=START).ok


def test_past_start_rejected_but_slightly_late_accepted():
    r = make()
    r.arm()
    assert not r.schedule_start(START - 2.0, now_s=START).ok       # 2 s late: reject
    assert r.state is State.ARMED
    assert r.schedule_start(START - 0.1, now_s=START).ok            # within tolerance
    assert r.state is State.WAITING


def test_silent_while_waiting_then_runs_with_correct_local_time():
    r = make()
    armed_and_scheduled(r)
    assert r.needs_timer
    assert r.tick(START + 4.9).commands is None                     # still waiting
    out = r.tick(START + 5.0 + 2.0)                                 # 2 s into the song
    assert r.state is State.RUNNING
    assert out.commands == {"a": (2.0, -2.0), "b": (2.0, -2.0)}


def test_shorter_dancer_rests_while_longer_continues():
    r = make()
    armed_and_scheduled(r)
    out = r.tick(START + 5.0 + 6.0)                                 # b ended at 4 s, a ends at 10 s
    assert out.commands == {"a": (6.0, -6.0), "b": (0.0, 0.0)}
    assert r.state is State.RUNNING


def test_end_sends_zero_burst_then_returns_to_idle():
    r = make(burst=3)
    armed_and_scheduled(r)
    r.tick(START + 5.0 + 1.0)
    for i in range(3):                                              # exactly 3 zero ticks
        out = r.tick(START + 5.0 + 10.0 + 0.05 * i)
        assert out.commands == {"a": (0.0, 0.0), "b": (0.0, 0.0)}
    assert r.state is State.IDLE and not r.needs_timer
    assert r.tick(START + 20.0).commands is None


def test_second_take_needs_a_new_arm():
    r = make(burst=1)
    armed_and_scheduled(r)
    r.tick(START + 5.0 + 11.0)                                      # ends, burst of 1, IDLE
    assert r.state is State.IDLE
    assert not r.schedule_start(START + 30, now_s=START + 20).ok    # not armed any more
    assert r.arm().ok and r.schedule_start(START + 30, now_s=START + 20).ok


def test_same_start_time_is_acknowledged_again_different_is_rejected():
    r = make()
    armed_and_scheduled(r)
    assert r.schedule_start(START + 5.0, now_s=START + 0.1).ok      # retry: ack again
    assert not r.schedule_start(START + 6.0, now_s=START + 0.1).ok  # reschedule: refused
    assert r.start_s == START + 5.0


def test_cannot_arm_while_busy():
    r = make()
    armed_and_scheduled(r)
    assert not r.arm().ok and r.state is State.WAITING


@pytest.mark.parametrize("when", ["idle", "armed", "waiting", "running"])
def test_abort_from_any_state_sends_zeros_then_idle(when):
    r = make(burst=2)
    if when != "idle":
        r.arm()
    if when in ("waiting", "running"):
        r.schedule_start(START + 5, now_s=START)
    if when == "running":
        r.tick(START + 7.0)
    assert r.abort().ok
    assert r.tick(START + 8.0).commands == {"a": (0.0, 0.0), "b": (0.0, 0.0)}
    assert r.tick(START + 8.05).commands == {"a": (0.0, 0.0), "b": (0.0, 0.0)}
    assert r.state is State.IDLE and r.tick(START + 8.1).commands is None


def test_abort_twice_is_safe():
    r = make(burst=2)
    r.abort()
    r.tick(START)
    assert r.abort().ok                                             # restarts the burst
    assert r.tick(START + 0.05).commands is not None


def test_gap_warning_only_for_long_gaps():
    r = make(gap_warn_s=0.25)
    armed_and_scheduled(r)
    assert not r.tick(START + 5.00).warnings
    assert not r.tick(START + 5.05).warnings                        # normal 20 Hz
    out = r.tick(START + 5.50)                                      # 0.45 s gap
    assert len(out.warnings) == 1 and "0.45" in out.warnings[0]