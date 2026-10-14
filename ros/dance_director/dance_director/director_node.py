"""Director node: ONE node that publishes the choreography to ALL robots.

This file is only wiring. All decisions (states, timing, end of dance, zero bursts)
live in dance_robots.runner.DanceRunner, which is tested without ROS.

    /dance/arm            (std_srvs/Trigger)             IDLE -> ARMED after pre-flight
    /dance/schedule_start (dance_interfaces/StartDance)  ARMED -> WAITING at an absolute time
    /dance/abort          (std_srvs/Trigger)             stop everything, from any state
    /<robot>/cmd_vel      (geometry_msgs/Twist)          published BEST_EFFORT, depth 1

Everything runs in ONE thread (the default executor), so service and timer callbacks
never overlap and the runner needs no locks. Keep every callback short.
"""

from __future__ import annotations

import sys
import time
import traceback

import rclpy
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
from dance_interfaces.srv import StartDance

try:
    # The library comes from the uv repo; it must be on PYTHONPATH (see the README).
    from dance_robots.check import format_issues
    from dance_robots.choreography import load_choreography
    from dance_robots.issues import has_errors
    from dance_robots.runner import DanceRunner, State
    from dance_robots.timetable import load_timetable
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"Cannot import the dance_robots library ({exc}).\n"
        "Add the repo's src folder to PYTHONPATH, e.g.\n"
        "  export PYTHONPATH=$HOME/dancing_turtlebot4/src:$PYTHONPATH") from exc


class ConfigError(Exception):
    """Bad parameters or CSV files: the node must not start."""


def _as_text(issues) -> str:
    """format_issues may return a string or a list of lines; give back one string."""
    text = format_issues(issues)
    return text if isinstance(text, str) else "\n".join(text)


class DirectorNode(Node):
    def __init__(self) -> None:
        super().__init__("dance_director")

        # ---- parameters (the per-dancer ones are declared once we know the names) ----
        self.declare_parameter("timetable_csv", "")
        self.declare_parameter("dancers", ["donatello"])  # non-empty default: ROS needs a type
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("start_tolerance_s", 0.5)
        self.declare_parameter("zero_burst_ticks", 5)
        self.declare_parameter("gap_warn_s", 0.25)
        names = [str(n) for n in self.get_parameter("dancers").value]
        for name in names:
            self.declare_parameter(f"{name}.choreography_csv", "")
            self.declare_parameter(f"{name}.music_start_offset", 0.0)

        rate = float(self.get_parameter("publish_rate_hz").value)
        if rate <= 0.0:
            raise ConfigError("publish_rate_hz must be positive")

        # ---- load and validate everything BEFORE creating anything that can move a robot ----
        choreographies = self._load_choreographies(names)
        self._runner = DanceRunner(
            choreographies,
            zero_burst_ticks=int(self.get_parameter("zero_burst_ticks").value),
            start_tolerance_s=float(self.get_parameter("start_tolerance_s").value),
            gap_warn_s=float(self.get_parameter("gap_warn_s").value))

        # ---- publishers: BEST_EFFORT / VOLATILE / depth 1 to match create3_repub ----
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        topic = str(self.get_parameter("cmd_vel_topic").value)
        self._pubs = {name: self.create_publisher(Twist, f"/{name}/{topic}", qos)
                            for name in names}

        # ---- timers ----
        # The periodic timer is created once and only started/stopped with reset()/cancel().
        self._tick_timer = self.create_timer(1.0 / rate, self._on_tick)
        self._tick_timer.cancel()
        # A one-shot timer that fires at the start time. Created per schedule_start.
        self._start_timer = None

        # ---- services (absolute names: one set for the whole fleet) ----
        self.create_service(Trigger, "/dance/arm", self._on_arm)
        self.create_service(StartDance, "/dance/schedule_start", self._on_schedule_start)
        self.create_service(Trigger, "/dance/abort", self._on_abort)

        self.get_logger().info(
            f"ready: dancers={names}, {rate:g} Hz, dance length "
            f"{self._runner.dance_length_s:.1f} s. Call /dance/arm next.")

    # ------------------------------------------------------------------ startup helpers
    def _load_choreographies(self, names):
        """Load the timetable and one choreography per robot. Raises ConfigError on any error."""
        timetable_path = str(self.get_parameter("timetable_csv").value)
        if not timetable_path:
            raise ConfigError("parameter 'timetable_csv' is empty")
        timetable, issues = load_timetable(timetable_path)
        all_issues = list(issues)

        choreographies = {}
        if timetable is not None:
            for name in names:
                path = str(self.get_parameter(f"{name}.choreography_csv").value)
                offset = float(self.get_parameter(f"{name}.music_start_offset").value)
                if not path:
                    raise ConfigError(f"parameter '{name}.choreography_csv' is empty")
                choreography, more = load_choreography(path, timetable, music_start_offset=offset)
                all_issues += list(more)
                if choreography is not None:
                    choreographies[name] = choreography

        if timetable is None or has_errors(all_issues) or len(choreographies) != len(names):
            raise ConfigError("the CSV files have errors:\n" + _as_text(all_issues))
        if all_issues:  # only warnings: show them, but start
            self.get_logger().warn("CSV warnings:\n" + _as_text(all_issues))
        return choreographies

    # ------------------------------------------------------------------ clock
    def _now_s(self) -> float:
        """Current time of the node clock in seconds (follows use_sim_time automatically)."""
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ services
    def _on_arm(self, request, response):
        problems = []
        for pub in self._pubs.values():
            if pub.get_subscription_count() == 0:
                problems.append(f"no subscriber on {pub.topic_name}")
            elif self.count_publishers(pub.topic_name) > 1:  # ours + someone else (teleop?)
                self.get_logger().warn(f"another node also publishes on {pub.topic_name}")
        # NOT checked yet: a Create 3 action goal in progress (it would ignore cmd_vel).
        result = self._runner.arm(problems)
        response.success, response.message = result.ok, result.message
        self.get_logger().info(f"arm: {result.message}")
        return response

    def _on_schedule_start(self, request, response):
        start_s = request.start_time.sec + request.start_time.nanosec * 1e-9
        now_s = self._now_s()
        result = self._runner.schedule_start(start_s, now_s)
        # A retry of the same start time is acknowledged by the runner; the timer already exists.
        if result.ok and self._start_timer is None:
            self._start_timer = self.create_timer(max(start_s - now_s, 0.001), self._on_start_time)
        response.success, response.message = result.ok, result.message
        self.get_logger().info(f"schedule_start: {result.message}")
        return response

    def _on_abort(self, request, response):
        result = self._runner.abort()
        self._release_start_timer()
        if self._tick_timer.is_canceled():
            self._tick_timer.reset()      # the zero burst needs the periodic timer
        self._handle_tick(self._now_s())  # send the first zeros NOW, not in 50 ms
        response.success, response.message = result.ok, result.message
        self.get_logger().warn("ABORT: zero velocity sent to all robots")
        return response

    # ------------------------------------------------------------------ timers
    def _on_start_time(self) -> None:
        """Fires once, at the start time. Starts the 20 Hz grid exactly there."""
        if self._start_timer is not None:
            self._start_timer.cancel()  # one-shot: never fire again
        if self._runner.state is not State.WAITING:
            return  # aborted in the meantime
        self._tick_timer.reset()  # periodic ticks from now on (first one in one period)
        # Tick at t = 0 right now. max(): a timer may fire a few microseconds early.
        self._handle_tick(max(self._now_s(), self._runner.start_s))
        self.get_logger().info("dance started")

    def _on_tick(self) -> None:
        self._handle_tick(self._now_s())

    def _handle_tick(self, now_s: float) -> None:
        """Ask the runner what to publish, publish it, stop the timer when idle."""
        try:
            out = self._runner.tick(now_s)
        except Exception:  # a bug must stop the robots, never leave them moving
            self.get_logger().error("exception in tick, aborting:\n" + traceback.format_exc())
            self._runner.abort()
            out = self._runner.tick(now_s)  # DONE state: zeros only, never calls the steps
        for warning in out.warnings:
            self.get_logger().warn(warning)
        if out.commands:
            for name, (linear_x, angular_z) in out.commands.items():
                msg = Twist()
                msg.linear.x = float(linear_x)
                msg.angular.z = float(angular_z)
                self._pubs[name].publish(msg)
        if not self._runner.needs_timer:  # finished (or aborted) and the zero burst is over
            self._tick_timer.cancel()
            self._release_start_timer()
            self.get_logger().info("idle (arm again for another take)")

    def _release_start_timer(self) -> None:
        if self._start_timer is not None:
            self._start_timer.cancel()
            self.destroy_timer(self._start_timer)
            self._start_timer = None

    # ------------------------------------------------------------------ shutdown
    def emergency_zeros(self) -> None:
        """Best effort on Ctrl+C. On Humble publishing during shutdown can fail (a known
        rclpy race), so the real safety nets are /dance/abort and the Create 3's own
        0.5 s command timeout."""
        if self._runner.state is State.IDLE:
            return
        try:
            for _ in range(3):
                for pub in self._pubs.values():
                    pub.publish(Twist())
                time.sleep(0.02)
        except Exception:
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    exit_code = 0
    try:
        node = DirectorNode()
        rclpy.spin(node)
    except ConfigError as exc:
        get_logger("dance_director").error(f"not starting: {exc}")
        exit_code = 1
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.emergency_zeros()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()