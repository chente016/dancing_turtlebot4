"""Stand-in for the sync team's orchestrator: arm the director, then start the dance.

    ros2 run dance_director start_dance                  # start in 5 s
    ros2 run dance_director start_dance --lead 8.0       # start in 8 s
    ros2 run dance_director start_dance --no-arm         # director was armed by hand
    ros2 run dance_director start_dance --ros-args -p use_sim_time:=true   # in Gazebo

It asks the director to start at an ABSOLUTE time (now + lead) on this node's clock, so
in the simulator it must use the same clock as the director (use_sim_time on both).
If the director does not confirm, it calls /dance/abort (the design rule: no ack, no dance).
Exit code 0 = scheduled, 1 = something failed.
"""

import argparse
import sys
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from std_srvs.srv import Trigger
from dance_interfaces.srv import StartDance


def _call(node, client, request, timeout_s):
    """Call a service and wait for the answer. Returns the response or None on timeout."""
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    return future.result() if future.done() else None


def _wait_for_clock(node, timeout_s=5.0) -> bool:
    """In sim time the clock reads 0 until the first /clock message arrives."""
    deadline = time.monotonic() + timeout_s
    while node.get_clock().now().nanoseconds == 0:
        if time.monotonic() > deadline:
            return False
        rclpy.spin_once(node, timeout_sec=0.1)
    return True


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Arm the director and schedule the dance start.")
    parser.add_argument("--lead", type=float, default=5.0,
                        help="seconds from now until song t = 0 (default 5)")
    parser.add_argument("--no-arm", action="store_true",
                        help="do not call /dance/arm first")
    options = parser.parse_args(remove_ros_args(args=sys.argv)[1:])

    rclpy.init(args=args)
    node = Node("start_dance_client")
    arm = node.create_client(Trigger, "/dance/arm")
    schedule = node.create_client(StartDance, "/dance/schedule_start")
    abort = node.create_client(Trigger, "/dance/abort")
    code = 1
    try:
        if not all(c.wait_for_service(timeout_sec=5.0) for c in (arm, schedule, abort)):
            print("The director services are not available. Is the director running "
                  "(and in the same ROS network)?")
            return
        if not _wait_for_clock(node):
            print("The clock is stuck at 0. With use_sim_time, is /clock being published?")
            return

        if not options.no_arm:
            reply = _call(node, arm, Trigger.Request(), timeout_s=5.0)
            if reply is None or not reply.success:
                print("ARM FAILED: " + ("no answer" if reply is None else reply.message))
                return
            print("armed")

        request = StartDance.Request()
        start = node.get_clock().now() + Duration(seconds=options.lead)
        request.start_time = start.to_msg()
        reply = _call(node, schedule, request, timeout_s=2.0)
        if reply is None or not reply.success:
            print("SCHEDULE FAILED: " + ("no answer" if reply is None else reply.message)
                  + " -> aborting")
            _call(node, abort, Trigger.Request(), timeout_s=2.0)
            return
        print(f"scheduled: {reply.message}")
        code = 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()