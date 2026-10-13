"""Demo: load a choreography, simulate the 50 Hz / 20 Hz timer and plot the dance.

Run from the project root:
    uv run python examples/demo_choreography.py
    uv run python examples/demo_choreography.py --rate 50
    uv run python examples/demo_choreography.py my_choreo.csv my_timetable.csv

No ROS needed. The path is dead-reckoned from the COMMANDED velocities in an ideal
world (no slip, no latency): it validates our logic, not the real robot.
The figure is saved to output/demo_choreography.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save a PNG instead of opening a window
import matplotlib.pyplot as plt

from dance_robots.check import format_issues
from dance_robots.choreography import load_choreography
from dance_robots.issues import has_errors
from dance_robots.simulation import simulate
from dance_robots.timetable import load_timetable

HERE = Path(__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("choreography", nargs="?", default=str(HERE / "choreography.csv"))
    parser.add_argument("timetable", nargs="?", default=str(HERE / "timetable.csv"))
    parser.add_argument("--rate", type=float, default=20.0, help="timer rate in Hz (default 20)")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="music_start_offset in seconds (negative = commands go out earlier)")
    args = parser.parse_args()

    # 1) Load and validate with the SAME code the robot node will use.
    timetable, issues = load_timetable(args.timetable)
    choreography = None
    if timetable is not None:
        choreography, more = load_choreography(
            args.choreography, timetable, music_start_offset=args.offset)
        issues += more
    if issues:
        print("\n".join(format_issues(issues)))
    if has_errors(issues) or choreography is None:
        print("Cannot simulate: fix the errors first.")
        return 1

    # 2) Simulate the timer and integrate the pose.
    sim = simulate(choreography, period=1.0 / args.rate)
    x_end, y_end, th_end = sim.pose_at(sim.t[-1])
    print(f"Simulated at {args.rate:g} Hz. Final pose: x={x_end:.3f} m, y={y_end:.3f} m, "
          f"heading={th_end * 180 / 3.141592653589793:.1f} deg")

    # 3) Plot: path on the left, commanded velocities on the right.
    fig = plt.figure(figsize=(15, 7))
    ax_path = fig.add_subplot(1, 2, 1)
    ax_v = fig.add_subplot(2, 2, 2)
    ax_w = fig.add_subplot(2, 2, 4, sharex=ax_v)

    ax_path.plot(sim.x, sim.y, color="tab:blue", linewidth=1.5)
    ax_path.plot(0, 0, "go", label="start")
    for m in timetable.measures:                     # a dot + number at every measure start
        t = m.start_s + choreography.music_start_offset
        if t <= sim.t[-1]:
            px, py, _ = sim.pose_at(t)
            ax_path.plot(px, py, ".", color="black", markersize=5)
            ax_path.annotate(str(m.number), (px, py), textcoords="offset points",
                             xytext=(4, 4), fontsize=8)
    ax_path.plot(x_end, y_end, "rs", label="end")
    ax_path.set_aspect("equal")
    ax_path.set_xlabel("x [m]"); ax_path.set_ylabel("y [m]")
    ax_path.set_title("Dead-reckoned path (numbers = start of each measure)")
    ax_path.grid(True, alpha=0.3); ax_path.legend()

    ax_v.step(sim.tick_t, sim.tick_v, where="post", color="tab:blue")
    ax_w.step(sim.tick_t, sim.tick_w, where="post", color="tab:orange")
    ax_v.set_ylabel("linear.x [m/s]"); ax_w.set_ylabel("angular.z [rad/s]")
    ax_w.set_xlabel("Node time [s]")
    for ax in (ax_v, ax_w):
        for m in timetable.measures:                 # measure boundaries
            ax.axvline(m.start_s + choreography.music_start_offset, color="gray", linewidth=0.5)
        ax.grid(True, alpha=0.3)
    for s in choreography.schedule:                  # label each move
        t0 = s.start_s + choreography.music_start_offset
        ax_v.axvspan(t0, t0 + s.duration_s, alpha=0.12, color="tab:green")
        ax_v.text(t0 + 0.05, ax_v.get_ylim()[1] * 0.92, s.spec.kind, fontsize=7, rotation=90, va="top")
    ax_v.set_title(f"Commands published by the node ({args.rate:g} Hz timer)")

    fig.tight_layout()
    out = Path("output"); out.mkdir(exist_ok=True)
    path = out / "demo_choreography.png"
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())