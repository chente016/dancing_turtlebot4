## ROS 2 stage: the director node

One node (`dance_director`) publishes the choreography to **all** robots. It runs on
the remote computer, not on the robots. The decisions live in
`src/dance_robots/runner.py` (tested with `uv run pytest -q`); the node in
`ros/dance_director/` is only wiring.

```
ros/
├── dance_interfaces/   StartDance.srv (ament_cmake)
└── dance_director/     the node, example parameters in config/director.yaml (ament_python)
```

### One-time setup (workspace `~/turtlebot4_ws`)

```bash
# Link the packages into the workspace (the source stays in this repo)
ln -s ~/dancing_turtlebot4/ros/dance_interfaces ~/turtlebot4_ws/src/dance_interfaces
ln -s ~/dancing_turtlebot4/ros/dance_director   ~/turtlebot4_ws/src/dance_director

# Let the system Python (used by ros2 run) import the library. Add this to ~/.bashrc
export PYTHONPATH=$HOME/dancing_turtlebot4/src:$PYTHONPATH

# Build WITHOUT the uv venv activated
cd ~/turtlebot4_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select dance_interfaces dance_director
source install/setup.bash
```

### Run

```bash
ros2 run dance_director director --ros-args --params-file \
  ~/dancing_turtlebot4/ros/dance_director/config/director.yaml
```

The node loads and checks the CSVs first. If there are errors it prints them (same text
as the checker) and does **not** start. Warnings are printed and it starts.

| Service | Type | What it does |
|---|---|---|
| `/dance/arm` | `std_srvs/Trigger` | Pre-flight (every `/<robot>/cmd_vel` has a subscriber), then ARMED |
| `/dance/schedule_start` | `dance_interfaces/StartDance` | Absolute ROS time at which song t = 0 happens |
| `/dance/abort` | `std_srvs/Trigger` | Zero velocity to all robots, from any state |

```bash
ros2 service call /dance/arm std_srvs/srv/Trigger
ros2 service call /dance/abort std_srvs/srv/Trigger
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `timetable_csv` | (required) | Timetable file |
| `dancers` | `[donatello]` | Robot namespaces |
| `<robot>.choreography_csv` | (required) | One per dancer |
| `<robot>.music_start_offset` | 0.0 | Seconds; negative sends commands earlier. Tuned per robot |
| `publish_rate_hz` | 20.0 | Timer rate |
| `cmd_vel_topic` | `cmd_vel` | Published on `/<robot>/cmd_vel` |
| `start_tolerance_s` | 0.5 | How late a start time may already be |
| `zero_burst_ticks` | 5 | Zero commands sent at the end or on abort |
| `gap_warn_s` | 0.25 | Warn if timer ticks are further apart (robots stop at 0.5 s) |

### Safety notes

* Commands are published **BEST_EFFORT**, depth 1, to match the robot's `create3_repub`
  subscriber. The Create 3 stops by itself after 0.5 s without commands.
* Stop any joystick node (`teleop_twist_joy`) on the robots during a dance, or it can
  fight the director on the same topic. `/dance/arm` logs a warning if it finds one.
* Ctrl+C sends zeros only on a best-effort basis (a known rclpy race on Humble). Use
  `/dance/abort` for a deliberate stop.

### Start a dance with the stand-in client (no audio yet)

```bash
# Terminal 1: the director (use_sim_time:=true only for Gazebo)
ros2 launch dance_director director.launch.py
# Terminal 2: arm, then start the dance in 5 s
ros2 run dance_director start_dance
```

`start_dance` calls `/dance/arm`, then `/dance/schedule_start` with `now + lead`. If the
director does not confirm, it calls `/dance/abort`. Options: `--lead SECONDS`, `--no-arm`.
The real orchestrator replaces this script; the services stay the same.

### Test in the TurtleBot 4 simulator (Gazebo / Ignition Fortress)

1. Do **not** use the lab's discovery server in these terminals (`unset ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT`, then `ros2 daemon stop`).
2. Launch the simulator **without** slam or nav2, with the same namespace as in `director.yaml`:
   `ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py namespace:=donatello`
3. Undock the robot and wait until the undock action finishes.
4. Run the director and the client with `use_sim_time:=true` on both (see the launch and run lines above; for the client add `--ros-args -p use_sim_time:=true`).
5. Compare the odometry before and after. The example dance should end about
   (0.32 m, 0.64 m, 120°) from where it started, in the robot's starting frame.