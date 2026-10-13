# dance_robots: rhythm-synced trajectory generator for TurtleBot 4

Open-loop dance moves for a fleet of TurtleBot 4 robots. Each move starts at
rest, travels a fixed displacement and stops again **inside a time window taken
from the music** (one or more whole measures of the score).

This repository contains the **pure-Python, ROS-free** part: the step classes,
the choreography sequencer, the CSV loaders, a plain-language checker and an
offline simulator. The ROS 2 node comes later (see [Roadmap](#roadmap)).

> **Vocabulary.** A *measure* is a *compás* (a bar of music). In 3/4 time one
> measure holds 3 beats. The timetable gives the start time and the duration of
> every measure in seconds, so time signature and tempo are already baked in.

## Project layout

```
dance_robots/
├── pyproject.toml            # created by uv (see setup)
├── .python-version           # created by uv, pins Python 3.10
├── README.md
├── examples/
│   ├── timetable.csv         # example timetable (3/4, 60 BPM then 70 BPM)
│   ├── choreography.csv      # example choreography (valid)
│   ├── bad_choreography.csv  # deliberately wrong: shows the error messages
│   └── demo_choreography.py  # simulate + plot a whole dance
├── src/dance_robots/
│   ├── dance_step.py         # DanceStep base class, RobotLimits, InfeasibleStepError
│   ├── scurve.py             # shared normalised S-curve + the two shape constants
│   ├── linear_step.py        # ForwardStep, BackwardStep
│   ├── turn_step.py          # TurnLeftStep, TurnRightStep (in place)
│   ├── circle_step.py        # CircleLeftStep, CircleRightStep (arc)
│   ├── rules.py              # what the choreography CSV is ALLOWED to ask for
│   ├── step_spec.py          # one move as plain values: validate, build, feasibility
│   ├── timetable.py          # measures on the song clock + timetable CSV loader
│   ├── choreography.py       # Choreography sequencer + choreography CSV loader
│   ├── check.py              # plain-language checker (python -m dance_robots.check)
│   ├── simulation.py         # offline 20/50 Hz timer simulation + dead reckoning
│   ├── csvio.py              # forgiving CSV reader
│   └── issues.py             # errors / warnings collected by the loaders
└── tests/                    # pytest suite (plots are written to tests/output/)
```

## Requirements

* [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python
  package and project manager). Check with `uv --version`.
* VS Code with the **Python** extension (Microsoft).
* You do **not** need to install Python yourself: uv downloads Python 3.10.

We use **Python 3.10** because that is the version ROS 2 Humble uses on
Ubuntu 22.04, so the code we write here will run unchanged on the robots.

## Setup (first time only)

Run these from the project folder (the one containing `src/` and `tests/`).

```bash
# 1. Create the project files. --lib gives the src/ layout, --python 3.10 pins
#    the interpreter (writes .python-version) and --no-readme keeps this README.
uv init --lib --python 3.10 --no-readme

# 2. Add the dependencies (this also creates the .venv and the uv.lock file).
uv add numpy matplotlib
uv add --dev pytest

# 3. Install everything exactly as locked, including this package in editable mode.
uv sync
```

Notes:

* `uv init` creates `src/dance_robots/__init__.py` containing a small `hello()`
  example. You can delete that function and leave the file empty.
* It must **not** overwrite our `.py` files or the tests. After running it, check
  with `git status` or by opening the files that nothing was replaced.
* **Everyone else on the team** only needs `uv sync` after cloning, because
  `pyproject.toml`, `.python-version` and `uv.lock` are committed.
* Add `output/` and `tests/output/` to `.gitignore` (generated plots).

## Running the tests

```bash
uv run pytest -v          # whole suite
uv run pytest -q          # compact output
uv run pytest tests/test_choreography.py    # a single file
uv run pytest -k "50hz"   # only tests whose name contains "50hz"
```

`uv run` automatically uses the project's `.venv`, so you never have to
activate it by hand. All tests should pass.

## VS Code setup

1. Open the project folder in VS Code (**File > Open Folder**).
2. Select the interpreter: `Ctrl+Shift+P` > **Python: Select Interpreter** >
   choose the one inside `.venv` (`./.venv/bin/python`).
3. Enable pytest in the Test Explorer: `Ctrl+Shift+P` >
   **Python: Configure Tests** > **pytest** > folder `tests`.
   Alternatively, create `.vscode/settings.json`:

   ```json
   {
     "python.testing.pytestEnabled": true,
     "python.testing.unittestEnabled": false,
     "python.testing.pytestArgs": ["tests"]
   }
   ```
4. Open the **Testing** panel (flask icon) to run or debug individual tests.

## The two CSV files

The choreography is described by **two** files, from two teams.

### 1. Timetable (from the music-analysis team)

One row per measure: its number, its start time and its duration in seconds.

```csv
measure,start_s,duration_s
1,0.000000,3.000000
2,3.000000,3.000000
7,18.000000,2.571429
```

* Measure numbers must be consecutive. Durations may change from measure to
  measure (tempo changes). A gap between measures gives a warning (a pause in
  the score?), an overlap is an error.
* Start times are taken from the table (never summed from durations), so rounding
  can't accumulate.

### 2. Choreography (written by an engineer with the choreographers)

One row per move, using **whole measures only**:

```csv
first_measure,last_measure,step,angle_deg,radius_m
40,,forward,,
41,43,turn_left,90,
45,,circle_right,45,0.3
```

* `last_measure` blank means a single measure. Measures that are not listed are
  **rests** (the robot stays still).
* A move over several measures is stretched over the whole window, from the start
  of the first measure to the end of the last one. Longer windows mean slower,
  gentler moves.
* `angle_deg` and `radius_m` are optional. Blank cells use the defaults.
* Lines starting with `#` are comments. Both `,` and `;` separators work (Spanish
  Excel exports `;` and decimal commas, which are understood).

| step | angle_deg | radius_m |
|------|-----------|----------|
| `forward`, `backward` | fixed 0.3 m, **leave blank** | leave blank |
| `turn_left`, `turn_right` | 1 to 90 (default 30) | leave blank |
| `circle_left`, `circle_right` | more than 0 and up to 45 (default 30) | 0.1 to **0.4** (default 0.4; **bigger is not allowed**) |

Turns are in place. Circles drive **forward** along an arc. Left = counter-
clockwise (ROS REP-103). These ranges are policy and live in `src/dance_robots/rules.py`.

## The plain-language checker

```bash
uv run python -m dance_robots.check examples/choreography.csv examples/timetable.csv
uv run python -m dance_robots.check examples/bad_choreography.csv examples/timetable.csv
```

It uses **exactly the same loading and validation code as the robot node**, so a
file that passes here can't be rejected later. It prints:

1. **Problems**: errors (must fix) and warnings (worth a look), each with the file
   line and, when possible, concrete fixes, for example:

   ```
   line 6: circle_left 45° r=0.4 m needs at least 2.66 s but measure 7 lasts only
   2.57 s. Options: use measures 7-8 (5.14 s); or reduce the angle to at most
   43.5°; or use a radius of at most 0.38 m.
   ```
2. **Moves**: one line per move with its time window and the margin it has.
3. **Song map**: measure by measure, which move sits where (use `--no-map` to hide).

Exit code: `0` = usable, `1` = errors. Warnings do not fail the check.

## The demo

```bash
uv run python examples/demo_choreography.py              # 20 Hz timer (default)
uv run python examples/demo_choreography.py --rate 50
uv run python examples/demo_choreography.py my_dance.csv my_timetable.csv
```

It validates the files, simulates the timer, integrates the commanded velocities
(unicycle model) and saves `output/demo_choreography.png`: the path with every
measure start numbered, and the commands with the move windows shaded.

> The path is dead-reckoned from the **commanded** velocities in an ideal world
> (no slip, no latency, no firmware smoothing). It validates our logic, not the
> real robot.

## How it works

### Who does what

* **Step** (`ForwardStep`, `TurnLeftStep`, ...): a pure function of *local* time.
  `get_velocity(t)` returns `(linear.x, angular.z)` for `t` seconds after *that
  step* started, and `(0, 0)` outside `[0, T)`. All the S-curve math is here.
* **Choreography** (sequencer): owns the schedule. Given the *global* time it finds
  the active step, converts `t_local = t_song - step_start`, and asks that step. It
  returns `(0, 0)` during rests and before/after the dance. It does no profile math.
* **ROS node** (later): every timer tick it calls
  `choreography.get_velocity(now - start)` and publishes a `Twist`.

```python
from dance_robots.timetable import load_timetable
from dance_robots.choreography import load_choreography

timetable, issues = load_timetable("examples/timetable.csv")
choreography, issues = load_choreography("examples/choreography.csv", timetable,
                                         music_start_offset=0.0)
v, omega = choreography.get_velocity(12.4)   # what to publish at t = 12.4 s
```

`music_start_offset` is the node time at which the song's t = 0 happens
(`t_song = t_node - offset`). A negative value sends commands earlier, to
compensate a robot that reacts late. It is one number, tuned on the real robot.

### Time-stretched S-curve

There is one fixed, normalised 7-segment jerk-limited profile (unit time, unit
distance). A move of size `D` in time `T` is that profile scaled by:

| Quantity     | Scale factor |
|--------------|--------------|
| velocity     | `D / T`      |
| acceleration | `D / T²`     |
| jerk         | `D / T³`     |

So the move always fills the whole window, nothing is solved at runtime, and the
robot limits become simple checks on the peak values.

### Limits

Defined in `RobotLimits` (conservative values, below the hardware limits):

| Limit                | Value         | Source |
|----------------------|---------------|--------|
| Max velocity         | 0.306 m/s     | firmware safe limit |
| Max acceleration     | 0.6 m/s²      | conservative choice |
| Max jerk             | 3.0 m/s³      | `a_max / 0.2 s` |
| Max angular velocity | 1.9 rad/s     | firmware default (the 2.64 rad/s override is **not** used) |
| Track width          | 0.235 m       | Create 3 documentation |
| Max angular accel.   | ≈ 5.1 rad/s²  | **estimate**: `2·a_max / track` |
| Max angular jerk     | ≈ 25.5 rad/s³ | **estimate**: `2·j_max / track` |

The angular acceleration and jerk limits are not published by the manufacturer.
They assume each wheel may accelerate as much as the body does in a straight line.
Validate them with a spin test on the real robot.

**Circles are checked per wheel.** On an arc the outer wheel moves at
`v · (1 + b/2R)` (b = track width), so its speed, acceleration and jerk are held to
`v_max`, `a_max` and `j_max`. This is stricter than limiting only the body.

### Minimum windows (what the choreographers need to know)

| Move | Minimum window | One 3/4 measure at 60 BPM (3.00 s) | at 70 BPM (2.57 s) |
|------|----------------|------|------|
| Forward / backward 0.3 m | 2.00 s | yes | yes |
| Turn 1° | 0.28 s | yes | yes |
| Turn 30° | 0.91 s | yes | yes |
| Turn 90° | 1.65 s | yes | yes |
| Circle 30°, R = 0.4 m | 1.90 s | yes | yes |
| Circle 45°, R = 0.4 m | 2.66 s | yes | **no, needs 2 measures** |
| Circle 45°, R = 0.3 m | 2.14 s | yes | yes |

* A single 3/4 measure fits the 45° circle (R = 0.4 m) only at 67.8 BPM or slower,
  and fits a forward/backward step only at 90 BPM or slower.
* A **larger** radius is harder, not easier: the arc is longer (`R · θ`).
* Turns above about 90° do not fit one measure (180° needs 3.31 s).
* Very slow moves (a tiny angle over a long window) give a **warning**: the real
  robot may not overcome friction. The thresholds are in `rules.py` and should be
  tuned on the robot.

### Tuning the feel

Two constants at the top of `src/dance_robots/scurve.py` control the profile shape
of **all** moves (fractions of the total duration `T`):

| Constant          | Default | Effect of increasing it |
|-------------------|---------|-------------------------|
| `ACCEL_FRACTION`  | 0.0     | Flat-top acceleration: **lowers** the minimum duration |
| `CRUISE_FRACTION` | 0.0     | Flatter velocity plateau: lowers peak velocity but **raises** the minimum duration |

Constraint: `2 * ACCEL_FRACTION + CRUISE_FRACTION < 1`. After changing them,
re-run the tests and the checker (the minimum windows change).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: dance_robots` | Run `uv sync` and use `uv run ...` (not bare `python`/`pytest`). In VS Code, select the `.venv` interpreter. |
| `uv: command not found` | Install uv (link above) and reopen the terminal. |
| Wrong Python version | `uv python pin 3.10` then `uv sync`. |
| Tests don't appear in the Testing panel | Re-run **Python: Configure Tests** and check the interpreter. |
| Checker says "unknown column" | Check the header spelling: `first_measure,last_measure,step,angle_deg,radius_m`. |
| Numbers like `0,4` rejected | Use `.` as the decimal point, or save with `;` as the separator. |

## Roadmap

- [x] `DanceStep` base class, `RobotLimits`, feasibility checks
- [x] `ForwardStep`, `BackwardStep`, `TurnLeftStep`, `TurnRightStep`, `CircleLeftStep`, `CircleRightStep`
- [x] Timetable and choreography CSV loaders, `Choreography` sequencer
- [x] Plain-language checker, offline simulation and demo
- [ ] ROS 2 Humble node: timer publishing `geometry_msgs/Twist` on `/cmd_vel`
- [ ] Wiring test in the TurtleBot 4 simulator, then tests on the real robot
- [ ] Spin test to validate the estimated angular limits; tune the warning thresholds

### Note for the ROS stage

**Publish rate.** The steps are continuous functions of time, so nothing in them
depends on the node rate. Tests and the simulator cover both 20 Hz and 50 Hz (final
poses agree within 2 mm). The node should compute `t` from the ROS clock on every
callback (never by counting ticks), so timer jitter does not accumulate. Each
command is held until the next tick, so the motion lags by about half a period
(25 ms at 20 Hz); `music_start_offset` absorbs this.

`rclpy` is **not** installable from PyPI: it comes with the ROS 2 Humble
installation. The offline stage above needs only uv. For the node we will run
from a sourced Humble environment (`source /opt/ros/humble/setup.bash`) and make
this package importable from there, for example by creating the venv with
`uv venv --system-site-packages`. We will document the exact steps when we get
there.