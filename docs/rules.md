# Rules

These rules keep results comparable. Most of them are enforced by the checker. The rest we check by hand before a result goes on the [leaderboard](leaderboard.md).

## 1. Your policy only reads the simulation

The policy acts only through what it returns: `ControlOutput` (joint level) or `Command` (command level). It never changes the simulation directly.

| Not allowed | How it is caught |
|---|---|
| writing `ctx.d.qpos`, `qvel`, `act` or `time` | the runner compares the state bit for bit before every physics step; any difference is counted in `summary.json` (`qpos_qvel_writes_after_init`) and fails G0 |
| setting `ctx.d.xfrc_applied` or `ctx.d.qfrc_applied` | the runner records the largest absolute value before every step; anything but 0 fails G0 |
| changing `ctx.m` (masses, limits, friction, gains, options, ...) | the runner hashes the model at every tick; a change fails G0 |
| adding equality constraints, mocap bodies or gravity compensation to the model | the runner refuses to start, and G0 fails |
| changing the scene file or anything else in `hri_eval/` | not caught by the checker: `run.py` simulates the scene file in your checkout and G0 compares the trial with that same file, so an edited scene still gives valid episodes. `report.md` records the git commit and the modified tracked files under `hri_eval/`, `run.py` and `requirements.txt`; leaderboard results need a clean checkout of a public commit |

For kinematics, distances, contacts or rendering of your own, use `ctx.scratch()`: a separate `MjData` set to the current state that you may write freely. It is reused by every `ctx.scratch()`, `ctx.site()` and `ctx.body()` call ([observations.md](observations.md#ctxscratch)).

The runner overwrites `ctx.d.ctrl` and the person's mocap poses every physics step, so writing them has no effect.

## 2. Actuation is the robot's own

- Joint torques are PD plus your feed-forward, clipped to the URDF effort limits ([actions.md](actions.md#torque-limits)).
- Gravity compensation exists only as the `gravcomp` mask in `ControlOutput`: the robot's own link weights, computed by the runner. It never includes the bag.
- Fingers are force-limited servos (35 N).
- No external forces, no constraints, no helper bodies.

## 3. Information your policy may use

The simulator gives a policy everything. That is useful for research, but results that use more information are not comparable with results that use less. So:

- **Use the current time only.** Read the state at `ctx.t`. The person is a replayed recording, so `ctx.recipient` can return poses from the future (`full_pose(t)` for any t). Calling it with a time other than `ctx.t` is not allowed.
- **Do not drive the person.** Never call methods that change the person's schedule (`reach`, `move`, `release`, `curl`, `idle`, `play`, `hold`). The scenario sets the person's behaviour. Any change shows up in `recipient-motion.json` (adaptation weights, schedules), and we check it.
- **Declare what you read.** When you submit, tick what your policy uses: robot proprioception, localization, person keypoints, object poses, the full simulator state, person contact forces. The leaderboard shows it.
- **The baseline is exempt from the current-time rule, and marked.** It reads the person's recorded pose at frame 200 (t = 2.67 s) before that time, to plan where to present the bag ([scenario-point-fetch.md](scenario-point-fetch.md#what-the-baseline-does)). The leaderboard says so in its row. Submitted policies follow the rule.
- **Mark privileged force use.** A decision that uses the person's contact forces (a real robot cannot measure them) is logged with `privileged=True` and disclosed in `disclosures()`. G8 checks this ([scoring.md](scoring.md#g8_controlled_handover)).

Information levels, from least to most:

| Level | What the policy reads |
|---|---|
| proprioception | `t`, `q29`, `dq29`, `base_quat`, `base_angvel_body`, `fingers_q`, the IMU sensors |
| plus localization | `base_pos`, `base_yaw`, `base_linvel_world` |
| plus person | person keypoints or bones at the current time |
| plus objects | the true backpack pose |
| full state | `qpos`, `qvel`, `ctx.d`, `ctx.scratch()`, contacts |
| privileged forces | the person's contact forces |

The command level interface gives proprioception, localization, person keypoints and the backpack pose. The baseline uses proprioception, localization, person keypoints (plus the frame 200 pose above) and the true backpack pose.

## 4. Time

- An episode is up to 24 s of simulated time (`--max-time 24`). Leaderboard results use 24.
- Wall-clock time is not limited. The simulation waits for each `policy_step`. `summary.json` records `wall_time_s` and `realtime_factor` so slow policies are visible.
- The policy may end the episode early with `ControlOutput(done=True)`. The gates then judge the episode up to that point, and `report.md` shows the end reason and the simulated time. G5 needs at least 1.0 s of the person holding the bag before the end. The baseline ends its own episodes at about 21.7 s.
- The policy runs at 50 Hz. Its output is held for the next 10 physics steps.

## 5. Report every episode

Seeds are fixed: seed 0 is the nominal recorded episode, seeds 1 and up are variations ([batch-and-variations.md](batch-and-variations.md)). Report the whole batch you ran (`report.md`), not the best episodes. Leaderboard results use `--episodes 10 --seed 0`.

## 6. Disclose your assumptions

Return a list of sentences from `disclosures()` for anything a reader should know: privileged information, assumed sensors, tuned constants. They are written to `summary.json` and the trial `README.md`.
