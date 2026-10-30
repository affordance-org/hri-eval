# Trial format

Every episode writes one trial folder. With `--episodes N` the folders are `DIR/ep000`, `DIR/ep001`, ... next to `DIR/report.json` and `DIR/report.md` ([batch-and-variations.md](batch-and-variations.md)).

A trial folder is self-contained evidence: the checker re-scores it from its own files, and anyone can re-check it.

| File | Written by | Rate | Contents |
|---|---|---|---|
| `trial.xml` | runner, before the episode | | the exact scene the episode simulated |
| `trajectory.npz` | runner | every physics step (500 Hz) | full state and inputs |
| `controller-log.npz` | runner | every policy tick (50 Hz) | what the policy commanded |
| `telemetry.json` | runner | 50 Hz | readable per-tick summary: pose, bag, contacts, torques |
| `events.json` | runner and policy | when they happen | phase changes and policy events |
| `summary.json` | runner | once | episode facts, measured invariants, disclosures |
| `scene-context.json` | runner | once | start poses and scene facts |
| `recipient-motion.json` | runner | 30 fps | the person's bones as simulated |
| `README.md` | runner | once | what drove what, invariants, disclosures |
| `gates.json` | checker | once | all ten gates ([scoring.md](scoring.md)) |
| `variation.json` | `run.py` | once | the seed and its variation values ([batch-and-variations.md](batch-and-variations.md#variations)) |

## trial.xml

The scene file the episode ran, as text, with `meshdir` and `texturedir` made absolute so it compiles from any working directory. The runner writes it first and then loads the model from it, and the checker loads the same file. Its sha256 is in `summary.json` (`model.trial_xml_sha256`) and in `gates.json`.

Because the asset paths are absolute, re-check a trial on the machine that ran it, or on a machine with the repository at the same path.

With a variation seed the varied start (robot pose, backpack pose) is written into this file before the model is loaded, so the runner and the checker start from the identical state.

## trajectory.npz

One row per physics step, plus row 0 for the initial state. N = number of physics steps + 1.

| Key | Shape | Contents |
|---|---|---|
| `time` | (N,) | simulation time [s], 0, 0.002, 0.004, ... |
| `qpos` | (N, 47) | MuJoCo `qpos` ([observations.md](observations.md#qpos-layout)) |
| `qvel` | (N, 45) | MuJoCo `qvel` |
| `ctrl` | (N, 33) | actuator inputs: 29 joint torques [Nm], then 4 finger servo targets [m] |
| `qfrc_actuator` | (N, 45) | actuator generalized forces |
| `act` | (N, 0) | actuator activations (none in this model) |
| `mocap_pos` | (N, 62, 3) | position of each person collider body |
| `mocap_quat` | (N, 62, 4) | orientation (w, x, y, z) of each person collider body |

Row i is the state after physics step i. `ctrl[i]`, `mocap_pos[i]` and `mocap_quat[i]` are the inputs applied during the step that produced row i. The checker replays exactly these inputs.

```python
import numpy as np
tr = np.load('trials/MyPolicy/ep000/trajectory.npz')
pelvis_xyz = tr['qpos'][:, 0:3]
bag_xyz = tr['qpos'][:, 40:43]
```

## controller-log.npz

One row per policy tick (50 Hz).

| Key | Shape | Contents |
|---|---|---|
| `time` | (T,) | tick time [s] |
| `phase_id` | (T,) | index into `phase_names` |
| `phase_names` | (P,) | phase labels in order of first use |
| `q_target` | (T, 29) | joint position targets [rad] |
| `kp`, `kd` | (T, 29) | PD gains |
| `tau_ff` | (T, 29) | feed-forward torque [Nm] (zeros if unused) |
| `gravcomp_mask` | (T, 29) | which joints got gravity compensation |
| `finger_target` | (T, 4) | finger servo targets [m] |
| `cmd` | (T, 7) | `vx, vy, yaw_rate, base_height, torso_roll, torso_pitch, torso_yaw` (NaN if not given) |
| `cmd_columns` | (7,) | the column names above |
| `arm_joint_target` | (T, 14) | `q_target[15:29]` |
| `arm_ik_targets` | (T, 14) | left tool position and quaternion, then right (NaN if unused) |
| `base_pos`, `base_yaw` | (T, 3), (T,) | pelvis pose the policy saw |
| `policy_obs`, `policy_action` | (T, ...) | the locomotion policy's observation and action, if given |
| `x_<name>` | (T, ...) | every array the policy put in `ControlOutput.log` |
| `joint_names` | (29,) | joint order |

The checker reads `cmd` (G2 crouch floor), `finger_target` (G8 release) and `q_target` (G9).

## telemetry.json

A list with one row per tick. Each row:

| Key | Contents |
|---|---|
| `time`, `phase` | tick time, phase label |
| `base_pos`, `base_yaw`, `base_height` | pelvis pose |
| `tilt_deg`, `torso_tilt_deg` | pelvis and torso tilt from vertical |
| `left_pad_normal_N`, `right_pad_normal_N` | normal force between each gripper's fingers and the bag |
| `bag_pos`, `bag_quat` | backpack pose |
| `bag_min_z`, `bag_clearance_above_table` | lowest bag point, and its height above the table top |
| `bag_grasp_sites` | world position of each `bag_grasp_*` site |
| `recipient_pad_normal_N` | normal force between the bag and each person collider body |
| `torque_ratio_max` | largest torque / limit per joint group (`legs`, `ankles`, `waist`, `arms`, `wrist_pitch_yaw`) |
| `contacts` | every contact: geoms, bodies, position, normal, force, distance |
| `reference_frame` | the recording frame the person is at |
| `recipient_reach_weight`, `recipient_gap_m`, `recipient_curl_l`, `recipient_source_frame_rate` | the person's arm adaptation state (no adaptation in point-fetch) |

Telemetry is for reading and plotting. Scores never use it.

## events.json

A list of events in time order:

```json
[
 {"time": 0.0, "event": "phase", "reason": "", "phase": "stand", "previous": null},
 {"time": 2.84, "event": "release", "reason": "wrist load dropped", "signals": ["arm_joint_torque"]}
]
```

The runner adds a `phase` event whenever `ControlOutput.phase` changes, and an `exception` event with the traceback if the policy raises. Your policy adds events with `ctx.event(name, reason, **data)`. G8 reads events ([scoring.md](scoring.md#g8_controlled_handover)).

## summary.json

| Key | Contents |
|---|---|
| `controller_execution`, `diagnostic_only` | `closed_loop_full_episode`, `false` |
| `description` | one line about the run |
| `episode` | `physics_steps`, `sim_time_s`, `wall_time_s`, `realtime_factor`, `end_reason` (`max_time`, `controller_done`, `diverged` or `exception`), `policy_ticks`, `initial_state` |
| `policy` | what your `policy_info()` returned |
| `physics` | timestep, integrator, cone, impratio, iterations, gravity, MuJoCo version, `policy_hz` 50, `pd_hz` 500 |
| `model` | sha256 of `trial.xml`, of the source scene and of every asset file |
| `invariants_measured` | what the runner measured: state writes after init, max `xfrc_applied` and `qfrc_applied`, model changes during the run, model fingerprint, and per joint torque limits, max applied torque, max demanded torque / limit and saturated steps |
| `outcome` | filled only when the runner itself runs the checker; with `run.py` read `gates.json` |
| `timings` | `human_duration_s` 4.4, `human_frames` [120, 252] |
| `failure` | `null`, or why the episode stopped early |
| `disclosures` | the scenario's disclosed modelling choices (the same for every policy: the upright backpack, its contact model, the gripper, the replayed person, the robot start pose), then your policy's `disclosures()`, then the seeded variation if any |

## scene-context.json

Facts about the start of the episode:

| Key | Contents |
|---|---|
| `frame`, `room_from_physics_translation`, `room_offset` | the room frame of the recording; the physics uses it directly (zero offset) |
| `table_top_z` | height of the table top [m] |
| `bag_start_pose` | the backpack start position and orientation (with the variation applied), its grasp sites in the bag frame, and its mass |
| `robot_start` | `keyframe`, the start pelvis position `pelvis_pos` and heading `pelvis_yaw_rad` (with the variation applied), and `source`: for point-fetch "chosen for this scenario, not taken from the recording" |
| `recipient` | `person` (the recording id), `kind` (how the person was driven in this episode, from the person's own decision log: in point-fetch a replayed recording, pointing gesture then idle ping-pong, not adapted, finger curl 0), `mocap_bodies` (the 62 collider bodies) and `schedules` (the person's playback schedule) |
| `recording` | the `hri_eval/` folder of the checkout that ran the episode (the recording is in its `person/` folder) |

## recipient-motion.json

The person exactly as simulated, at 30 fps of simulation time, in the `h2r-avatar-motion/1` format: the 21 bones with rest pose, and per frame `rotation_wxyz` and `position` of every bone in the room frame, `source_frame` (recording frame), `robot_time_s`, the arm adaptation weights (0 in point-fetch) and `curl_l` (0 in point-fetch). `robot_time.robot_time_to_source_frame` maps simulation time to recording frame. G7 compares this file with the simulated colliders.

## README.md

A human readable page: the scene and episode length, what drove what (policy, PD, finger servos, person), the measured invariants, and every disclosure (the scenario's, then your policy's, then the seeded variation). Your policy's `describe()` and `disclosures()` appear here. The baseline's own disclosures (its perception, its privileged information, that it never lets go) appear only in baseline trials.

## gates.json

Written by the checker. See [scoring.md](scoring.md#gatesjson).

## variation.json

The seed, the six variation values and `nominal` (true for seed 0). The checker recomputes the variation from the seed and fails G0 if the values differ. See [batch-and-variations.md](batch-and-variations.md#variations).

## Viewing a trial

There is no viewer or video export in the repository yet. It is [on request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Trial%20viewer%20and%20video%20export&feature=Trial%20viewer%20and%20video%20export).
