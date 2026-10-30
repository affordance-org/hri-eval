# Observations

What a policy can read, with units and frames. Joint level policies get `ctx`; command level policies get `obs`. Which of these you may use for a leaderboard result is in [rules.md](rules.md#3-information-your-policy-may-use).

## Frames and units

| | |
|---|---|
| World frame | the room frame of the recording: x and y on the floor plan, z up. Origin on the floor at a corner of the kitchen cabinet. The scenario page calls +x east and +y north |
| Pelvis frame | G1 convention: x forward, y left, z up |
| Units | SI: metres, seconds, radians, newtons, newton metres. Finger positions in metres |
| Quaternions | (w, x, y, z), MuJoCo convention |
| Yaw | rotation about world z; 0 faces +x, positive is counterclockwise seen from above |
| Time | `t` is simulated time in seconds, 0 at the start. Policy ticks are at t = 0, 0.02, 0.04, ... |

## Joint level: ctx

`policy_step(ctx)` and `reset(ctx)` receive the same `ctx` object, refreshed before every tick. All arrays are copies, so you may keep or change them.

| Field | Shape | Unit | Frame | Contents |
|---|---|---|---|---|
| `ctx.t` | float | s | | simulated time of this tick |
| `ctx.step` | int | | | physics step index (500 Hz) |
| `ctx.tick` | int | | | policy tick index (50 Hz), `step // 10` |
| `ctx.q29` | (29,) | rad | joint | joint positions, [joint order](#joint-order) |
| `ctx.dq29` | (29,) | rad/s | joint | joint velocities, same order |
| `ctx.base_pos` | (3,) | m | world | pelvis position |
| `ctx.base_quat` | (4,) | | world | pelvis orientation, (w, x, y, z) |
| `ctx.base_yaw` | float | rad | world | pelvis heading from `base_quat` |
| `ctx.base_linvel_world` | (3,) | m/s | world | pelvis linear velocity |
| `ctx.base_angvel_body` | (3,) | rad/s | pelvis | pelvis angular velocity |
| `ctx.fingers_q` | (4,) | m | joint | finger slide positions, [finger order](#finger-order) |
| `ctx.qpos` | (47,) | | | full MuJoCo `qpos`, [layout](#qpos-layout) |
| `ctx.qvel` | (45,) | | | full MuJoCo `qvel` |
| `ctx.m` | `mujoco.MjModel` | | | the episode model. Read only |
| `ctx.d` | `mujoco.MjData` | | | the live simulation data. Read only: writes are detected and fail G0 |
| `ctx.scratch()` | `mujoco.MjData` | | | a separate `MjData` set to the current state, with `mj_forward` already run. Writing to it does not change the simulation. It is the same object on every call ([below](#ctxscratch)) |
| `ctx.site(name)` | (pos (3,), R (3, 3)) | m | world | pose of a site |
| `ctx.body(name)` | (pos (3,), R (3, 3)) | m | world | pose of a body |
| `ctx.event(name, reason, **data)` | | | | write an event to `events.json` |
| `ctx.recipient` | object | | world | the person ([below](#the-person-in-ctx)) |
| `ctx.source_frame` | float | | | the recording frame the person is at, at `ctx.t` |

### ctx.scratch()

`ctx.scratch()` reuses one `MjData` for the whole episode. Every call copies the live state into it (positions, velocities, person poses, controls, time) and runs `mj_forward`, so it overwrites whatever you wrote to it before. `ctx.site(name)` and `ctx.body(name)` call `ctx.scratch()` themselves. So:

- read what you need from `ds` before the next `ctx.scratch()`, `ctx.site()` or `ctx.body()` call in the same tick, or `.copy()` it;
- if you need many poses in one tick, call `ds = ctx.scratch()` once and read `ds.site_xpos`, `ds.xpos`, `ds.xmat`, `ds.contact`, `ds.sensordata` and so on, instead of calling `ctx.site` many times (each call runs `mj_forward`).

### Examples

```python
import numpy as np
import mujoco

def policy_step(self, ctx):
    bag_pos, bag_R = ctx.body('backpack')                 # true backpack pose (world)
    tool_pos, tool_R = ctx.site('left_grasp_center')      # left gripper grasp point (world)
    to_bag = bag_pos - tool_pos

    ds = ctx.scratch()                                    # many reads in one tick, after the ctx.site/body calls
    chest = ds.xpos[ctx.m.body('recipient_spine_03').id].copy()   # the person's upper torso collider

    s = ctx.m.sensor('imu-pelvis-angular-velocity')       # IMU sensors from the model
    gyro = ds.sensordata[s.adr[0]:s.adr[0] + s.dim[0]].copy()
    ...
```

### Useful names in the model

| Kind | Names |
|---|---|
| Robot bodies | `pelvis`, `torso_link`, `left_umi_base`, `right_umi_base`, `left_umi_left_finger_holder`, ... (all G1 link names) |
| Gripper sites | `left_grasp_center`, `right_grasp_center` (between the pads), `left_fingertip_center`, `right_fingertip_center` (5.15 cm further along the approach axis) |
| IMU sites | `imu_in_pelvis`, `imu_in_torso` |
| Sensors | `imu-pelvis-angular-velocity`, `imu-pelvis-linear-acceleration`, `imu-pelvis-orientation`, `imu-torso-angular-velocity`, `imu-torso-linear-acceleration`, `imu-torso-orientation` (MuJoCo adds no noise) |
| Backpack | body `backpack`, free joint `backpack_free`, sites `bag_com`, `bag_top_center`, `bag_grasp_top`, `bag_grasp_top_flip` ([scenario page](scenario-point-fetch.md#the-backpack)) |
| Table | geoms `table_top`, `table_leg_0..3`, `table_apron_*` |
| Fixed objects | geoms `object_water_bottle`, `object_cup`, `object_candy_box` |
| Room | geom `floor`, geoms `room_*` |
| Person colliders | bodies `recipient_hand_l`, `recipient_hand_l_*`, `recipient_hand_r`, `recipient_lowerarm_l`, `recipient_lowerarm_r`, `recipient_pelvis`, `recipient_spine_01..03`, `recipient_thigh_l/r`, `recipient_calf_l/r`, `recipient_foot_l/r` |

## Joint order

`ctx.q29`, `ctx.dq29`, `ControlOutput.q_target`, `kp`, `kd`, `tau_ff` and `gravcomp` all use this order (`episode.JOINT_NAMES_29`):

| Index | Joint | Index | Joint |
|---|---|---|---|
| 0 | `left_hip_pitch_joint` | 15 | `left_shoulder_pitch_joint` |
| 1 | `left_hip_roll_joint` | 16 | `left_shoulder_roll_joint` |
| 2 | `left_hip_yaw_joint` | 17 | `left_shoulder_yaw_joint` |
| 3 | `left_knee_joint` | 18 | `left_elbow_joint` |
| 4 | `left_ankle_pitch_joint` | 19 | `left_wrist_roll_joint` |
| 5 | `left_ankle_roll_joint` | 20 | `left_wrist_pitch_joint` |
| 6 | `right_hip_pitch_joint` | 21 | `left_wrist_yaw_joint` |
| 7 | `right_hip_roll_joint` | 22 | `right_shoulder_pitch_joint` |
| 8 | `right_hip_yaw_joint` | 23 | `right_shoulder_roll_joint` |
| 9 | `right_knee_joint` | 24 | `right_shoulder_yaw_joint` |
| 10 | `right_ankle_pitch_joint` | 25 | `right_elbow_joint` |
| 11 | `right_ankle_roll_joint` | 26 | `right_wrist_roll_joint` |
| 12 | `waist_yaw_joint` | 27 | `right_wrist_pitch_joint` |
| 13 | `waist_roll_joint` | 28 | `right_wrist_yaw_joint` |
| 14 | `waist_pitch_joint` | | |

Groups: legs 0 to 11, waist 12 to 14, left arm 15 to 21, right arm 22 to 28. GR00T drives 0 to 14.

## Finger order

`ctx.fingers_q` and `ControlOutput.fingers` (`episode.FINGER_JOINTS`):

| Index | Joint |
|---|---|
| 0 | `left_umi_left_finger_joint` |
| 1 | `left_umi_right_finger_joint` |
| 2 | `right_umi_left_finger_joint` |
| 3 | `right_umi_right_finger_joint` |

Range -0.035 m (wide open) to 0.05 m (closed), see [scenario-point-fetch.md](scenario-point-fetch.md#gripper).

## qpos layout

| `qpos` index | Contents |
|---|---|
| 0:3 | pelvis position (world) |
| 3:7 | pelvis orientation (w, x, y, z) |
| 7:19 | legs (joints 0 to 11) |
| 19:22 | waist (joints 12 to 14) |
| 22:29 | left arm (joints 15 to 21) |
| 29:31 | left gripper fingers 0 and 1 |
| 31:38 | right arm (joints 22 to 28) |
| 38:40 | right gripper fingers 2 and 3 |
| 40:43 | backpack position (world, body origin at the bottom centre) |
| 43:47 | backpack orientation (w, x, y, z) |

`qvel` has 45 entries: 6 for the pelvis, 29 joints and 4 fingers in the same order, then 6 for the backpack. Prefer `ctx.q29` and `ctx.fingers_q` over slicing, or look addresses up by name:

```python
adr = ctx.m.jnt_qposadr[ctx.m.joint('backpack_free').id]
bag_pos, bag_quat = ctx.qpos[adr:adr + 3], ctx.qpos[adr + 3:adr + 7]
```

The person is not in `qpos`: their colliders are mocap bodies (`ctx.d.mocap_pos`, `ctx.d.mocap_quat`).

## The person in ctx

`ctx.recipient` is the replayed person. Read it at the current time only ([rules.md](rules.md)):

```python
P, Q, s, w = ctx.recipient.full_pose(ctx.t)   # 21 bone positions (21, 3) and quaternions (21, 4), world frame
P, Q = P.copy(), Q.copy()                      # full_pose returns the person's cached arrays: copy before changing
idx = ctx.recipient.rec.idx                    # bone name -> row in P and Q
head = P[idx['head']]
pos, R = ctx.recipient.hand_frame(ctx.t, 'l')  # palm centre and hand frame of the left hand ('r' for right)
```

`s` is the recording frame, `w` the largest arm adaptation weight (0 in point-fetch). Do not change the arrays `full_pose` returns, and do not call methods that change the person's schedule.

### Person bones

The 21 bones of the recording, in order:

`Root`, `pelvis`, `spine_01`, `spine_02`, `spine_03`, `clavicle_l`, `upperarm_l`, `lowerarm_l`, `hand_l`, `clavicle_r`, `upperarm_r`, `lowerarm_r`, `hand_r`, `neck_01`, `head`, `thigh_l`, `calf_l`, `foot_l`, `thigh_r`, `calf_r`, `foot_r`

A bone's position is its head (the joint at its start). `_l` is the person's left.

## Command level: obs

`act(obs)` and `reset(obs)` receive a dict, built fresh every tick (`hri_eval/interfaces.py`). All arrays are numpy, world frame, SI units.

| Key | Shape | Unit | Contents |
|---|---|---|---|
| `t` | float | s | simulated time |
| `q29` | (29,) | rad | joint positions, [joint order](#joint-order) |
| `dq29` | (29,) | rad/s | joint velocities |
| `base_pos` | (3,) | m | pelvis position |
| `base_quat` | (4,) | | pelvis orientation (w, x, y, z) |
| `base_yaw` | float | rad | pelvis heading |
| `grippers` | (2,) | | measured closing of the (left, right) gripper: 0 wide open, 1 at the closed end of the servo range. Empty and fully closed reads about 0.90: the pads meet before the end of the range |
| `person` | dict | m | keypoint name to world position (3,) of the recorded person at `t` |
| `objects` | dict | | object name to `{'pos': (3,), 'quat': (4,), 'center': (3,)}` |

`obs['person']` keys (`interfaces.PERSON_KEYPOINTS`):

| Keypoint | From bone | Keypoint | From bone |
|---|---|---|---|
| `pelvis` | `pelvis` | `wrist_r` | `hand_r` |
| `spine` | `spine_02` | `hip_l` | `thigh_l` |
| `chest` | `spine_03` | `knee_l` | `calf_l` |
| `neck` | `neck_01` | `ankle_l` | `foot_l` |
| `head` | `head` | `hip_r` | `thigh_r` |
| `head_top` | tip of `head` | `knee_r` | `calf_r` |
| `shoulder_l` | `upperarm_l` | `ankle_r` | `foot_r` |
| `elbow_l` | `lowerarm_l` | `hand_l` | palm centre of the left hand |
| `wrist_l` | `hand_l` | `hand_r` | palm centre of the right hand |
| `shoulder_r` | `upperarm_r` | | |
| `elbow_r` | `lowerarm_r` | | |

`obs['objects']` keys:

| Object | `pos`, `quat` | Moves |
|---|---|---|
| `backpack` | the bag's body frame, origin at the bottom centre of the bag (true simulated pose) | yes |
| `water_bottle` | the frame of its collision mesh | no, fixed to the table |
| `cup` | the frame of its collision mesh | no, fixed to the table |
| `candy_box` | the frame of its collision mesh | no, fixed to the table |

`center` is the centre of the object's collision mesh bounding box in the world frame. For the backpack it moves with the bag. Use it when you want "where the object is".

The person keypoints and the backpack pose are ideal: no noise, no occlusion. A command level result therefore uses person keypoints and object poses ([rules.md](rules.md#3-information-your-policy-may-use)).

## What is not available

| Observation | Status |
|---|---|
| Rendered RGB and depth camera images (for vision and VLA policies) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Rendered%20RGB%20and%20depth%20camera%20observations&feature=Rendered%20RGB%20and%20depth%20camera%20observations) |
| Noisy or occluded person keypoints, perception models | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Noisy%20or%20occluded%20person%20keypoints&feature=Noisy%20or%20occluded%20person%20keypoints) |
| Force-torque or tactile sensors | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Force-torque%20and%20tactile%20sensors&feature=Force-torque%20and%20tactile%20sensors) |
