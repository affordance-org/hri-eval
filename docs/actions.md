# Actions

What a policy returns and what the simulator does with it.

| Interface | Your method returns | Rate |
|---|---|---|
| Joint level | `episode.ControlOutput` from `policy_step(ctx)` | 50 Hz |
| Command level | `interfaces.Command` from `act(obs)` | 50 Hz |

Both run at 50 Hz. The output of a tick is held for the next 10 physics steps (dt 0.002 s, 500 Hz). The first tick is at t = 0.

## Joint level: ControlOutput

```python
import episode as E
E.ControlOutput(q_target, kp, kd, tau_ff=None, gravcomp=None, fingers=np.zeros(4), cmd=None,
                policy_obs=None, policy_action=None, arm_ik_targets=None, phase='', phase_reason='',
                log=None, done=False)
```

| Field | Shape | Unit | Required | Meaning |
|---|---|---|---|---|
| `q_target` | (29,) | rad | yes | joint position targets, [joint order](observations.md#joint-order) |
| `kp` | (29,) or scalar | Nm/rad | yes | stiffness per joint (a scalar is used for all 29) |
| `kd` | (29,) or scalar | Nm s/rad | yes | damping per joint |
| `tau_ff` | (29,) | Nm | no | feed-forward torque, held for the 10 steps |
| `gravcomp` | (29,) bool | | no | joints that get the robot's own gravity torque, recomputed every physics step |
| `fingers` | (4,) | m | no, default 0 | finger servo targets, [finger order](observations.md#finger-order), clipped to -0.035 to 0.05 |
| `cmd` | dict | | no | the locomotion command, logged: `vx`, `vy`, `yaw_rate`, `base_height`, `torso_rpy` |
| `policy_obs`, `policy_action` | arrays | | no | logged to `controller-log.npz` |
| `arm_ik_targets` | (14,) | | no | logged: left tool position (3) and quaternion (4), then right |
| `phase` | str | | no | phase label; every change is written to `events.json` |
| `phase_reason` | str | | no | the reason stored with the phase change |
| `log` | dict of arrays | | no | extra fixed-shape arrays, saved as `x_<key>` in `controller-log.npz` |
| `done` | bool | | no | end the episode after this tick |

Note on `fingers`: the default `np.zeros(4)` is not wide open. 0 is the stock UMI open width (83 mm gap). Wide open is -0.035 (153 mm), which is also the start value. See [scenario-point-fetch.md](scenario-point-fetch.md#gripper).

Note on smoothness: G9 checks that no arm target (`q_target[15:29]`) changes by more than 0.25 rad from one tick to the next, and no right arm target by more than 0.05 rad ([scoring.md](scoring.md#g9_smoothness)).

Note on `cmd`: if your policy lowers the pelvis on purpose, put the commanded height in `cmd['base_height']`. The no fall score lowers its pelvis floor from 0.55 m to 0.30 m for 1.0 s after any commanded `base_height` below 0.65 m ([scoring.md](scoring.md#g2_no_fall-score-no-fall)).

### The control law

At every physics step (500 Hz), with q and dq the measured joint positions and velocities:

```
tau = kp * (q_target - q) - kd * dq + tau_ff + gravcomp_mask * g_robot(qpos)
tau = clip(tau, -limit, +limit)            # URDF effort limits, per joint
ctrl[joint motors] = tau                   # plain torque motors (gain 1, no bias)
ctrl[finger servos] = clip(fingers, -0.035, 0.05)
```

- The target velocity is 0: the D term damps motion.
- `g_robot(qpos)` is the generalized gravity torque of the robot's own links (recursive Newton-Euler with zero velocity, on a scratch copy). It never includes the bag or anything the robot holds.
- The clip comes after everything is summed. `summary.json` records, per joint, the largest demanded torque over its limit and the number of saturated steps.
- Fingers are MuJoCo position servos: kp 5000 N/m, kv 30, force limited to 35 N.

### Torque limits

The official G1 29-DoF rev 1.0 URDF effort limits (`gr00t_wbc.URDF_EFFORT_LIMIT`). The model's motor ctrlrange and forcerange equal them, and G0 checks that.

| Joint | Limit [Nm] |
|---|---|
| hip pitch | 88 |
| hip roll | 139 |
| hip yaw | 88 |
| knee | 139 |
| ankle pitch | 35 |
| ankle roll | 35 |
| waist yaw | 88 |
| waist roll | 35 |
| waist pitch | 35 |
| shoulder pitch | 25 |
| shoulder roll | 25 |
| shoulder yaw | 25 |
| elbow | 25 |
| wrist roll | 25 |
| wrist pitch | 5 |
| wrist yaw | 5 |

### Helpers in episode.py

You do not have to write a walking controller. `episode.py` includes NVIDIA GR00T-WholeBodyControl (Balance and Walk ONNX policies, downloaded on the first run) for the legs and waist.

```python
import episode as E

loco = E.Gr00tLocomotion()
loco.reset()                                            # in reset(ctx)
q15 = loco.step(ctx, vx, vy, yaw_rate, base_height=0.74, torso_rpy=(0.0, 0.0, 0.0))
q, kp, kd, g = E.compose_gr00t(q15, arm14)             # arm14: 14 arm joint targets [rad]
return E.ControlOutput(q_target=q, kp=kp, kd=kd, gravcomp=g, fingers=f, cmd=loco.cmd,
                       policy_obs=loco.obs, policy_action=loco.action)
```

| Helper | What it does |
|---|---|
| `Gr00tLocomotion.step(ctx, vx, vy, yaw_rate, base_height, torso_rpy)` | one 50 Hz step of the GR00T lower-body policy; returns 15 position targets (legs, waist). `vx`, `vy` [m/s] and `yaw_rate` [rad/s] are in the pelvis heading frame. When the norm of (vx, vy, yaw_rate) is below 0.05 it runs the Balance model (stand in place), otherwise Walk. `base_height` default 0.74 m. `torso_rpy` is the torso orientation relative to the pelvis heading |
| `Gr00tLocomotion.cmd`, `.obs`, `.action` | the last command dict, observation and raw action, for logging |
| `Gr00tLocomotion.info()` | model name and the sha256 of the weights, for `policy_info()` |
| `compose_gr00t(q15, arm14, arm_kp=ARM_KP_14, arm_kd=ARM_KD_14, gravcomp_arms=True)` | the 29-joint `q_target`, `kp`, `kd` and `gravcomp` mask: GR00T gains for joints 0 to 14, arm gains and gravity compensation for 15 to 28 |
| `E.JOINT_NAMES_29`, `E.FINGER_JOINTS`, `E.CAPS_29` | joint order, finger order, torque limits |

Gains (from `hri_eval/gr00t_wbc.py`, the upstream GR00T deploy configuration):

| Joints | kp | kd |
|---|---|---|
| hip pitch, roll, yaw (each leg) | 150, 150, 150 | 2, 2, 2 |
| knee | 200 | 4 |
| ankle pitch, roll | 40, 40 | 2, 2 |
| waist yaw, roll, pitch | 250, 250, 250 | 5, 5, 5 |
| arm: shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw (`ARM_KP_14`, `ARM_KD_14`, each arm) | 100, 100, 40, 40, 20, 20, 20 | 5, 5, 2, 2, 2, 2, 2 |

You may use any gains and any controller for any joint. GR00T is a helper, not a requirement.

## Command level: Command

```python
import interfaces as I
I.Command(vx=0.0, vy=0.0, yaw_rate=0.0, arms=None, grippers=(0.0, 0.0), base_height=0.74, phase='')
```

| Field | Type | Unit | Default | Meaning |
|---|---|---|---|---|
| `vx`, `vy` | float | m/s | 0 | base velocity command to GR00T, in the pelvis heading frame, sent as given |
| `yaw_rate` | float | rad/s | 0 | turning rate command to GR00T |
| `base_height` | float | m | 0.74 | GR00T base height command |
| `arms` | 14 floats or `None` | rad | `None` | arm joint targets, left arm 7 then right arm 7 (joint order 15 to 28), clipped to the joint limits. `None` holds the last target; at the start that is the start pose |
| `grippers` | 2 floats or `None` | | (0, 0) | (left, right) in [0, 1]: 0 wide open, 1 closed. `None` holds the last value |
| `phase` | str | | `''` | optional label; a change is written to `events.json` as a `phase` event (empty means `command`) |

`act(obs)` may also return a dict with these keys, or `None` (the same as `Command()`, a zero command).

`None` and `Command()` open both grippers, because `grippers` defaults to `(0, 0)`, while `arms` defaults to holding the last target. A policy that holds the bag and returns `None` or `Command(phase=...)` drops it. Pass `grippers=None` to keep the current grip.

What the adapter (`interfaces.CommandAdapter`) does every tick:

| Part | How |
|---|---|
| legs and waist | `Gr00tLocomotion.step` with your `vx`, `vy`, `yaw_rate`, `base_height` (all zero = Balance, stand in place) |
| arms | joint PD with `ARM_KP_14`, `ARM_KD_14` and robot-only gravity compensation, tracking `arms` |
| grippers | finger target = -0.035 + g x 0.085 m for both fingers of a gripper, so 0 is wide open and 1 is the closed end |
| logging | `cmd`, the GR00T observation and action, and your gripper command (`x_cmd_grippers`) go to `controller-log.npz` |

Velocity commands are sent to GR00T as given. Large or sudden commands can make it fall; ramp them (the example ramps at 1 m/s^2 and 3 rad/s^2).

Arm targets are not smoothed either. G9 fails if any arm target changes by more than 0.25 rad from one tick to the next, or a right arm target by more than 0.05 rad ([scoring.md](scoring.md#g9_smoothness)). G9 is not one of the five scores.

## Actions we do not have yet

| Action interface | Status |
|---|---|
| End-effector pose targets (gripper pose in the world or pelvis frame, solved by IK for you) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20End-effector%20pose%20action%20interface&feature=End-effector%20pose%20action%20interface) |
| Event logging for command level policies (`ctx.event` has no command level equivalent yet, so a release decision cannot be logged for G8) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Event%20logging%20for%20command%20level%20policies&feature=Event%20logging%20for%20command%20level%20policies) |
| LeRobot policy adapter | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20LeRobot%20policy%20adapter&feature=LeRobot%20policy%20adapter) |
| ROS 2 bridge | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20ROS%202%20bridge&feature=ROS%202%20bridge) |
| Remote policy server (websocket or gRPC) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Remote%20policy%20server%20%28websocket%20or%20gRPC%29&feature=Remote%20policy%20server%20%28websocket%20or%20gRPC%29) |
