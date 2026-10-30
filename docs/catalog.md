# Catalog: everything you can test

**Available** means it is in this repository and works today. **On request** means it is not built yet: the link opens a pre-filled GitHub issue, and we build items in the order people ask for them. `python run.py --list` prints the same items in the terminal, each on-request item with its link.

Not on the list? [Request a scenario](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20), [a robot](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20) or [a feature](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20).

## Scenarios

| Scenario | Status |
|---|---|
| `point-fetch`: a person points at a backpack on a low table; the robot fetches it and hands it over ([details](scenario-point-fetch.md)) | Available |
| Handover to a seated person | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20Handover%20to%20a%20seated%20person&scenario=Handover%20to%20a%20seated%20person) |
| Passing a person in a narrow space | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20Passing%20a%20person%20in%20a%20narrow%20space&scenario=Passing%20a%20person%20in%20a%20narrow%20space) |
| Tidying a table next to a working person | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20Tidying%20a%20table%20next%20to%20a%20working%20person&scenario=Tidying%20a%20table%20next%20to%20a%20working%20person) |
| Sit-to-stand help | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20Sit-to-stand%20help&scenario=Sit-to-stand%20help) |
| Carrying with a person (an object carried together) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20Carrying%20with%20a%20person&scenario=Carrying%20with%20a%20person) |

## People and rooms

| Item | Status |
|---|---|
| One real person, recorded with cameras in one real room, replayed in the room rebuilt in MuJoCo; hands, forearms and body are colliders fitted to the visible skin | Available |
| Reactive humans that respond to the robot (reach for the bag, step aside, wait) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Reactive%20humans%20that%20respond%20to%20the%20robot&feature=Reactive%20humans%20that%20respond%20to%20the%20robot) |
| More recorded people and rooms | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20More%20recorded%20people%20and%20rooms&feature=More%20recorded%20people%20and%20rooms) |
| Capturing your own room and people | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Capturing%20your%20own%20room%20and%20people&feature=Capturing%20your%20own%20room%20and%20people) |

## Robots

| Robot | Status |
|---|---|
| Unitree G1, 29 joints, two UMI parallel grippers (35 N per finger) | Available |
| G1 with Dex3 hands | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20G1%20with%20Dex3%20hands&robot_name=G1%20with%20Dex3%20hands) |
| Unitree H1 | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20Unitree%20H1&robot_name=Unitree%20H1) |
| Fourier GR-1 | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20Fourier%20GR-1&robot_name=Fourier%20GR-1) |
| Mobile manipulators such as Hello Robot Stretch | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20Mobile%20manipulators%20such%20as%20Hello%20Robot%20Stretch&robot_name=Mobile%20manipulators%20such%20as%20Hello%20Robot%20Stretch) |
| Your own URDF or MJCF | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20Your%20own%20URDF%20or%20MJCF&robot_name=Your%20own%20URDF%20or%20MJCF) |

## Policy interfaces

| Interface | Status |
|---|---|
| Joint level: `reset(ctx)`, `policy_step(ctx) -> episode.ControlOutput` (29 joint targets and gains, feed-forward torque, gravity compensation mask, 4 finger targets) | Available ([example](../examples/stand_still.py)) |
| Command level: `interfaces.CommandPolicy.act(obs) -> interfaces.Command` (base velocity to GR00T, 14 arm joint targets, 2 gripper values) | Available ([example](../examples/walk_to_table.py)) |
| GR00T WBC walking helper for joint level policies (`episode.Gr00tLocomotion`, `episode.compose_gr00t`) | Available |
| `--params` JSON passed to your constructor | Available |
| End-effector pose action interface | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20End-effector%20pose%20action%20interface&feature=End-effector%20pose%20action%20interface) |
| Event logging for command level policies (a logged release decision for G8, like `ctx.event` at the joint level) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Event%20logging%20for%20command%20level%20policies&feature=Event%20logging%20for%20command%20level%20policies) |
| LeRobot policy adapter | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20LeRobot%20policy%20adapter&feature=LeRobot%20policy%20adapter) |
| ROS 2 bridge | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20ROS%202%20bridge&feature=ROS%202%20bridge) |
| Remote policy server (websocket or gRPC) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Remote%20policy%20server%20%28websocket%20or%20gRPC%29&feature=Remote%20policy%20server%20%28websocket%20or%20gRPC%29) |

## Observations

| Observation | Status |
|---|---|
| Robot proprioception: joint positions and velocities, pelvis pose and velocities, finger positions | Available (pelvis velocities and finger positions at the joint level; `obs['grippers']` at the command level) |
| IMU sensors in the model (pelvis and torso gyro, accelerometer, orientation) | Available (joint level, via `ctx.scratch().sensordata`) |
| Person keypoints at the current time (20 points, ideal) | Available (`obs['person']`; joint level via `ctx.recipient`) |
| Object poses (true backpack pose; the fixed table objects) | Available (`obs['objects']`; joint level via `ctx.body`) |
| Full simulator state, read only (`ctx.qpos`, `ctx.d`, `ctx.scratch()`) | Available (joint level) |
| Rendered RGB and depth camera observations, for vision and VLA policies | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Rendered%20RGB%20and%20depth%20camera%20observations&feature=Rendered%20RGB%20and%20depth%20camera%20observations) |
| Noisy or occluded person keypoints | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Noisy%20or%20occluded%20person%20keypoints&feature=Noisy%20or%20occluded%20person%20keypoints) |
| Force-torque and tactile sensors | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Force-torque%20and%20tactile%20sensors&feature=Force-torque%20and%20tactile%20sensors) |

## Scores

| Score | Status |
|---|---|
| no fall, pickup, carry, handover, pace (printed by `run.py`, in `report.json`; totals count valid episodes only) | Available ([scoring.md](scoring.md)) |
| valid: the checker's G0 invariants and G1 motor replay pass | Available |
| Independent checker gates G0 to G9, re-simulated from the trial files (`gates.json`) | Available |
| Minimum robot to person distance per episode (a measurement, not a score) | Available |
| Comfort and proxemics scores | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Comfort%20and%20proxemics%20scores&feature=Comfort%20and%20proxemics%20scores) |

## Variations and batches

| Item | Status |
|---|---|
| `--episodes N --seed S`: seeded episodes; seed 0 is the nominal recorded episode | Available |
| Robot start offset (dx, dy within 0.10 m, yaw within 10 deg) | Available |
| Backpack start offset on the table (dx, dy within 0.05 m) | Available |
| Person playback speed (0.9 to 1.1) | Available |
| `--jobs J`: episodes in parallel processes | Available |
| More variation types (person position, other objects, table height) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20More%20variation%20types&feature=More%20variation%20types) |

## Tooling

| Tool | Status |
|---|---|
| `run.py`: run, score and report in one command | Available |
| `report.json` and `report.md` per run, with the git commit and any local changes | Available |
| Trial folders with full trajectories, logs and events | Available ([trial-format.md](trial-format.md)) |
| `hri_eval/check_gates.py`: re-score any trial folder | Available |
| `python run.py --list` | Available |
| Trial viewer and video export | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Trial%20viewer%20and%20video%20export&feature=Trial%20viewer%20and%20video%20export) |

## Evaluation services

| Service | Status |
|---|---|
| Leaderboard table in this repository, updated from submitted results | Available ([leaderboard.md](leaderboard.md)) |
| Hosted evaluation: send us a checkpoint, we run it | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=hosted_eval.yml&title=%5BHosted%20eval%5D%20) |
| Public leaderboard service | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Public%20leaderboard%20service&feature=Public%20leaderboard%20service) |
| Real-robot replay in the same room | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Real-robot%20replay%20in%20the%20same%20room&feature=Real-robot%20replay%20in%20the%20same%20room) |
