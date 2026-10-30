# HRI Eval

![HRI Eval](https://github.com/user-attachments/assets/7c36e5ae-86d0-4709-bec9-ab9d6dfe78f0)

## Test your robot policy here.

Evaluating robots around people is hard. Testing with real people isn't safe, and simulated people don't move like real ones.

We recorded a real person in a real room and rebuilt both in simulation. Roll out your robot policy there and get scored.

**[Quick start](docs/quickstart.md) · [Write a policy](docs/write-a-policy.md) · [What you can test](#what-you-can-test) · [Ask for a feature](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20)**

## Quick start

```bash
git clone https://github.com/affordance-org/hri-eval && cd hri-eval
pip install -r requirements.txt                                                  # Python 3.12 or newer
python run.py                                                                    # our baseline, scored
python run.py --policy examples.walk_to_table:WalkToTable --episodes 10 --jobs 4  # an example policy, 10 seeded episodes
python run.py --policy my_policy:MyPolicy --episodes 10 --jobs 4                 # yours (see below)
```

CPU only. The first run downloads the GR00T walking policy weights. `python run.py --list` prints the full catalog, with a request link for every On request item.

## What you can test

| Area | What | Status |
|---|---|---|
| Scenarios | `point-fetch`: a person points at a backpack on a table; fetch it and hand it over ([details](docs/scenario-point-fetch.md)) | Available |
| | Handover to a seated person | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20Handover%20to%20a%20seated%20person&scenario=Handover%20to%20a%20seated%20person) |
| | Passing a person in a narrow space | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20Passing%20a%20person%20in%20a%20narrow%20space&scenario=Passing%20a%20person%20in%20a%20narrow%20space) |
| | Tidying a table next to a working person | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20Tidying%20a%20table%20next%20to%20a%20working%20person&scenario=Tidying%20a%20table%20next%20to%20a%20working%20person) |
| | Sit-to-stand help | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20Sit-to-stand%20help&scenario=Sit-to-stand%20help) |
| | Carrying with a person | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20Carrying%20with%20a%20person&scenario=Carrying%20with%20a%20person) |
| Robots | Unitree G1 (29 joints) with two UMI grippers | Available |
| | G1 with Dex3 hands | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20G1%20with%20Dex3%20hands&robot_name=G1%20with%20Dex3%20hands) |
| | Unitree H1 | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20Unitree%20H1&robot_name=Unitree%20H1) |
| | Fourier GR-1 | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20Fourier%20GR-1&robot_name=Fourier%20GR-1) |
| | Mobile manipulators such as Hello Robot Stretch | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20Mobile%20manipulators%20such%20as%20Hello%20Robot%20Stretch&robot_name=Mobile%20manipulators%20such%20as%20Hello%20Robot%20Stretch) |
| | Your own URDF or MJCF | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20Your%20own%20URDF%20or%20MJCF&robot_name=Your%20own%20URDF%20or%20MJCF) |
| Policy interfaces | Joint level: 29 joint targets and gains, 4 finger targets | Available |
| | Command level: base velocity (GR00T walks), 14 arm joints, 2 grippers | Available |
| | End-effector pose actions | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20End-effector%20pose%20action%20interface&feature=End-effector%20pose%20action%20interface) |
| | LeRobot policy adapter | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20LeRobot%20policy%20adapter&feature=LeRobot%20policy%20adapter) |
| | ROS 2 bridge | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20ROS%202%20bridge&feature=ROS%202%20bridge) |
| | Remote policy server (websocket or gRPC) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Remote%20policy%20server%20%28websocket%20or%20gRPC%29&feature=Remote%20policy%20server%20%28websocket%20or%20gRPC%29) |
| | Event logging for command level policies (for example a logged release decision) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Event%20logging%20for%20command%20level%20policies&feature=Event%20logging%20for%20command%20level%20policies) |
| Observations | Proprioception and pelvis pose | Available |
| | IMU sensors (joint level) | Available |
| | Person keypoints, object poses (ideal), full simulator state (read only, joint level) | Available |
| | Rendered RGB and depth cameras, for vision and VLA policies | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Rendered%20RGB%20and%20depth%20camera%20observations&feature=Rendered%20RGB%20and%20depth%20camera%20observations) |
| | Noisy or occluded person keypoints | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Noisy%20or%20occluded%20person%20keypoints&feature=Noisy%20or%20occluded%20person%20keypoints) |
| | Force-torque and tactile sensors | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Force-torque%20and%20tactile%20sensors&feature=Force-torque%20and%20tactile%20sensors) |
| Humans | One real person, recorded in one real room, replayed with colliders fitted to their skin | Available |
| | Reactive humans that respond to the robot | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Reactive%20humans%20that%20respond%20to%20the%20robot&feature=Reactive%20humans%20that%20respond%20to%20the%20robot) |
| | More recorded people and rooms | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20More%20recorded%20people%20and%20rooms&feature=More%20recorded%20people%20and%20rooms) |
| | Capturing your own room and people | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Capturing%20your%20own%20room%20and%20people&feature=Capturing%20your%20own%20room%20and%20people) |
| Scores | No fall, pickup, carry, handover, pace; checker gates G0 to G9; min robot to person distance | Available |
| | Comfort and proxemics scores | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Comfort%20and%20proxemics%20scores&feature=Comfort%20and%20proxemics%20scores) |
| Tooling | Seeded variations, batches, parallel jobs, `report.json` / `report.md`, independent re-simulating checker | Available |
| | More variation types (person position, other objects, table height) | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20More%20variation%20types&feature=More%20variation%20types) |
| | Trial viewer and video export | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Trial%20viewer%20and%20video%20export&feature=Trial%20viewer%20and%20video%20export) |
| Services | Leaderboard in this repository | Available |
| | Hosted evaluation: send us a checkpoint, we run it | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=hosted_eval.yml&title=%5BHosted%20eval%5D%20) |
| | Public leaderboard service | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Public%20leaderboard%20service&feature=Public%20leaderboard%20service) |
| | Real-robot replay in the same room | [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Real-robot%20replay%20in%20the%20same%20room&feature=Real-robot%20replay%20in%20the%20same%20room) |

Full list: [docs/catalog.md](docs/catalog.md).

## Write your policy

Command level: set the walking velocity, the arm joint targets and the grippers. GR00T walks for you. Save this as `my_policy.py` in the repository root:

```python
import interfaces as I

class MyPolicy(I.CommandPolicy):
    def act(self, obs):                              # 50 Hz; obs: robot state, obs['person'], obs['objects']
        bag = obs['objects']['backpack']['center']   # where the backpack is (world frame, metres)
        hand = obs['person']['hand_l']               # the person's left palm
        return I.Command(vx=0.0, vy=0.0, yaw_rate=0.0, arms=None, grippers=(0.0, 0.0))   # stands still: add yours
```

```bash
python run.py --policy my_policy:MyPolicy
```

It stands still, so it passes no fall and nothing else. A walking command sent without looking (say `vx=0.2` every tick) walks the robot into the table and the person. [`examples/walk_to_table.py`](examples/walk_to_table.py) shows how to steer.

Joint level: control every joint. A sketch (the complete version is [`examples/stand_still.py`](examples/stand_still.py)):

```python
import episode as E

class MyPolicy:
    def reset(self, ctx): ...
    def policy_step(self, ctx):              # 50 Hz; ctx: full read-only simulation
        ...                                  # compute q, kp, kd (29 joints) and f (4 finger targets)
        return E.ControlOutput(q_target=q, kp=kp, kd=kd, fingers=f)
```

Examples: [`examples/walk_to_table.py`](examples/walk_to_table.py) (command level), [`examples/stand_still.py`](examples/stand_still.py) (joint level). Step by step: [docs/write-a-policy.md](docs/write-a-policy.md).

## Docs

Index: [docs/README.md](docs/README.md)

| | |
|---|---|
| [Quick start](docs/quickstart.md) | install, run, read the output |
| [Write a policy](docs/write-a-policy.md) | both interfaces with complete examples |
| [Scenario: point-fetch](docs/scenario-point-fetch.md) | room, robot, backpack, the person's timeline |
| [Observations](docs/observations.md) | every field, units, frames, joint order |
| [Actions](docs/actions.md) | ControlOutput, PD law, torque limits, Command |
| [Scoring](docs/scoring.md) | the five scores and gates G0 to G9 |
| [Batches and variations](docs/batch-and-variations.md) | seeds, variations, jobs, report.json |
| [Trial format](docs/trial-format.md) | every file in a trial folder |
| [Rules](docs/rules.md) | fair play |
| [Catalog](docs/catalog.md) | available and on request |
| [Leaderboard](docs/leaderboard.md) | results and how to submit |
| [FAQ](docs/faq.md) | short answers |
| [How it was made](docs/how-it-was-made.md) | real to sim |

## Leaderboard

| Policy | Interface | Seeds | No fall | Pickup | Carry | Handover | Pace |
|---|---|---|---|---|---|---|---|
| Baseline (reads the person's frame 200 pose early: exempt, [see rules](docs/rules.md#3-information-your-policy-may-use)) | joint | 0 to 9 | 10/10 | 10/10 | 10/10 | 0/10 | 0/10 |

The baseline fetches the bag and holds it out but never lets go. Beat it: [submit your report.md](https://github.com/affordance-org/hri-eval/issues/new?template=submit_result.yml&title=%5BResult%5D%20). Details: [docs/leaderboard.md](docs/leaderboard.md).

## Don't see what you need? Ask for it

We build what people ask for, in that order.

- [Request a scenario](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20)
- [Request a robot](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20)
- [Request a feature](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20): interfaces, observations, scores, tooling, people and rooms
- [Request a hosted evaluation](https://github.com/affordance-org/hri-eval/issues/new?template=hosted_eval.yml&title=%5BHosted%20eval%5D%20)
- [Submit a result](https://github.com/affordance-org/hri-eval/issues/new?template=submit_result.yml&title=%5BResult%5D%20)
- [Report a bug](https://github.com/affordance-org/hri-eval/issues/new?template=bug.yml&title=%5BBug%5D%20)

[affordance.homes](https://affordance.homes) · code Apache-2.0 · data CC BY 4.0 ([licenses](licenses))
