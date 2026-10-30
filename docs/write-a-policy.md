# Write a policy

A policy is one Python class. `run.py` imports it, builds it, and calls it at 50 Hz for the whole episode.

| | Joint level | Command level |
|---|---|---|
| You write | `reset(ctx)`, `policy_step(ctx)` | a subclass of `interfaces.CommandPolicy` with `act(obs)` |
| You return | `episode.ControlOutput`: 29 joint targets and gains, 4 finger targets | `interfaces.Command`: base velocity, 14 arm joint targets, 2 gripper values |
| Walking | yours, or the GR00T helper | GR00T runs for you |
| You read | everything in the simulation (read only) | robot state, person keypoints, object poses |
| Example | [`examples/stand_still.py`](../examples/stand_still.py) | [`examples/walk_to_table.py`](../examples/walk_to_table.py) |
| Fields | [observations.md](observations.md#joint-level-ctx), [actions.md](actions.md#joint-level-controloutput) | [observations.md](observations.md#command-level-obs), [actions.md](actions.md#command-level-command) |

Pick the command level if you want to work on what the robot does (where to walk, where to put the arms, when to grasp). Pick the joint level if you want to control the whole body yourself.

`run.py` decides which one your class is: a subclass of `CommandPolicy` (or any class with `act` and no `policy_step`) is command level, everything else is joint level.

## Step 1: run the examples

```bash
python run.py --policy examples.stand_still:StandStill       # joint level
python run.py --policy examples.walk_to_table:WalkToTable    # command level
```

Each writes `trials/<ClassName>/ep000/` (the trial), `trials/<ClassName>/report.json` and `report.md`, and prints the scores. Neither example picks the bag up, so pickup, carry, handover and pace fail. That is expected.

A run replaces what an earlier run wrote to the same folder. Give each experiment its own `--out` (for example `--out trials/walk_standoff_06`) to keep the earlier results.

## Step 2: the joint level example

`examples/stand_still.py`, complete:

```python
"""The smallest policy: GR00T balance with a zero walking command, arms held at the start pose.
python run.py --policy examples.stand_still:StandStill"""
import numpy as np

import episode as E


class StandStill:
    def __init__(self):
        self.loco = E.Gr00tLocomotion()

    def reset(self, ctx):
        self.loco.reset()
        self.arm = ctx.q29[15:].copy()

    def policy_step(self, ctx):
        q15 = self.loco.step(ctx, 0.0, 0.0, 0.0, base_height=0.74)
        q, kp, kd, g = E.compose_gr00t(q15, self.arm)
        return E.ControlOutput(q_target=q, kp=kp, kd=kd, gravcomp=g, fingers=np.zeros(4), cmd=self.loco.cmd,
                               policy_obs=self.loco.obs, policy_action=self.loco.action, phase='stand')

    def policy_info(self):
        return self.loco.info()

    def describe(self):
        return 'GR00T WBC Balance with a zero command; arms hold the start pose; grippers open.'

    def disclosures(self):
        return []
```

Line by line:

1. `import episode as E`: `run.py` puts `hri_eval/` on the Python path, so `episode`, `interfaces`, `gr00t_wbc` and the other modules import by name.
2. `__init__` builds the GR00T walking policy once. Constructor keywords come from `--params` (step 5).
3. `reset(ctx)` runs once before the first tick. Here it stores the start arm pose (`ctx.q29[15:]`, the 14 arm joints).
4. `policy_step(ctx)` runs every 20 ms. It asks GR00T for the 15 leg and waist targets with a zero velocity command (Balance model), joins them with the held arm pose (`compose_gr00t` adds the arm gains and gravity compensation), and returns a `ControlOutput`.
5. `fingers=np.zeros(4)` sets the finger targets to 0 m, the stock UMI open width.
6. `cmd`, `policy_obs`, `policy_action` and `phase` are only logged.
7. `policy_info`, `describe` and `disclosures` are optional. They fill `summary.json` and the trial `README.md`.

## Step 3: the command level example

`examples/walk_to_table.py`, complete:

```python
"""Command level: walk to the table, stop in front of the backpack facing it, and stand. Arms hold the start pose,
grippers stay open. python run.py --policy examples.walk_to_table:WalkToTable"""
import math

import numpy as np

import interfaces as I


class WalkToTable(I.CommandPolicy):
    def __init__(self, standoff=0.50, v_max=0.35, wz_max=0.6, tol=0.05, yaw_tol_deg=4.0):
        self.standoff, self.v_max, self.wz_max, self.tol = standoff, v_max, wz_max, tol
        self.yaw_tol = math.radians(yaw_tol_deg)

    def reset(self, obs):
        bag = obs['objects']['backpack']['pos']
        self.goal = np.array([bag[0] - self.standoff, bag[1]])   # west of the table, facing east (+x) toward the bag
        self.goal_yaw = 0.0
        self.mode = 'turn'
        self.v = np.zeros(3)

    def act(self, obs):
        e = self.goal - obs['base_pos'][:2]
        dist = float(np.linalg.norm(e))
        yaw = obs['base_yaw']
        to_goal = I.wrap(math.atan2(e[1], e[0]) - yaw)
        to_final = I.wrap(self.goal_yaw - yaw)
        des = np.zeros(3)
        if self.mode == 'turn':                     # face the stance point
            des[2] = self.turn(to_goal)
            if abs(to_goal) < 0.1:
                self.mode = 'walk'
        if self.mode == 'walk':                     # walk to it, steering toward it; the last 15 cm without steering
            if dist > 0.15:
                des[0] = min(self.v_max, 0.8 * dist) * max(0.0, math.cos(to_goal))
                des[2] = float(np.clip(1.5 * to_goal, -self.wz_max, self.wz_max))
            else:
                c, s = math.cos(yaw), math.sin(yaw)
                des[:2] = 0.12 * np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]]) / max(dist, 1e-6)
            if dist < self.tol or obs['t'] > 15.0:
                self.mode = 'align'
        if self.mode == 'align':                    # turn in place to face the bag
            des = np.array([0.0, 0.0, self.turn(to_final)])
            if abs(to_final) < self.yaw_tol:
                self.mode = 'stand'
        if self.mode == 'stand':
            self.v[:] = 0.0
            return I.Command(phase='stand')
        lim = np.array([0.02, 0.02, 0.06])        # per 50 Hz tick: 1 m/s^2, 3 rad/s^2
        self.v += np.clip(des - self.v, -lim, lim)
        return I.Command(vx=self.v[0], vy=self.v[1], yaw_rate=self.v[2], grippers=(0.0, 0.0), phase=self.mode)

    def turn(self, err):
        return math.copysign(min(self.wz_max, max(0.15, 1.5 * abs(err))), err)

    def describe(self):
        return (f'Walks to a point {self.standoff:.2f} m west of the backpack (the TRUE bag pose at the start), turns to '
                'face it (east) and stands; arms hold the start pose, grippers open.')
```

What it shows:

1. `reset(obs)` reads the backpack position once and sets a goal 0.5 m west of it.
2. `act(obs)` is a small state machine: `turn` to face the goal, `walk` there, `align` to face east, `stand`.
3. Velocities ramp by at most 0.02 m/s and 0.06 rad/s per tick, so GR00T gets smooth commands.
4. `I.Command(phase='stand')` is a zero command: GR00T switches to its Balance model and the robot stands. A zero command also opens both grippers, because `grippers` defaults to `(0, 0)`. A policy that holds something passes `grippers=None` to keep its grip.
5. `arms` is never set, so the arms hold the start pose. `grippers=(0.0, 0.0)` keeps both grippers wide open.
6. `I.wrap(angle)` wraps an angle to [-pi, pi).

## Step 4: write your own

Create a file anywhere, for example `my_policy.py` in the repository root:

```python
import numpy as np
import interfaces as I


class MyPolicy(I.CommandPolicy):
    def __init__(self, speed=0.2):             # keywords come from --params (step 5)
        self.speed = speed

    def act(self, obs):
        bag = obs['objects']['backpack']['center']
        ...                                    # your logic: use self.speed, bag, obs['person'], ...
        return I.Command(vx=0.0, vy=0.0, yaw_rate=0.0, arms=None, grippers=(0.0, 0.0))
```

As written it stands still. Do not send a walking command without steering: a constant `vx` walks the robot into the table and then into the person.

Run it:

```bash
python run.py --policy my_policy:MyPolicy                 # module on the path (repository root or current folder)
python run.py --policy path/to/my_policy.py:MyPolicy      # or a file path
```

`MODULE:CLASS` is imported with `importlib`. The repository root, `hri_eval/` and your current folder are on the path, in that order, so a module in the repository root or in `hri_eval/` wins over one of the same name in your current folder. Do not name your module like one in `hri_eval/` (`episode`, `interfaces`, `report`, `catalog`, ...). For a package elsewhere, set `PYTHONPATH`, or pass `path/to/file.py:CLASS`.

## Step 5: parameters

`--params` is JSON, or a path to a `.json` file, passed to your constructor as keyword arguments:

```bash
python run.py --policy examples.walk_to_table:WalkToTable --params '{"standoff": 0.6, "v_max": 0.3}' --out trials/walk_standoff_06
echo '{"speed": 0.3}' > my_policy_params.json
python run.py --policy my_policy:MyPolicy --params my_policy_params.json --out trials/my_policy_speed_03
```

Every key must be a keyword of your constructor. An unknown key makes the constructor raise: that episode becomes an error row in the report, and `run.py` exits with 1.

## Step 6: many episodes

```bash
python run.py --policy my_policy:MyPolicy --episodes 10 --seed 0 --jobs 4
```

Seed 0 is the nominal episode, seeds 1 to 9 move the robot start and the backpack and change the person's playback speed. See [batch-and-variations.md](batch-and-variations.md).

## Optional methods

| Method | Interface | Called | Used for |
|---|---|---|---|
| `reset(ctx)` / `reset(obs)` | both | once, before the first tick | initial state |
| `policy_info()` | joint | at the end | `summary.json` `policy` (return a dict) |
| `describe()` | both | at the end | the trial `README.md` (return a string) |
| `disclosures()` | both | at the end | `summary.json` and `README.md` (return a list of strings) |

For a command level policy, `policy_info` is filled by the adapter (GR00T model, weights hash, your class name).

## Common patterns (joint level)

Read the backpack and the gripper:

```python
bag_pos, bag_R = ctx.body('backpack')
grasp_pos, grasp_R = ctx.site('bag_grasp_top')
tool_pos, tool_R = ctx.site('left_grasp_center')
```

Phases. A changed `phase` writes an event, with `phase_reason` as the reason:

```python
return E.ControlOutput(..., phase='approach', phase_reason='target selected')
```

Your own events:

```python
ctx.event('grasp_closed', 'both pads in contact', fingers_q=ctx.fingers_q)
```

Extra logs (fixed shape every tick):

```python
return E.ControlOutput(..., log={'tool_err': tool_pos - grasp_pos})   # controller-log.npz key x_tool_err
```

End the episode early:

```python
return E.ControlOutput(..., done=True)
```

Kinematics without touching the simulation:

```python
ds = ctx.scratch()                 # separate from the simulation; writing to it changes nothing in the episode
ds.qpos[22:29] = my_left_arm       # try a left arm posture
mujoco.mj_kinematics(ctx.m, ds)
tool = ds.site_xpos[ctx.m.site('left_grasp_center').id].copy()
```

`ctx.scratch()` returns the same `MjData` object on every call, reset to the live state. `ctx.site()` and `ctx.body()` call it too. So read what you need from `ds` before the next `ctx.scratch()`, `ctx.site()` or `ctx.body()` call in the same tick.

## Letting go (handover)

If your policy opens the gripper that holds the bag after the pickup, G8 needs a release event at the tick where you decide to open (from 1 s before to one tick after the first opening command). Name it `release` and list the robot-side signals the decision used:

```python
ctx.event('release', 'arm load dropped', signals=['arm_joint_torque', 'umi_finger_force'])
```

If the decision uses the person's contact forces, mark it privileged and disclose it. Details and the list of robot-side signals: [scoring.md](scoring.md#g8_controlled_handover).

The command level has no way to log events yet: a `phase` change writes a `phase` event, which G8 does not accept as a release decision. So a command level policy that opens the holding gripper after a pickup fails G8 part e. G8 is not one of the five scores, but it is in `gates.json` and `report.json`. Event logging for command level policies is [On request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Event%20logging%20for%20command%20level%20policies&feature=Event%20logging%20for%20command%20level%20policies).

## Checklist before you submit

- [ ] The policy never writes to `ctx.d` or `ctx.m` ([rules.md](rules.md)).
- [ ] It reads the person only at `ctx.t`.
- [ ] `disclosures()` lists the privileged information it uses.
- [ ] You ran `--episodes 10 --seed 0` with the default `--max-time 24`, from a clean checkout (the `report.md` header says local changes `none`).
- [ ] `report.md` is ready to paste into the [submit form](https://github.com/affordance-org/hri-eval/issues/new?template=submit_result.yml&title=%5BResult%5D%20).

## Interfaces we do not have yet

End-effector pose actions, event logging for command level policies, rendered camera images, a LeRobot adapter, a ROS 2 bridge and a remote policy server are on request: [catalog.md](catalog.md#policy-interfaces).
