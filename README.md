# HRI Eval

Evaluate a robot policy next to a real, recorded person, in simulation.

The room is a MuJoCo copy of a real room. The person is motion recorded in that room with cameras only. A Unitree G1 with two UMI grippers stands in the room. The person points at a backpack on the table, and the robot has to bring it to them.

By [Affordance](https://affordance.homes).

## Install

```bash
git clone https://github.com/affordance-org/hri-eval && cd hri-eval
pip install -r requirements.txt
python scripts/get_weights.py      # NVIDIA GR00T-WholeBodyControl walking policy (used by the baseline and the example)
```

Python 3.12. Runs on CPU.

## Run

```bash
python run.py trials/baseline --params baseline/params.json          # our baseline
python run.py trials/mine --policy examples.stand_still:StandStill   # the smallest example
python run.py trials/mine --policy my_package.my_module:MyPolicy     # yours
```

Each run writes a trial folder (trajectory, contacts, events, telemetry) and prints the scores.

## Your policy

```python
import episode as E

class MyPolicy:
    def reset(self, ctx): ...                       # once, at t = 0
    def policy_step(self, ctx) -> E.ControlOutput:  # 50 Hz
        ...
        return E.ControlOutput(q_target=q, kp=kp, kd=kd, fingers=f)
```

`ctx` gives the robot state (`q29`, `dq29`, `base_pos`, `base_quat`, ...), the time, and read access to the MuJoCo model and data. `ControlOutput` sets 29 joint position targets with PD gains and the 4 gripper finger targets. The runner applies the torques, clipped to the G1 URDF limits. Writing to the simulation is detected and reported in the trial.

See `examples/stand_still.py` and `hri_eval/fetch_present.py` (the baseline).

## Scores

An independent checker re-simulates the trial and reports:

| Gate | Passes when |
|---|---|
| G2 no fall | the robot stays up and never touches the person |
| G3 pickup | the bag is lifted at least 5 cm, held by the grippers only |
| G4 carry | the bag is carried at least 0.4 m |
| G5 handover | the person ends up holding the bag |
| G6 pace | the handover is done within 3x the time the human took (4.4 s) |

The baseline passes G2, G3 and G4. It presents the bag without handing it over, so G5 and G6 fail.

## Recorded and simulated

* The person is a replayed recording. They do not react to the robot: their pointing gesture plays, then a quiet stretch of their recording loops.
* The backpack is a scanned mesh with an assumed mass of 1.2 kg. The grippers are assumed to have 35 N per finger.
* The baseline reads the person's skeleton and the object poses from the simulator, limited to what an assumed head camera can see. No images are rendered.

## License

Code: Apache-2.0 (`LICENSE`). Room, person and object data: CC BY 4.0 (`DATA_LICENSE.md`). Third-party parts: `THIRD_PARTY_NOTICES.md`.
