# HRI Eval

https://github.com/user-attachments/assets/9f9f6c20-cfe0-4b2f-93f4-ab1914502f2b

Roll out your robot policy in our room, with real human motion.

```bash
git clone https://github.com/affordance-org/hri-eval && cd hri-eval
pip install -r requirements.txt
python run.py                                # our baseline
python run.py --policy my_module:MyPolicy    # yours
```

```python
import episode as E

class MyPolicy:
    def reset(self, ctx): ...
    def policy_step(self, ctx):  # 50 Hz: robot state in, 29 joint targets and 4 gripper fingers out
        return E.ControlOutput(q_target=q, kp=kp, kd=kd, fingers=f)
```

Example: [`examples/stand_still.py`](examples/stand_still.py)

| Score | Passes when the robot |
|---|---|
| no fall | stays up and never touches the person |
| pickup | lifts the bag with its grippers |
| carry | carries it 0.4 m |
| handover | puts it in the person's hand |
| pace | does it within 3x the human's time |

[affordance.homes](https://affordance.homes) · code Apache-2.0 · data CC BY 4.0 ([licenses](licenses))
