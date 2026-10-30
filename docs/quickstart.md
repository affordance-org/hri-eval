# Quick start

From zero to a scored episode.

## 1. Install

```bash
git clone https://github.com/affordance-org/hri-eval && cd hri-eval
pip install -r requirements.txt
```

| Requirement | |
|---|---|
| Python | 3.12 or newer, with pip (we test with 3.12.3). The pinned NumPy 2.5.4 and SciPy 1.18.1 in `requirements.txt` do not install on older Pythons |
| GPU | not needed: everything runs on the CPU |
| Display | not needed: nothing is rendered |
| OS | we run it on Linux |
| Network | only on the first run, to download the walking policy weights |

A virtual environment is a good idea:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins MuJoCo 3.14.0, NumPy, SciPy, onnxruntime and OSQP.

## 2. Run the baseline

```bash
python run.py
```

On the first run `hri_eval/weights.py` downloads the two NVIDIA GR00T-WholeBodyControl ONNX policies (Balance and Walk) from the NVlabs GitHub repository into `hri_eval/policy/` and checks their sha256. They are under the NVIDIA Open Model License (`licenses/NVIDIA_Open_Model_License.txt`). Later runs reuse them.

Then it runs one episode of the [point-fetch scenario](scenario-point-fetch.md) with the baseline controller (up to 24 s of simulated time; the baseline ends the episode itself at about 21.7 s), re-simulates it with the independent checker, and prints the scores. The baseline passes no fall, pickup and carry. It fails handover and pace because it never lets go of the bag.

The output lands in `trials/baseline/`. A second run with the same `--out` folder replaces the first; use another `--out` to keep both.

## 3. Run an example policy

```bash
python run.py --policy examples.stand_still:StandStill       # joint level: stands in place
python run.py --policy examples.walk_to_table:WalkToTable    # command level: walks to the table and stands
```

## 4. Read the output

```
trials/WalkToTable/
  report.md        the scores, one row per episode
  report.json      the same, machine readable
  ep000/           the trial: every file the checker needs to re-score it
    gates.json     all ten checker gates with their measurements
    events.json    phase changes and your events
    summary.json   episode facts and measured invariants
    ...
```

`run.py` exits with 0 when every episode ran to a normal end and is valid, whatever the scores, and with 1 otherwise (see [batch-and-variations.md](batch-and-variations.md#exit-code)).

- `report.md` / `report.json`: [batch-and-variations.md](batch-and-variations.md#output)
- the trial folder: [trial-format.md](trial-format.md)
- what each score means: [scoring.md](scoring.md)

## 5. Run many episodes

```bash
python run.py --policy examples.walk_to_table:WalkToTable --episodes 10 --seed 0 --jobs 4
```

Seed 0 is the nominal episode. Seeds 1 to 9 move the robot start, the backpack start and the person's playback speed. See [batch-and-variations.md](batch-and-variations.md).

On our 8 vCPU Linux machine one baseline episode takes about 2 minutes of wall time including the checker, and 10 episodes with `--jobs 5` about 6 minutes. An episode in which the robot touches the person takes longer to check (see the [FAQ](faq.md#why-is-the-checker-slow-on-my-episode)).

## 6. See everything that exists

```bash
python run.py --list
```

Prints the full catalog: scenarios, people and rooms, robots, policy interfaces, observations, scores, variations, tooling and services. Items that are not built yet are marked `on request`, each with a link that opens a pre-filled GitHub issue.

## 7. Write your own policy

[write-a-policy.md](write-a-policy.md). Short version:

```python
# my_policy.py
import interfaces as I

class MyPolicy(I.CommandPolicy):
    def act(self, obs):                     # 50 Hz
        return I.Command(vx=0.0, vy=0.0, yaw_rate=0.0, arms=None, grippers=(0.0, 0.0))   # stands still
```

```bash
python run.py --policy my_policy:MyPolicy
```

## 8. Re-check a trial

```bash
python hri_eval/check_gates.py trials/WalkToTable/ep000
```

Re-simulates the trial from its own files and rewrites `gates.json`. Exit code 0 only if all ten gates pass.

## Something missing?

Run `python run.py --list` or read [catalog.md](catalog.md) for what is on request, and ask for it: [new scenario](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20), [new robot](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20), [new feature](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20).
