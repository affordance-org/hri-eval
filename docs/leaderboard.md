# Leaderboard

Policies evaluated in the `point-fetch` scenario. Each cell is episodes passed / episodes run, from the `report.md` of the run.

| Policy | Interface | Information used | Seeds | No fall | Pickup | Carry | Handover | Pace |
|---|---|---|---|---|---|---|---|---|
| Baseline ([`fetch_present.py`](../hri_eval/fetch_present.py)) | joint | proprioception, localization, person keypoints, true backpack pose; also reads the person's recorded pose at frame 200 before that time (exempt from the current-time rule, see [rules.md](rules.md#3-information-your-policy-may-use)) | 0 to 9 | 10/10 | 10/10 | 10/10 | 0/10 | 0/10 |

All 10 baseline episodes are valid. Pickup starts at 10.3 to 11.0 s, and the closest the robot comes to the person is 0.21 m. The baseline ends each episode itself at about 21.4 to 22.1 s.

The baseline picks the bag up and carries it to the person, then holds it out and never lets go, so handover and pace fail by design. Reproduce the row with `python run.py --episodes 10 --seed 0 --jobs 4`.

Rows are sorted by handover, then pace, carry, pickup and no fall. Only valid episodes count (the checker's G0 invariants and G1 motor replay pass); `report.md` already counts an episode that is not valid as failed.

## Submit your result

1. Run 10 episodes with the defaults:

   ```bash
   python run.py --policy my_policy:MyPolicy --episodes 10 --seed 0
   ```

   `--jobs` is fine. Keep `--max-time 24` and `--scenario point-fetch`.

2. Open the [submit form](https://github.com/affordance-org/hri-eval/issues/new?template=submit_result.yml&title=%5BResult%5D%20) and paste `trials/MyPolicy/report.md`. Tick the information your policy uses ([rules.md](rules.md#3-information-your-policy-may-use)) and give the commit of your clone (`git rev-parse HEAD`).

3. We check the report, may ask for the trial folders or the code, and add your row here.

What we check before adding a row:

| Check | How |
|---|---|
| all 10 seeds 0 to 9 reported | `report.md` |
| every episode valid | the `valid` column |
| an unmodified repository at a public commit | the commit and the local changes in the `report.md` header (`none`) |
| no writes to the simulation, person read at the current time only, privileged information disclosed | [rules.md](rules.md); we may ask for the trial folders |

A code link is optional. Entries with code get a link in the Policy column.

## Hosted evaluation

If you cannot run the eval yourself (a large checkpoint, a GPU stack), [request a hosted evaluation](https://github.com/affordance-org/hri-eval/issues/new?template=hosted_eval.yml&title=%5BHosted%20eval%5D%20). It is not a running service yet: tell us what you need and we reply in the issue.

A public leaderboard service with automatic submission is [on request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Public%20leaderboard%20service&feature=Public%20leaderboard%20service). Today the table above is updated by hand from submitted issues.
