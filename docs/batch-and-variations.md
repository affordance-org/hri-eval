# Batches, seeds and variations

One episode tells you little. Run many, each with a different start, and read the pass rate.

```bash
python run.py --policy my_policy:MyPolicy --episodes 10 --seed 0 --jobs 4
```

## Command line

| Option | Default | Meaning |
|---|---|---|
| `--policy MODULE:CLASS` | the baseline | your policy class; `path/to/file.py:CLASS` also works |
| `--params JSON\|FILE` | `{}` (baseline: `hri_eval/baseline_params.json`) | keyword arguments for the policy constructor |
| `--episodes N` | 1 | number of episodes |
| `--seed S` | 0 | first seed; the episodes use seeds S, S+1, ..., S+N-1 |
| `--jobs J` | 1 | episodes run in parallel processes |
| `--out DIR` | `trials/<ClassName>` or `trials/baseline` | output folder; a run replaces the `epNNN` folders and reports an earlier run left there |
| `--max-time SECONDS` | 24 | longest simulated length of each episode (a policy may end it earlier) |
| `--scenario NAME` | `point-fetch` | the only scenario today; any other name prints the request link and exits |
| `--list` | | print the full catalog, with every on-request item and its link |

Episode i (counting from 0) uses seed S+i and writes `DIR/ep<i>` with three digits: `ep000`, `ep001`, ... This holds for a single episode too.

Before it starts, `run.py` removes the `epNNN` trial folders, `report.json` and `report.md` that an earlier run wrote to the same `DIR`, so the folder always holds one run. To keep a run, give the next one another `--out`.

### Exit code

| Code | When |
|---|---|
| 0 | every episode ran to a normal end (`max_time` or `controller_done`) and is valid, whatever the scores |
| 1 | an episode could not run (`error`), ended in an `exception` or diverged (`diverged`), or is not valid; also a bad `--policy` |
| 2 | an unknown `--scenario` |

The exit code says nothing about the scores. For a regression test, read `totals` in `report.json`.

## Seeds

| Seed | Episode |
|---|---|
| 0 | the nominal recorded episode: the scene exactly as in [scenario-point-fetch.md](scenario-point-fetch.md), the person at recorded speed |
| 1 and up | a variation drawn from the seed |

The same seed always gives the same variation, on any machine. `--seed 0 --episodes 10` runs the nominal episode plus seeds 1 to 9. `--seed 100 --episodes 50` runs 50 variations and no nominal episode.

## Variations

For seed != 0, `hri_eval/variations.py` draws six numbers from `numpy.random.default_rng(seed)`, in this order, each uniform:

| Key | Range | What changes |
|---|---|---|
| `robot_dx_m` | -0.10 to 0.10 m | robot start position, world x |
| `robot_dy_m` | -0.10 to 0.10 m | robot start position, world y |
| `robot_dyaw_deg` | -10 to 10 deg | robot start heading, turned about the vertical axis |
| `bag_dx_m` | -0.05 to 0.05 m | backpack start position on the table, world x |
| `bag_dy_m` | -0.05 to 0.05 m | backpack start position on the table, world y |
| `person_speed` | 0.9 to 1.1 | playback speed of the person's recording |

How each is applied:

- **Robot and backpack start.** The offsets are written into keyframe `start` of the episode's `trial.xml` before the model is loaded. The runner and the checker load the same file, so they start from the identical state. The checker also recomputes the variation from the seed and fails G0 if `variation.json` does not match it.
- **Person speed.** The whole clip plays at that factor: the pointing gesture, the arm lowering, and the waiting motion afterwards. At speed 1.1 the pointing happens at t = 0.24 to 1.30 s instead of 0.27 to 1.43 s. The checker replays the recorded person poses, so it sees exactly what the runner simulated.

Each episode folder has `variation.json`. This one is seed 7; the same seed gives the same values for every policy:

```json
{
 "robot_dx_m": 0.025019093320933383,
 "robot_dy_m": 0.07944276019391511,
 "robot_dyaw_deg": 5.513713804903871,
 "bag_dx_m": -0.027479281000940815,
 "bag_dy_m": -0.019983371508877457,
 "person_speed": 1.0747106890792524,
 "seed": 7,
 "nominal": false
}
```

For seed 0 all offsets are 0, `person_speed` is 1.0 and `nominal` is true.

Other variations (the person standing elsewhere, other objects on the table, a different table height, lighting for camera policies) are [on request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20More%20variation%20types&feature=More%20variation%20types).

## Parallel runs

`--jobs J` runs up to J episodes at once, each in its own process (`spawn`). Every episode builds a new policy object and writes only its own folder. GR00T runs on one CPU thread per process, so J up to your number of cores is a reasonable start.

If your policy raises during an episode, the runner keeps everything recorded so far, ends that episode with `end_reason` `exception`, and the checker scores it; no fall fails for that episode. If an episode cannot run at all (for example your constructor raises), its row gets `end_reason` `error` with the traceback in `failure`, and every score fails. Either way the batch continues, and `run.py` exits with 1 at the end.

## Output

```
trials/MyPolicy/
  report.json
  report.md
  ep000/   trial.xml trajectory.npz controller-log.npz telemetry.json events.json summary.json
           scene-context.json recipient-motion.json README.md gates.json variation.json
  ep001/
  ...
```

The trial files are described in [trial-format.md](trial-format.md).

### Terminal

After each episode `run.py` prints one line:

```
ep003 seed 3: valid, no fall PASS pickup FAIL carry FAIL handover FAIL pace FAIL
```

At the end it prints the table from `report.md` and the pass rate per score. The pass rate counts a score only in a valid episode.

### report.json

```
{
  "scenario": "point-fetch",
  "policy": "my_policy:MyPolicy",          # or "baseline"
  "interface": "command",                  # or "joint"
  "params": {...},                         # constructor keywords, or the baseline params file
  "seeds": [0, 1, ..., 9],
  "max_time_s": 24.0,
  "commit": "dbac1a8",                     # short git commit of the repository, null outside a git checkout
  "local_changes": [],                     # modified tracked files under hri_eval/, run.py, requirements.txt
  "created": "2026-11-01T09:00:00+00:00",
  "episodes": [ <episode row>, ... ],
  "totals": {
    "valid":    {"passed": 10, "episodes": 10, "rate": 1.0},
    "no_fall":  {"passed": 9,  "episodes": 10, "rate": 0.9},
    "pickup":   {...}, "carry": {...}, "handover": {...}, "pace": {...},
    "min_robot_person_distance_m": 0.412
  }
}
```

Episode row:

| Key | Type | Meaning |
|---|---|---|
| `episode` | str | folder name, `ep003` |
| `seed` | int | seed |
| `variation` | dict | the six variation values and `nominal` |
| `valid` | bool | the checker's G0 invariants and G1 motor replay pass: the trial is a real, unmodified simulation |
| `scores` | dict | `no_fall`, `pickup`, `carry`, `handover`, `pace`: true or false. Each is its gate; no fall also needs a normal end ([scoring.md](scoring.md)) |
| `t_pickup_s` | float or null | start of the lift that passed the pickup score |
| `t_handover_s` | float or null | `t_handover_complete_s` of a passed handover: from then on the person alone holds the bag |
| `min_robot_person_distance_m` | float or null | smallest distance between any robot collider and any collider of the person over the episode, sampled at 50 Hz; negative means they overlapped |
| `gates` | dict | pass or fail of all ten gates, `G0_invariants` to `G9_smoothness` |
| `end_reason` | str | `max_time`, `controller_done`, `diverged`, `exception`, or `error` (the episode could not run) |
| `failure` | str or null | why the episode stopped early |
| `sim_time_s` | float | simulated time |
| `wall_time_s` | float | wall-clock time of the episode, including the checker and the distance measurement |

`totals` has `passed`, `episodes` and `rate` for `valid` and each score, and the smallest `min_robot_person_distance_m` of the batch. A score counts as passed only in a valid episode: an episode that is not valid counts as failed for every score, even if its own row says `true`. `episodes` is always the number of episodes run.

### report.md

The same content as a page you can paste into an issue: a header line (scenario, policy, interface, episodes, seeds, max time, commit, local changes, MuJoCo version), one row per episode, and a totals row.

| episode | seed | robot dx dy [m] yaw [deg] | bag dx dy [m] | person speed | valid | no fall | pickup | carry | handover | pace | pickup [s] | handover [s] | min dist [m] | end | sim [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ep000 | 0 | nominal | nominal | 1.000 | ... | ... | ... | ... | ... | ... | ... | ... | ... | max_time | 24.00 |

`end` is the `end_reason` and `sim [s]` the simulated time, so an episode your policy ended early is visible.

This is the file you paste into the [submit form](https://github.com/affordance-org/hri-eval/issues/new?template=submit_result.yml&title=%5BResult%5D%20) for the [leaderboard](leaderboard.md).

## Reading a batch

- Start with `valid`. An episode that is not valid means the simulation was changed (see `gates.json` G0); the totals count it as failed for every score.
- Check `end` and `sim [s]`. An `exception` row is a bug in your policy; the traceback is in `failure` and in the trial's `events.json`.
- Compare the nominal episode with the variations. A policy that passes seed 0 and fails most other seeds is fitted to the exact start.
- Open the failing episode's `gates.json` for the reason: for example `G2_no_fall.contact_violations_worst` names the contact and its time.
