# HRI Eval documentation

Test a robot policy next to a real, recorded person in a real room rebuilt in simulation, and get scored by an independent checker.

## Start here

| Page | What it covers |
|---|---|
| [quickstart.md](quickstart.md) | install, run the baseline and the examples, read the output |
| [write-a-policy.md](write-a-policy.md) | both policy interfaces, step by step, with complete examples |
| [catalog.md](catalog.md) | everything you can test today, and everything on request |

## Reference

| Page | What it covers |
|---|---|
| [scenario-point-fetch.md](scenario-point-fetch.md) | the room, the robot, the backpack, the person's timeline, the human reference, the baseline |
| [observations.md](observations.md) | every `ctx` field and `obs` key, units, frames, joint and finger order, person bones and keypoints |
| [actions.md](actions.md) | `ControlOutput`, the PD law, torque limits, the GR00T helper, `Command` |
| [scoring.md](scoring.md) | the five scores and all ten checker gates G0 to G9 with their thresholds |
| [batch-and-variations.md](batch-and-variations.md) | episodes, seeds, variations, parallel jobs, `report.json` and `report.md` |
| [trial-format.md](trial-format.md) | every file in a trial folder |
| [rules.md](rules.md) | fair play: read-only simulation, allowed information, time |

## Results and help

| Page | What it covers |
|---|---|
| [leaderboard.md](leaderboard.md) | results so far and how to submit yours |
| [faq.md](faq.md) | short answers |
| [how-it-was-made.md](how-it-was-made.md) | how the person and the room were rebuilt in simulation, and the modelling choices |

## Ask for what is missing

| Form | Use it for |
|---|---|
| [Request a scenario](https://github.com/affordance-org/hri-eval/issues/new?template=request_scenario.yml&title=%5BScenario%5D%20) | a new task with people |
| [Request a robot](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20) | another robot, hand, or your own URDF or MJCF |
| [Request a feature](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20) | interfaces, observations, scores, tooling, people and rooms |
| [Request a hosted evaluation](https://github.com/affordance-org/hri-eval/issues/new?template=hosted_eval.yml&title=%5BHosted%20eval%5D%20) | we run your checkpoint |
| [Submit a result](https://github.com/affordance-org/hri-eval/issues/new?template=submit_result.yml&title=%5BResult%5D%20) | your `report.md` for the leaderboard |
| [Report a bug](https://github.com/affordance-org/hri-eval/issues/new?template=bug.yml&title=%5BBug%5D%20) | something does not work as written |
