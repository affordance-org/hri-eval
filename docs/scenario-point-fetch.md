# Scenario: point-fetch

The only scenario today. Run it with `--scenario point-fetch` (the default).

A person stands in a room and points at a backpack on a low table. The robot has to pick the backpack up, carry it to the person and hand it over, without falling and without bumping into the person or the furniture.

| | |
|---|---|
| Scenario name | `point-fetch` |
| Scene file | `hri_eval/scene/room_point_fetch.xml` |
| Person recording | `hri_eval/person/point_fetch_01.json` (21 bones, 30 fps) |
| Robot | Unitree G1, 29 joints, two UMI parallel grippers |
| Policy rate | 50 Hz |
| Physics | MuJoCo 3.14.0, dt 0.002 s (500 Hz), `implicitfast`, elliptic cones, impratio 10 |
| Episode length | up to 24 s of simulated time (`--max-time 24`); a policy may end it earlier |
| Human reference time | 4.4 s |
| Pace limit | 3 x 4.4 s = 13.2 s |

Other scenarios are on request: see [catalog.md](catalog.md#scenarios).

## The room

All positions are in the room frame: x and y on the floor plan, z up, metres. The origin is on the floor at a corner of the kitchen cabinet. Below, east means +x and north means +y. See [observations.md](observations.md#frames-and-units).

| Thing | Where | In the physics model |
|---|---|---|
| Floor | z = 0 | plane `floor` |
| Low table | centre (1.949, 5.165), top 0.90 m x 0.75 m, top surface at z = 0.330 m | boxes `table_top`, `table_leg_0..3`, `table_apron_*` |
| Backpack | upright on the table at (1.663, 5.399) | free body `backpack`, 1.2 kg |
| Water bottle | on the table at (1.561, 5.074) | fixed mesh `object_water_bottle` |
| Cup | on the table at (1.883, 5.113) | fixed mesh `object_cup` |
| Candy box | on the table at (2.017, 5.414) | fixed mesh `object_candy_box` |
| Bed | north of the table, y from about 5.9 to 7.0 | 10 meshes `room_bed_0..9` |
| Tall shelf | west wall, x 0 to 0.2, y 4.96 to 7.10 | box `room_shelf_tall` |
| Shelf | east wall, x 2.49 to 2.77, y 5.77 to 7.02 | box `room_shelf_right` |
| Fridge | x -0.67 to 0.41, y 4.04 to 4.95 | box `room_fridge` |
| Kitchen cabinets | along the y = 0 wall | boxes `room_kitchen_lower`, `room_kitchen_upper` |
| Bins, low shelf | south half of the room | boxes `room_bins`, `room_shelf_low` |

The bottle, the cup and the candy box are fixed to the world. They do not move, and a robot that touches them breaks the no fall score (a robot to world contact other than feet on the floor). The backpack is the only object that moves.

## The robot

| | |
|---|---|
| Model | Unitree G1 29-DoF rev 1.0 from MuJoCo Menagerie |
| Hands | one UMI parallel gripper on each wrist (instead of the stock hands) |
| Joints you command | 29: 12 leg, 3 waist, 14 arm (list in [observations.md](observations.md#joint-order)) |
| Motors | plain torque motors, clipped to the official URDF effort limits ([actions.md](actions.md#torque-limits)) |
| Grippers | 4 finger slides, force-limited position servos, 35 N per finger |
| Start pose | pelvis at (0.750, 5.950), height 0.784 m, heading -55.0 deg |
| Start posture | legs: hip pitch -0.1, knee 0.3, ankle pitch -0.2 rad; arms: shoulder pitch 0.3, shoulder roll +-0.45, elbow 0.97, wrist roll +-0.15 rad; waist 0; grippers wide open (-0.035 m) |

The start heading points between the table and the person. The start pose is a choice made for this scenario. It is not the position of anyone in the recording.

At the start the robot stands about 1.07 m from the backpack and about 2.4 m from the person.

### Gripper

Each gripper has two finger slides. A finger target is in metres along the slide:

| Finger target | Meaning |
|---|---|
| -0.035 m | wide open, 153 mm between the pads (the start value) |
| 0.0 m | the open width of the stock UMI model, 83 mm |
| 0.05 m | the closed end of the range |

The servos are position servos with kp 5000 N/m, kv 30 and a 35 N force limit, so a closed target on an object squeezes it with up to 35 N per finger. The wide stroke and the 35 N limit are assumptions, not manufacturer data.

## The backpack

| | |
|---|---|
| Body | `backpack`, one rigid body with a free joint (`backpack_free`) |
| Geometry | 29 convex collision hulls of the scanned parts of the bag (plus 29 visual meshes that do not collide) |
| Mass | 1.2 kg (assumed); weight W = 11.772 N |
| Start | upright on the table, body origin at (1.663, 5.399, 0.330), yaw 0 |
| Handle | none: the scanned carry handle is not in the model and no strap was added. Grasp the bag body. |

Sites on the backpack (read them with `ctx.site(name)`):

| Site | Position in the bag frame [m] | Use |
|---|---|---|
| `bag_com` | (0.000, 0.005, 0.198) | centre of mass |
| `bag_top_center` | (0, 0, 0.436) | top of the bag |
| `bag_grasp_top` | (0, 0.017, 0.386) | a top-down grasp frame across the bag's thickness |
| `bag_grasp_top_flip` | (0, 0.017, 0.386) | the same grasp, closing axis flipped (the baseline uses it) |

For the grasp sites the axes are: x closing direction, y pad width, z approach direction (the UMI base axes).

In the recording the backpack lies on its back at the same spot. In the eval it stands upright, because no direct grasp of the lying bag fits the gripper. This is a disclosed change from the recording.

## The person

The person was recorded with cameras in this room. The eval replays that recording: the person's bones follow the recorded motion, and their hands, forearms, pelvis, torso and legs are rigid collision shapes fitted to the visible skin and moved with the bones every physics step. The person is kinematic (infinite mass): the robot cannot push them.

| | |
|---|---|
| Recording | `hri_eval/person/point_fetch_01.json`, 304 frames at 30 fps |
| Skeleton | 21 bones (names in [observations.md](observations.md#person-bones)) |
| Colliders | 62 bodies named `recipient_*`: both hands (the left hand as finger segments), both forearms, pelvis, three spine pieces, thighs, calves, feet |
| Where | pelvis at about (1.40, 3.58) at t = 0, then about (1.48, 3.69); head at about 1.56 m height |
| Reacts to the robot | no. Reactive people are [on request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Reactive%20humans%20that%20respond%20to%20the%20robot&feature=Reactive%20humans%20that%20respond%20to%20the%20robot). |

### Timeline

Simulation time t = 0 is frame 120 of the recording. At the nominal playback speed (seed 0):

| t [s] | Recording frame | The person |
|---|---|---|
| 0.00 | 120 | stands, facing the room |
| 0.27 to 1.43 | 128 to 163 | points at the backpack with the right arm |
| 1.43 to 2.10 | 163 to 183 | lowers the arm |
| 2.10 to the end | 183 to 219, back and forth | waits: the quietest stretch of the recording plays forward and backward, at most 0.45 x real time, so the person keeps moving slightly instead of freezing |

With a variation seed the person's whole motion, the waiting included, plays at a speed factor between 0.9 and 1.1 ([batch-and-variations.md](batch-and-variations.md#variations)), so these times shift. The robot and backpack start positions vary too.

The person never reaches for the bag in this version. Their hands stay where the recording has them while waiting, at about hip height. So a handover only passes if the robot leaves the bag resting on the person's hand or forearm with the grippers clear. A person who reaches out and takes the bag is the reactive-human feature, [on request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Reactive%20humans%20that%20respond%20to%20the%20robot&feature=Reactive%20humans%20that%20respond%20to%20the%20robot).

Nothing scores whether the robot waits for the pointing gesture. A policy may react to it (the baseline does) or ignore it.

## What a successful episode looks like

The five scores that `run.py` prints, in the order they usually happen. Full definitions in [scoring.md](scoring.md).

| Score | Passes when |
|---|---|
| no fall | the robot stays up for the whole episode, touches the floor only with its feet, the bag only with its grippers, and the person's hands or forearms only lightly with its grippers; the episode does not end in an exception |
| pickup | the bag is 5 cm or more above its start height, held only by the finger pads, for 0.5 s |
| carry | while held only by the pads, the bag moves 0.4 m and the pelvis moves 0.3 m horizontally |
| handover | at the end of the episode, for 1.0 s or more, the person alone holds the bag (hands or forearms carry 80% of its weight), the robot does not touch it, every gripper part is 5 cm or more from it, and the robot stands still |
| pace | the handover is complete within 13.2 s; the clock starts when the robot starts moving, but no later than t = 1.0 s |

## The human reference

In the recording a second person fetched the backpack and handed it over. From frame 120 (the same moment as t = 0) to frame 252 (handover complete) took 132 frames, 4.4 s. The pace score allows the robot three times that: 13.2 s.

## What the baseline does

`python run.py` without `--policy` runs `hri_eval/fetch_present.py` (`PointFetch`, `PointFetchRunner`) with `hri_eval/baseline_params.json`.

| Phase | What happens |
|---|---|
| `watch` | balances in place and watches the person; detects the pointing gesture from the person's arm keypoints and picks the object the ray points at |
| `approach` | plans a path and walks (GR00T WBC) to a stance west of the table, facing the backpack |
| `pre_grasp`, `align`, `descend`, `close` | brings the left gripper down over the top of the bag and closes it on the bag body |
| `lift` | lifts the bag off the table |
| `hold`, `turn`, `carry` | steps back, turns and walks to the person with the arm frozen |
| `present`, `present_hold` | holds the bag out in front of the person's chest and keeps holding it |

The baseline never lets go, so handover and pace fail by design. Its result is in [leaderboard.md](leaderboard.md).

The baseline uses privileged information, and says so in each trial's `README.md`: the person's keypoints from the replay (used only while they would be visible from an assumed head camera), the true backpack pose, and the true pelvis pose for walking.

It also reads the person's recorded pose at frame 200 of the recording (t = 2.67 s, in the waiting stretch) before that time: at the start (the left shoulder and hand, which it logs) and when it plans where to present the bag (the chest, which it walks up to and faces). This breaks the current-time rule ([rules.md](rules.md#3-information-your-policy-may-use)). The baseline is exempt, and the [leaderboard](leaderboard.md) marks it.

To change baseline parameters, copy `hri_eval/baseline_params.json`, edit the copy and pass it:

```bash
cp hri_eval/baseline_params.json baseline_edit.json
# edit baseline_edit.json, e.g. "t_present_hold": 3.0
python run.py --params baseline_edit.json --out trials/baseline_edit
```

Pass the full file, not a single key: without `--policy`, `--params` replaces `baseline_params.json` and the rest falls back to the defaults in `fetch_present.py`.
