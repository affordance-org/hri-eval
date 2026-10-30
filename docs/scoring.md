# Scoring

Every episode is scored by `hri_eval/check_gates.py`, an independent checker. It does not trust the policy or the runner's own telemetry. It loads the trial's `trial.xml`, starts from row 0 of `trajectory.npz` and re-simulates the whole episode using only the recorded motor commands and the recorded person poses. Contacts, forces, distances and poses all come from that re-simulation.

The checker computes ten gates, `G0_invariants` to `G9_smoothness`, and writes them to `gates.json` in the trial folder.

## The five scores run.py prints

| Score | Gate | One line |
|---|---|---|
| no fall | `G2_no_fall` | stays up, touches only what it is allowed to touch, and the episode ends normally |
| pickup | `G3_pickup` | lifts the bag 5 cm, held only by the finger pads, for 0.5 s |
| carry | `G4_carry` | carries it 0.4 m while walking 0.3 m |
| handover | `G5_handover` | the person alone holds the bag for the last 1.0 s or more |
| pace | `G6_pace` | handover complete within 3 x the human's 4.4 s |

These five are the scores in the terminal table, in `report.json` and `report.md`, and on the [leaderboard](leaderboard.md). Each score is its gate, with one addition: no fall also needs the episode to end normally (`end_reason` `max_time` or `controller_done`). An episode that ends in an exception or a divergence fails no fall even if G2 passed up to that point.

The other five gates (G0, G1, G7, G8, G9) are integrity and quality checks. They are in each episode's `gates.json`. An episode is **valid** when G0 and G1 pass. The totals in `report.json`, `report.md` and the terminal count a score only in a valid episode.

All gates judge the simulated time the episode ran. If your policy ends the episode early with `done=True`, the gates judge it up to that point; `report.md` shows every episode's end reason and simulated time, so a short episode is visible.

The baseline passes no fall, pickup and carry, and fails handover and pace (it never lets go of the bag).

## Common definitions

| Term | Definition |
|---|---|
| W | bag weight, 1.2 kg x 9.81 m/s^2 = 11.772 N |
| contact carries force | normal force above 0.01 N |
| sample | G2 to G6 analyse every 5th physics step (every 10 ms, `--stride 5`). G1, G7 parts b and g, G8 and G9 part b use every physics step. G7 parts a, c and f use the 30 fps frames of `recipient-motion.json`. G9 parts a and c use the 50 Hz policy ticks |
| finger pads | the robot bodies `left_umi_*finger*` and `right_umi_*finger*` |
| UMI base | the other bodies of the grippers, `left_umi_*` and `right_umi_*` |
| person hand and arm colliders | `recipient_hand_l`, `recipient_hand_l_*` (finger segments), `recipient_hand_r`, `recipient_lowerarm_l`, `recipient_lowerarm_r` |
| person body colliders | `recipient_pelvis`, `recipient_spine_01..03`, `recipient_thigh_*`, `recipient_calf_*`, `recipient_foot_*` |
| bag support | the force each contact puts on the bag, grouped by what the bag touches |
| torso tilt | angle between the `torso_link` z axis and the world z axis |

## G0_invariants

Checks that the episode is a real, unmodified, closed-loop simulation. All of these must hold:

| Check | Requirement |
|---|---|
| equality constraints | `neq == 0` |
| mocap bodies | only `recipient_*` (the person), each a direct child of the world with no children |
| gravity compensation in the model | none (`body_gravcomp` and `jnt_actgravcomp` all zero) |
| physics options | dt 0.002, `implicitfast`, elliptic cone, impratio 10, noslip 0, gravity (0, 0, -9.81) |
| motors | each of the 29 joints has exactly one plain torque motor (gain 1, no bias, gear 1, no dynamics) with ctrlrange = forcerange = its URDF limit; each finger has exactly one force-limited servo with a limit of 35 N or less |
| physical parameters | robot and bag bodies, geoms, joints, sites, finger actuators, the table top and the start keyframe identical to the reference scene `hri_eval/scene/room_point_fetch.xml` in your checkout; for a variation seed, the expected start keyframe is the reference one with that seed's variation applied |
| variation | if the trial has `variation.json`, its values equal the variation recomputed from its seed by `hri_eval/variations.py` |
| recorded torques | recorded motor `ctrl` and `qfrc_actuator` within the limits at every step |
| time | starts at 0 and advances by exactly dt every row (one continuous episode); all values finite |
| initial state | row 0 equals keyframe `start` with zero velocity |
| runner instrumentation (`summary.json`) | `controller_execution` is `closed_loop_full_episode`, `diagnostic_only` is false, 0 state writes after init, 0 `xfrc_applied` and `qfrc_applied`, the model never changed during the run |
| model fingerprint | the sha256 over masses, inertias, limits, actuators, contact parameters and options of the model compiled from `trial.xml` equals the one the runner recorded |

A policy fails G0 if it writes to the simulation: see [rules.md](rules.md).

G0 compares the trial with the reference scene in your own checkout, which is also the scene `run.py` simulates. So if you edit `hri_eval/scene/room_point_fetch.xml` itself, G0 still passes: valid means the trial matches the scene in your checkout, not that the checkout is unmodified. `report.json` and `report.md` record the git commit and the modified tracked files under `hri_eval/`, `run.py` and `requirements.txt` for that reason.

## G1_motor_replay

Re-simulating `trial.xml` from row 0 with only the recorded `ctrl`, `mocap_pos` and `mocap_quat` reproduces the recorded `qpos` to within 1e-6 at every physics step.

Reported: `max_qpos_error`, `max_qvel_error`, `first_step_qpos_error_ge_1e-6`, `physics_steps`.

## G2_no_fall (score: no fall)

At every sample of the whole episode:

| Check | Requirement |
|---|---|
| pelvis height | z >= 0.55 m. Within 1.0 s after a commanded base height below 0.65 m (`cmd['base_height']` in `ControlOutput`), z >= 0.30 m |
| torso tilt | < 25 deg |
| robot and world | the only robot to world contacts are the feet (`left_ankle_roll_link`, `right_ankle_roll_link`) on the floor. A gripper may touch the table with less than 5 N. Anything else (a knee on the floor, a hand on the bed, an arm on the cup) is a violation |
| robot and bag | only through the grippers; the UMI base (not the finger pads) with less than 5 N |
| robot and person | only a gripper on the person's hand or forearm colliders, with less than 5 N. Any contact with the person's body colliders, or by any other robot link, is a violation |

Robot self contacts are not violations.

Reported: `min_pelvis_z`, `pelvis_floor_violations`, `first_low_pelvis_t`, `max_torso_tilt_deg`, `max_pelvis_tilt_deg`, `contact_violation_samples`, `first_contact_violation_t`, `contact_violations_worst` (per contact type: largest normal force and its time).

If your policy crouches, put the commanded height in `cmd={'base_height': ...}` so the lower pelvis floor applies. `episode.Gr00tLocomotion` fills `cmd` for you.

## G3_pickup (score: pickup)

The bag is at least 5 cm above its start height **and** the only things touching it are the finger pads (no table, no UMI base, no person), continuously for at least 0.5 s.

Reported: `bag_rest_z`, `max_bag_lift_m`, `longest_lifted_robot_pads_only_s`, `interval` (start and end time of the longest such stretch), `robot_pads_only_samples`.

The pickup time used by G5 is the start of that interval plus 0.5 s.

## G4_carry (score: carry)

Within one continuous stretch where only the finger pads touch the bag:

- the bag's largest horizontal distance from where it was at the start of the stretch is at least 0.4 m, and
- the pelvis's largest horizontal distance from where it was at the start of the stretch is at least 0.3 m.

Lifting the bag and swinging it with the arm while standing does not count: the robot has to walk.

Reported: `interval`, `bag_horizontal_travel_m`, `pelvis_horizontal_travel_m`, `max_lift_m` (of the first stretch that passes, or of the best one).

## G5_handover (score: handover)

The handover is judged on the end of the episode. Find the final stretch, ending at the last sample, where at every sample:

- no robot part touches the bag (total robot to bag normal force at most 0.01 N),
- every gripper collider is at least 5 cm from every bag collision geom, and
- the bag touches only the person, and at least one of the contacts is a hand or arm collider.

Its start is `t_handover_complete_s` (t_hc). G5 passes when:

| Check | Requirement |
|---|---|
| duration | the stretch lasts at least 1.0 s (t_hc is at least 1.0 s before the episode end) |
| support | the mean upward force from the person's hand and arm colliders is at least 0.8 W (9.42 N), both over the whole stretch and over the last 1.0 s |
| pickup first | a G3 pickup completed before t_hc |
| robot at rest | over the last 1.0 s: pelvis z >= 0.55 m, torso tilt < 25 deg, pelvis horizontal speed < 0.5 m/s, no G2 contact violations |

Support from the person's torso or legs (the bag resting against their body) does not count toward the 0.8 W.

Reported, always: `t_handover_complete_s`, `episode_end_s`, `final_recipient_only_duration_s`, `final_state_bag_support_classes`, `final_umi_handle_distance_m`, `robot_standing_stable_last_1s`, `max_pelvis_speed_last_1s`, `umi_clearance_reference` (which bag geoms the 5 cm clearance is measured to). Only when a final stretch with the person alone holding the bag exists: `mean_vertical_support_by_recipient_pads_N`, `mean_vertical_support_last_1s_N`, `pickup_before_handover`.

In this version the person does not reach for the bag ([scenario-point-fetch.md](scenario-point-fetch.md#the-person)), so passing G5 means leaving the bag resting on the person's hand or forearm.

## G6_pace (score: pace)

```
robot_duration = t_hc - min(t_motion_start, 1.0 s)
pass if robot_duration <= 3.0 x 4.4 s = 13.2 s
```

- `t_motion_start` is the first sample where either foot's floor force drops below 1 N (the first step) or the pelvis is more than 10 cm from its start position. A sway of the balance policy is not the start of the motion.
- Up to 1.0 s of standing still at the start is free. After 1.0 s the clock runs whether or not the robot moves.
- 4.4 s is the time a second person in the recording took to fetch the bag and hand it over (frame 120 to frame 252). The person in the eval is the one who points; they never fetch anything.
- No handover (no t_hc) means G6 fails.

Reported: `t_motion_start_s`, `pre_roll_s`, `t_handover_complete_s`, `human_duration_s` (4.4), `limit_s` (13.2), `robot_duration_s`, `ratio_to_human`.

## G7_visual_honesty

Checks that what you would see (the person's skinned avatar in `recipient-motion.json`) matches what the physics simulated, so no invisible shape helps the robot. Seven parts, all must pass:

| Part | Requirement |
|---|---|
| a, penetration | at every 30 fps frame, no robot surface point (collision and visual geoms, sampled every 2 mm) is more than 2 mm inside the person's visible skin; no bag point more than 3 mm inside |
| b, contacts on the skin | every bag to person contact point lies within 3 mm of the visible skin; at every robot to person contact, no robot point is more than 2 mm inside the skin |
| c, colliders follow the bones | each person collider pose equals its bone pose in `recipient-motion.json` within 0.5 mm and 0.5 deg |
| d, hand motion when adapted | if the person's arm is adapted: hand skin speed <= 1.0 m/s, hand rotation <= 300 deg/s, no frame to frame jump > 12 deg. The person is not adapted in point-fetch, so nothing is gated here |
| e, anatomy when adapted | wrist flexion within +-70 deg, radial deviation <= 20, ulnar <= 30, elbow 0 to 145 deg on adapted frames. Not adapted in point-fetch |
| f, no freeze | the person's skin never stands completely still (less than 0.3 mm between frames) for more than 1.0 s |
| g, slow while holding | while the person carries 0.25 W or more, every contact point on them moves at most 2 cm/s, and when it moves faster than 2 mm/s, within 20 deg of vertical |

G7 parts a and b depend on your policy: keep the robot out of the person's skin. Part g also depends on your policy: if you leave the bag on the person's moving hand, the contact points under the bag must move slowly. The other parts mostly check the scenario itself.

Parts a and b are the expensive ones when the robot touches the person: each frame or contact step near the skin costs a skin surface build and thousands of sampled points. So the checker stops once the result is known: part a stops separately for the robot and the bag at the first frame that fails (`a_stopped_after_failure_at_s`, `null` = every frame checked); if part a failed, part b is not checked at all (`parts_pass` shows `null` for it, and `b_not_checked` says why); otherwise part b stops at its first failing contact step (`b_stopped_after_failure_at_s`). The G7 pass or fail result is the same as a full check; the reported maxima and lists are the ones up to the stop.

## G8_controlled_handover

Parts a, b, c and e apply from the G3 pickup on; without a pickup they pass as not applicable. Part d applies to the whole episode.

| Part | Requirement |
|---|---|
| a, contact continuity | the bag is never without any contact for more than 10 ms (no throwing, no dropping) |
| b, peak force | the net contact force on the bag, in total and from each kind of contact, never exceeds 3 W (35.3 N). Squeeze forces of a pinch cancel; impacts and drops do not |
| c, pinch until taken | the holding gripper (the one with more pad force at pickup) keeps both pads at 5 N or more until the person carries at least 0.6 W (7.06 N) |
| d, privileged forces logged | every event in `events.json` that uses the person's contact forces carries `privileged: true`, and `summary.json` disclosures say so |
| e, release decision logged | if the holding gripper opens (finger target down by more than 2 mm) after the pickup, a release event is logged from 1 s before to one tick after the first opening command |

For part e the event name must be `release`, `open`, `let_go` or `release_decision`, or the event carries `decision='release'`. It must either list only robot-side signals in `signals`, or carry `privileged=True` and be disclosed. Robot-side signals:

```
umi_finger_position  umi_finger_force  finger_actuator_force  joint_position  joint_velocity
joint_torque  arm_joint_torque  wrist_force_torque  tactile  ego_camera  time
robot_proprioception  qfrc_actuator  base_imu  controller_state
```

Example, at the tick where the policy decides to open:

```python
ctx.event('release', 'wrist load dropped below 30% of the bag weight',
          signals=['arm_joint_torque', 'umi_finger_force'])
```

If the decision uses the person's contact forces (which a real robot cannot measure), log it as privileged and disclose it:

```python
ctx.event('release', 'person carries the bag', signals=['recipient_force'], privileged=True)

def disclosures(self):
    return ['privileged: the release decision reads the person contact force from the simulator']
```

The disclosure text must contain the word `privileged` and the word `force` or `load`.

## G9_smoothness

| Part | Requirement |
|---|---|
| a, arm target jumps | the 14 arm joint targets (`q_target[15:29]` in `controller-log.npz`) never change by more than 0.25 rad from one 50 Hz tick to the next |
| b, gripper speed | the gripper (site `left_grasp_center` / `right_grasp_center`) moves at most 0.6 m/s near the person (within 10 cm of the bag to person contact points, or of the person's hands if there are none), during the grasp (0.5 s before the first finger contact to 0.5 s after the bag first counts as lifted in G3) and during the release (0.5 s before the opening command to 0.5 s after the last finger contact); at most 1.2 m/s elsewhere. Checked from 0.5 s before the first finger contact to t_hc or the end |
| c, right arm ramped | the 7 right arm targets never change by more than 0.05 rad per tick (2.5 rad/s) |

Speeds in b are smoothed over 5 physics steps (10 ms).

## Run the checker yourself

`run.py` runs it after every episode. You can run it again on any trial folder:

```bash
python hri_eval/check_gates.py trials/baseline/ep000
```

It prints a pass/fail summary per gate, details for every failing gate, writes `gates.json` into the folder and exits with 0 only if all ten gates pass.

| Option | Default | Meaning |
|---|---|---|
| `--stride N` | 5 | analyse every Nth physics step for G2 to G6 |
| `--frame-stride N` | 1 | check every Nth 30 fps frame in G7 |
| `--reference-xml FILE` | `hri_eval/scene/room_point_fetch.xml` | scene used for the G0 parameter comparison |
| `--v4-only` | off | skip G7 to G9 |
| `--out FILE` | none | write the result to FILE instead of `gates.json` |

From Python:

```python
import sys; sys.path.insert(0, 'hri_eval')
import check_gates
res = check_gates.check('trials/baseline/ep000', write=False)
print(res['summary'])                     # {'G0_invariants': ..., 'G1_motor_replay': ..., ...}
print(res['gates']['G3_pickup'])
```

## gates.json

```
{
  "trial": "...", "checker": "...", "mujoco": "3.14.0", "reference_xml": "...",
  "stride_steps": 5, "eps_contact_N": 0.01, "bag_weight_N": 11.772,
  "trajectory_sha256": "...", "trial_xml_sha256": "...",
  "gates": {
    "G0_invariants": {"pass": ..., ...},
    "G1_motor_replay": {"pass": ..., ...},
    ...
    "G9_smoothness": {"pass": ..., "parts_pass": {...}, ...}
  },
  "all_pass": ...,
  "summary": {"G0_invariants": true, ..., "G9_smoothness": false}
}
```

Every gate has `pass`. G7, G8 and G9 also have `parts_pass` with one entry per part.

## Scores we do not have yet

Comfort and proxemics scores (distance kept to the person's head and body, approach speed and direction, time spent in their personal space) are [on request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Comfort%20and%20proxemics%20scores&feature=Comfort%20and%20proxemics%20scores). Today `report.json` gives the minimum robot to person distance per episode, without a score attached.
