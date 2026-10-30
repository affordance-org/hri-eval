# FAQ

**Do I need a GPU?**
No. MuJoCo, the GR00T walking policy and the checker all run on the CPU. Your own policy may use a GPU if it wants to; `run.py` calls it in the same process.

**Do I need a display?**
No. Nothing is rendered, so it runs on headless servers.

**How long does an episode take?**
It depends on your CPU and your policy. Each episode simulates up to 24 s, and the checker then re-simulates it. On our 8 vCPU Linux machine one baseline episode on its own took about 2.2 minutes of wall time: 39 s to simulate, the rest for the checker and the distance measurement. `--episodes 10 --jobs 5` took about 6 minutes in all. `summary.json` records `wall_time_s` and `realtime_factor`, and `report.json` the wall time per episode including the checker. Use `--jobs` to run episodes in parallel; with `--jobs` the terminal prints one line per finished episode, so a long silence is normal.

**Why is the checker slow on my episode?**
Contact with the person. G7 compares the robot and every contact on the person with the person's visible skin, and every frame or contact step near the skin costs a skin surface build and thousands of sampled points. The checker stops each of these checks at its first failure, which bounds the time, but an episode with contact still checks slower than one without: in our test, an episode in which the robot walked into the table and then into the person took about 4 minutes to check on a busy 8 vCPU machine. Keep the robot off the person: a policy that walks without steering (a constant `vx`) walks into the table and then into the person.

**Is the person real?**
The motion is. A real person was recorded with cameras in a real room, and the eval replays that recording in the room rebuilt in MuJoCo. Their hands, forearms and body are collision shapes fitted to the visible skin, moved with the recorded bones. See [how-it-was-made.md](how-it-was-made.md).

**Does the person react to the robot?**
No. The person points at the backpack and then waits, whatever the robot does. Reactive people are [on request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Reactive%20humans%20that%20respond%20to%20the%20robot&feature=Reactive%20humans%20that%20respond%20to%20the%20robot).

**Can a handover pass if the person never reaches?**
Only if the robot leaves the bag resting on the person's hand or forearm: for the last second or more the person's hand and arm colliders must carry 80% of the bag's weight while every gripper part is 5 cm away. See [scoring.md](scoring.md#g5_handover-score-handover).

**Why does the baseline fail handover and pace?**
It holds the bag out to the person and never lets go. Pace needs a handover, so it fails too.

**Does the robot have to wait for the pointing gesture?**
No. Nothing scores it. The baseline waits for it; your policy may go straight for the backpack.

**What does "valid" mean in the report?**
The checker re-simulated the trial and two integrity gates passed: G0 (the scene, the motors and the physics match the reference scene of your checkout and the policy never wrote to the simulation) and G1 (replaying the recorded motor commands reproduces the recorded motion). The totals count an episode that is not valid as failed for every score.

**Can I use my own robot?**
Not yet. Other robots and your own URDF or MJCF are [on request](https://github.com/affordance-org/hri-eval/issues/new?template=request_robot.yml&title=%5BRobot%5D%20Your%20own%20URDF%20or%20MJCF&robot_name=Your%20own%20URDF%20or%20MJCF).

**Can I evaluate a vision or VLA policy?**
Not yet: there are no rendered camera images. They are [on request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Rendered%20RGB%20and%20depth%20camera%20observations&feature=Rendered%20RGB%20and%20depth%20camera%20observations). Today a policy reads proprioception, person keypoints and object poses.

**Can I use LeRobot, ROS 2 or a remote policy server?**
Not yet. All three are on request: [LeRobot](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20LeRobot%20policy%20adapter&feature=LeRobot%20policy%20adapter), [ROS 2](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20ROS%202%20bridge&feature=ROS%202%20bridge), [remote server](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Remote%20policy%20server%20%28websocket%20or%20gRPC%29&feature=Remote%20policy%20server%20%28websocket%20or%20gRPC%29). Today your policy is a Python class.

**Can I send you a checkpoint instead?**
Hosted evaluation is [on request](https://github.com/affordance-org/hri-eval/issues/new?template=hosted_eval.yml&title=%5BHosted%20eval%5D%20). It is not a running service yet; tell us what you need.

**Is the simulation deterministic?**
The checker requires that replaying the recorded motor commands reproduces the recorded motion to 1e-6 (G1). Whether two runs of your policy produce identical episodes depends on your policy.

**Can I change the scene, the robot or the physics?**
For your own experiments, yes, it is your copy. The checker will not stop you: `run.py` simulates `hri_eval/scene/room_point_fetch.xml` and G0 compares the trial with that same file, so episodes from an edited scene are still valid. `report.md` records the git commit and lists modified tracked files under `hri_eval/`, `run.py` and `requirements.txt`. Leaderboard results need a clean checkout of a public commit.

**Does wall-clock time count?**
No. The simulation waits for your policy every tick. Only simulated time counts (pace, the episode length of up to 24 s).

**Can my policy end the episode early?**
Yes, return `ControlOutput(done=True)`. The gates judge the episode up to that point, and `report.md` shows its end reason and simulated time. A handover still needs 1.0 s of the person holding the bag before the end. A policy that raises an exception fails no fall for that episode.

**What does the exit code of run.py mean?**
0: every episode ran to a normal end and is valid. 1: an episode could not run, raised, diverged or is not valid. 2: unknown scenario. It says nothing about the scores; read `totals` in `report.json` for that.

**Why is the backpack upright when it lies on its back in the recording?**
No direct grasp of the lying bag fits the gripper. The change is disclosed in every trial's `README.md`.

**Where do the walking weights come from?**
NVIDIA GR00T-WholeBodyControl (Balance and Walk ONNX policies), downloaded from the NVlabs GitHub repository on the first run and checked by sha256. They are under the NVIDIA Open Model License and are not stored in this repository.

**What can I do with the data?**
The room model, the scanned objects, the person's avatar and recorded motion are CC BY 4.0 (`licenses/DATA.md`). The code is Apache-2.0. The G1 model is BSD-3-Clause (Unitree Robotics, via MuJoCo Menagerie) and the UMI gripper model MIT. See `licenses/`.

**I found a bug.**
[Report it](https://github.com/affordance-org/hri-eval/issues/new?template=bug.yml&title=%5BBug%5D%20) with the command, the output and your environment.

**I want something that is not in the catalog.**
[Ask for it](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20). We build features in the order people ask.
