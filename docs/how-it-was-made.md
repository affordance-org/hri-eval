# How it was made

A real person in a real room, rebuilt in simulation so that a simulated robot can act next to them.

## 1. Record

A person was recorded with cameras in a room while they pointed at a backpack on a low table and waited. In the same recording a second person fetched the bag and handed it over; that took 4.4 s and is the human reference for the pace score.

## 2. Reconstruct the person

The person's body motion was reconstructed as a 21-bone skeleton at 30 fps, in metres in the room frame, together with a skinned avatar mesh of the person (`hri_eval/person/`). The skeleton has no finger bones, so the fingers do not move.

From the avatar, rigid convex collision shapes were fitted to the visible skin of the hands, forearms, pelvis, torso and legs (the left hand as separate finger segments). In every physics step each shape is placed at its recorded bone. The person is kinematic: they move exactly as recorded, and the robot cannot push them.

## 3. Rebuild the room

The room was rebuilt in MuJoCo in the same frame as the recording. The bed and the objects on the table are meshes; the other furniture is modelled as boxes at their places in the room. The backpack is one rigid body made of the scanned parts of the bag as convex collision hulls.

## 4. Add the robot

The robot is the Unitree G1 29-DoF model from MuJoCo Menagerie with a UMI parallel gripper from MuJoCo Menagerie on each wrist. Every joint is a plain torque motor limited to the official URDF effort limit. Walking comes from NVIDIA GR00T-WholeBodyControl: a Python port of its deploy code with the published Balance and Walk weights.

## 5. Check every episode independently

The runner records everything: the full state at every physics step, every motor command, the person's poses. It measures whether anything wrote to the simulation. The checker (`hri_eval/check_gates.py`) then loads the trial's own scene file and re-simulates the episode from the recorded commands alone, and scores that re-simulation. See [scoring.md](scoring.md).

## Modelling choices

Every trial's `README.md` and `summary.json` list these, for every policy (a policy's own choices follow them):

| Choice | Why |
|---|---|
| The backpack stands upright; in the recording it lies on its back | no direct grasp of the lying bag fits the gripper |
| The backpack has no handle or strap | the scanned handle is not in the model; the robot grasps the bag body |
| Backpack mass 1.2 kg, gripper mass 0.8 kg | assumed values |
| Gripper stroke 153 mm wide open, 35 N per finger | assumed; not manufacturer data |
| Stiff, low-dissipation contact between the bag and the finger pads | limits MuJoCo soft-contact creep in the pinch; constant for the whole episode |
| The person replays the recording and does not adapt to the robot | a reactive person is not built yet |
| While waiting, a quiet stretch of the recording plays back and forth | real people do not freeze completely while they wait |
| The robot start pose is chosen, not taken from the recording | chosen so that a forward-looking head camera on the robot would see the person, the pointing arm and the table |

## What is not modelled

The person's reaction to the robot, finger motion, soft tissue and clothing deformation, and sensor noise. Requests for any of these are welcome: [catalog.md](catalog.md).

## More people and rooms

The same process works for other rooms, other people and other tasks. More recordings, and capturing your own room and people, are [on request](https://github.com/affordance-org/hri-eval/issues/new?template=request_feature.yml&title=%5BFeature%5D%20Capturing%20your%20own%20room%20and%20people&feature=Capturing%20your%20own%20room%20and%20people).
