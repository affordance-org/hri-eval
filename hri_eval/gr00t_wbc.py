"""Faithful Python port of NVIDIA GR00T-WholeBodyControl "decoupled WBC" lower-body policy.

Upstream: https://github.com/NVlabs/GR00T-WholeBodyControl  commit 45c03facbf848e0c743091b7d9c17952096ff908
Weights : decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-{Balance,Walk}.onnx
          (NVIDIA Open Model License; see provenance.json / LICENSE copies next to this file)

The policy is an RL MLP that outputs 15 joint-position offsets for the legs (12) and waist (3).
Arms are NOT driven by this policy: upstream the arms come from an external upper-body
(IK / teleop / interpolation) policy and are only *observed* by the lower-body policy.

Every element below cites the upstream file:line it replicates.  Paths are relative to the
repo root at the pinned commit.
  [POL]  decoupled_wbc/control/policy/g1_gear_wbc_policy.py            (deploy policy class)
  [YAML] decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml  (the GEAR_WBC_CONFIG the
         deploy loads: decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml:1)
  [CFG]  decoupled_wbc/control/main/teleop/configs/configs.py
  [DEP]  decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml (deploy robot config)
  [SIM]  decoupled_wbc/control/envs/g1/sim/base_sim.py                  (deploy MuJoCo simulator)
  [S2M]  decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py        (stand-alone MuJoCo runner)
  [UTL]  decoupled_wbc/control/utils/gear_wbc_utils.py
  [WBP]  decoupled_wbc/control/policy/g1_decoupled_whole_body_policy.py
  [SUP]  decoupled_wbc/control/robot_model/supplemental_info/g1/g1_supplemental_info.py
  [BRG]  decoupled_wbc/control/envs/g1/sim/unitree_sdk2py_bridge.py

Nothing in this module applies forces to the simulation.  The upstream elastic band
([SIM]:97-109, 284-312; enabled by [DEP]:22) is deliberately NOT ported.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

UPSTREAM_REPO = "https://github.com/NVlabs/GR00T-WholeBodyControl"
UPSTREAM_SHA = "45c03facbf848e0c743091b7d9c17952096ff908"
HERE = Path(__file__).resolve().parent
DEFAULT_BALANCE_ONNX = HERE / "policy" / "GR00T-WholeBodyControl-Balance.onnx"
DEFAULT_WALK_ONNX = HERE / "policy" / "GR00T-WholeBodyControl-Walk.onnx"

# ---------------------------------------------------------------------------------------------
# Joint ordering.  The observation uses the robot-model "body" group = sorted(lower_body +
# upper_body_no_hands) indices ([POL]:89-96, [SUP]:241-244); with the pinocchio model built from
# g1_29dof_with_hand.urdf this is legs(L,R) -> waist -> left arm -> right arm, i.e. the order of
# body_actuated_joints in [SUP]:44-79, which is also the MuJoCo joint order of the G1 MJCF.
# The 15 actions are the "lower_body" group = legs then waist ([SUP]:241, [POL]:236,
# [WBP]:140-143), matching default_angles in [YAML]:21-23.
# ---------------------------------------------------------------------------------------------
JOINT_NAMES_29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LOWER_BODY_JOINTS = JOINT_NAMES_29[:15]   # policy action order
ARM_JOINTS = JOINT_NAMES_29[15:]          # externally commanded (IK / retarget)

# [YAML]:21-23 default_angles (legs then waist).  Upstream pads with zeros for the remaining
# observed joints ([POL]:102-107), so the arm "default" in the observation is 0 rad.
DEFAULT_ANGLES_15 = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                              -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                              0.0, 0.0, 0.0], dtype=np.float32)
PADDED_DEFAULTS_29 = np.zeros(29, dtype=np.float32)
PADDED_DEFAULTS_29[:15] = DEFAULT_ANGLES_15

# PD gains for the 15 policy joints: [YAML]:13-18 (identical to [DEP]:87-101 MOTOR_KP/KD[0:15],
# which the deploy BodyCommandSender actually sends, see command_sender.py:37-40,101-102).
KP_15 = np.array([150, 150, 150, 200, 40, 40,
                  150, 150, 150, 200, 40, 40,
                  250, 250, 250], dtype=np.float64)
KD_15 = np.array([2, 2, 2, 4, 2, 2,
                  2, 2, 2, 4, 2, 2,
                  5, 5, 5], dtype=np.float64)
# Upstream deploy arm gains ([DEP]:87-101 MOTOR_KP/KD[15:29]).  Not part of the policy; offered as
# the reference choice for the external arm PD.
ARM_KP_14 = np.array([100, 100, 40, 40, 20, 20, 20] * 2, dtype=np.float64)
ARM_KD_14 = np.array([5, 5, 2, 2, 2, 2, 2] * 2, dtype=np.float64)
# Upstream stand-alone runner arm gains ([S2M]:271-280: kp 100, kd 0.5, target 0) for reference.
S2M_ARM_KP = 100.0
S2M_ARM_KD = 0.5
# Upstream "LOW elbow" default upper-body pose ([SUP]:39,292-295; robot_model.py:85-110):
# zeros except shoulder roll +/-0.2 rad.
DEFAULT_ARM_POSE_14 = np.zeros(14)
DEFAULT_ARM_POSE_14[1] = 0.2     # left_shoulder_roll
DEFAULT_ARM_POSE_14[8] = -0.2    # right_shoulder_roll

# Scales: [YAML]:25-29
ANG_VEL_SCALE = 0.5
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
# Sizes: [YAML]:32-34, [POL]:116
NUM_ACTIONS = 15
SINGLE_OBS_DIM = 86
OBS_HISTORY_LEN = 6
NUM_OBS = 516
# Initial / default commands: [YAML]:37-40, constants.py:12-13
DEFAULT_HEIGHT_CMD = 0.74
DEFAULT_RPY_CMD = (0.0, 0.0, 0.0)
FREQ_CMD = 0.75   # kept by upstream ([POL]:49,71) but NOT part of the 86-d observation
# Model selection: [POL]:223-228  ->  ||cmd_raw[:3]|| < 0.05 : Balance, else Walk.
# ([S2M]:298 uses "<= 0.05"; the deploy policy class uses "<".)
POLICY_SWITCH_NORM = 0.05
# Rates: policy at 50 Hz ([CFG]:98 control_frequency=50; [YAML]:8-9 dt 0.005 x decimation 4).
POLICY_DT = 0.02

# Official G1 29-DoF rev_1_0 URDF effort limits (Nm).  The upstream deploy clips at
# motor_effort_limit_list ([DEP]:243-251, [SIM]:320), which is *looser* for the ankles (50 vs 35),
# waist roll/pitch (50 vs 35) and *tighter* for hip roll (88 vs 139).  Our sims use the URDF.
URDF_EFFORT_LIMIT = {
    "hip_pitch": 88.0, "hip_roll": 139.0, "hip_yaw": 88.0, "knee": 139.0,
    "ankle_pitch": 35.0, "ankle_roll": 35.0,
    "waist_yaw": 88.0, "waist_roll": 35.0, "waist_pitch": 35.0,
    "shoulder_pitch": 25.0, "shoulder_roll": 25.0, "shoulder_yaw": 25.0, "elbow": 25.0,
    "wrist_roll": 25.0, "wrist_pitch": 5.0, "wrist_yaw": 5.0,
}
UPSTREAM_DEPLOY_EFFORT_LIMIT_29 = np.array(
    [88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50, 88, 50, 50,
     25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], dtype=np.float64)   # [DEP]:243-251


def urdf_effort_limits(joint_names):
    out = []
    for n in joint_names:
        key = n.replace("left_", "").replace("right_", "").replace("_joint", "")
        out.append(URDF_EFFORT_LIMIT[key])
    return np.array(out, dtype=np.float64)


def quat_rotate_inverse(q, v):
    """[UTL]:36-57 / [S2M]:126-141 (q = w,x,y,z)."""
    w, x, y, z = q
    qc = np.array([w, -x, -y, -z])
    return np.array([
        v[0] * (qc[0] ** 2 + qc[1] ** 2 - qc[2] ** 2 - qc[3] ** 2)
        + v[1] * 2 * (qc[1] * qc[2] - qc[0] * qc[3])
        + v[2] * 2 * (qc[1] * qc[3] + qc[0] * qc[2]),
        v[0] * 2 * (qc[1] * qc[2] + qc[0] * qc[3])
        + v[1] * (qc[0] ** 2 - qc[1] ** 2 + qc[2] ** 2 - qc[3] ** 2)
        + v[2] * 2 * (qc[2] * qc[3] - qc[0] * qc[1]),
        v[0] * 2 * (qc[1] * qc[3] - qc[0] * qc[2])
        + v[1] * 2 * (qc[2] * qc[3] + qc[0] * qc[1])
        + v[2] * (qc[0] ** 2 - qc[1] ** 2 - qc[2] ** 2 + qc[3] ** 2),
    ])


def gravity_orientation(quat_wxyz):
    """[UTL]:60-63: gravity unit vector (0,0,-1) expressed in the pelvis frame."""
    return quat_rotate_inverse(np.asarray(quat_wxyz, dtype=np.float64), np.array([0.0, 0.0, -1.0]))


def torso_rpy_command(R_pelvis, R_torso):
    """Torso-orientation command as computed upstream from the *upper-body target pose*
    ([WBP]:124-134): rpy of R_yaw(pelvis)^T R_torso, pinocchio ZYX convention
    (R = Rz(yaw) Ry(pitch) Rx(roll)).  Only used when the caller derives the command from a
    waist target; by default the command is (0,0,0) as in [YAML]:39."""
    yaw = np.arctan2(R_pelvis[1, 0], R_pelvis[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    R = Rz.T @ R_torso
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw2 = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw2])


class Gr00tWbcPolicy:
    """Lower-body (legs + waist) GR00T decoupled-WBC policy.

    Usage (50 Hz):
        pol = Gr00tWbcPolicy()
        pol.reset()
        pol.set_command(vx=0.4, vy=0.0, yaw_rate=0.0, base_height=0.74, torso_rpy=(0,0,0))
        q15_target = pol.step(pelvis_quat_wxyz, pelvis_ang_vel_body, q29, dq29)
    then every physics step:  tau15 = KP_15*(q15_target - q15) - KD_15*dq15   (target dq = 0,
    feed-forward tau = 0, [POL]:238-241, [SIM]:189-201), clipped to the effort limits.

    Inputs (exactly what upstream observes, [SIM]:258-259 -> [BRG]:193-200 -> [POL]:95-100):
      pelvis_quat_wxyz    : floating-base orientation  (MuJoCo qpos[3:7] of the free joint)
      pelvis_ang_vel_body : pelvis angular velocity in the pelvis frame (MuJoCo free-joint qvel[3:6])
      q29, dq29           : joint positions / velocities in JOINT_NAMES_29 order (arms included;
                            the arms are observed but not commanded by this policy)
    """

    def __init__(self, balance_onnx=DEFAULT_BALANCE_ONNX, walk_onnx=DEFAULT_WALK_ONNX,
                 intra_op_threads=1, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.inter_op_num_threads = 1
        self._sess = {
            "balance": ort.InferenceSession(str(balance_onnx), so, providers=list(providers)),
            "walk": ort.InferenceSession(str(walk_onnx), so, providers=list(providers)),
        }
        for s in self._sess.values():
            assert s.get_inputs()[0].shape[1] == NUM_OBS and s.get_outputs()[0].shape[1] == NUM_ACTIONS
        self._in_name = {k: s.get_inputs()[0].name for k, s in self._sess.items()}
        self.reset()

    # -- state --------------------------------------------------------------------------------
    def reset(self):
        self.action = np.zeros(NUM_ACTIONS, dtype=np.float32)                   # [POL]:45
        self.target_q15 = DEFAULT_ANGLES_15.astype(np.float64).copy()          # [POL]:46
        self.cmd = np.zeros(3, dtype=np.float32)                               # [POL]:47, [YAML]:37
        self.height_cmd = float(DEFAULT_HEIGHT_CMD)                            # [POL]:48, [YAML]:38
        self.rpy_cmd = np.array(DEFAULT_RPY_CMD, dtype=np.float32)             # [POL]:50-52
        # History: deque(maxlen=6), zero-filled on the left, newest last ([POL]:39,150-161;
        # [S2M]:47-50 initialises with 6 zero vectors - identical result).
        self.obs_history = collections.deque(
            [np.zeros(SINGLE_OBS_DIM, dtype=np.float32)] * OBS_HISTORY_LEN, maxlen=OBS_HISTORY_LEN)
        self.obs_buffer = np.zeros(NUM_OBS, dtype=np.float32)
        self.last_model = None
        self.n_steps = 0

    def set_command(self, vx=None, vy=None, yaw_rate=None, base_height=None, torso_rpy=None):
        """vx, vy [m/s] and yaw_rate [rad/s] in the pelvis heading frame (raw, unscaled; the
        policy scales them by CMD_SCALE), base_height [m] (default 0.74), torso_rpy [rad]
        (torso orientation relative to the pelvis yaw frame, default 0)."""
        if vx is not None:
            self.cmd[0] = vx
        if vy is not None:
            self.cmd[1] = vy
        if yaw_rate is not None:
            self.cmd[2] = yaw_rate
        if base_height is not None:
            self.height_cmd = float(base_height)
        if torso_rpy is not None:
            self.rpy_cmd = np.asarray(torso_rpy, dtype=np.float32).copy()

    # -- observation ----------------------------------------------------------------------------
    def compute_single_obs(self, pelvis_quat_wxyz, pelvis_ang_vel_body, q29, dq29):
        """[POL]:68-133 (identical layout to [S2M]:147-240)."""
        q29 = np.asarray(q29, dtype=np.float64)
        dq29 = np.asarray(dq29, dtype=np.float64)
        assert q29.shape == (29,) and dq29.shape == (29,)
        obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:3] = self.cmd[:3] * CMD_SCALE                                    # [POL]:120
        obs[3] = self.height_cmd                                               # [POL]:121
        obs[4:7] = self.rpy_cmd                                                # [POL]:122
        obs[7:10] = np.asarray(pelvis_ang_vel_body) * ANG_VEL_SCALE            # [POL]:113,123
        obs[10:13] = gravity_orientation(pelvis_quat_wxyz)                     # [POL]:112,124
        obs[13:42] = (q29 - PADDED_DEFAULTS_29) * DOF_POS_SCALE                # [POL]:110,127
        obs[42:71] = dq29 * DOF_VEL_SCALE                                      # [POL]:111,128
        obs[71:86] = self.action                                               # [POL]:129 (previous raw action)
        return obs

    def which_model(self):
        return "balance" if float(np.linalg.norm(self.cmd)) < POLICY_SWITCH_NORM else "walk"   # [POL]:223-228

    def step(self, pelvis_quat_wxyz, pelvis_ang_vel_body, q29, dq29):
        """One 50 Hz policy step: append observation, run Balance or Walk, return the 15 joint
        position targets (legs L/R then waist) = action*0.25 + default ([POL]:233-234)."""
        single = self.compute_single_obs(pelvis_quat_wxyz, pelvis_ang_vel_body, q29, dq29)
        self.obs_history.append(single)                                        # [POL]:150
        for i, h in enumerate(self.obs_history):                               # [POL]:157-161 (oldest first)
            self.obs_buffer[i * SINGLE_OBS_DIM:(i + 1) * SINGLE_OBS_DIM] = h
        model = self.which_model()
        out = self._sess[model].run(None, {self._in_name[model]: self.obs_buffer[None, :]})[0]
        self.action = np.asarray(out, dtype=np.float32).reshape(NUM_ACTIONS)   # [POL]:230 (no clipping upstream)
        # float32 arithmetic exactly as upstream (action float32 * scale + float32 defaults), [POL]:234
        self.target_q15 = (self.action * np.float32(ACTION_SCALE) + DEFAULT_ANGLES_15).astype(np.float64)
        self.last_model = model
        self.n_steps += 1
        return self.target_q15.copy()

    def full_targets(self, arm_targets_14):
        """29-vector of joint position targets: policy legs+waist + externally supplied arms."""
        q = np.zeros(29)
        q[:15] = self.target_q15
        q[15:] = np.asarray(arm_targets_14, dtype=np.float64)
        return q

    @staticmethod
    def pd_torque(q_target, q, dq, kp, kd, dq_target=None, tau_ff=None):
        """[UTL]:31-33 / [SIM]:189-201: tau = tau_ff + kp (q* - q) + kd (dq* - dq)."""
        dq_target = 0.0 if dq_target is None else dq_target
        tau = kp * (q_target - q) + kd * (dq_target - dq)
        return tau if tau_ff is None else tau + tau_ff


if __name__ == "__main__":
    # Smoke test: zero observation through both models.
    p = Gr00tWbcPolicy()
    q = np.zeros(29)
    q[:15] = DEFAULT_ANGLES_15
    t = p.step([1, 0, 0, 0], np.zeros(3), q, np.zeros(29))
    print("balance target", np.round(t, 4))
    p.set_command(vx=0.4)
    t = p.step([1, 0, 0, 0], np.zeros(3), q, np.zeros(29))
    print("walk target", np.round(t, 4), p.last_model)
