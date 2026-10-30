#!/usr/bin/env python3
"""v11 FRONTAL point-fetch-present controller (no handover) on the v9 body scene (rigid backpack, no handle, no strap).

Copy of v10/final/run/point_fetch_ctrl.py (the k04 / final05 controller) with the v11 changes (README.md):
* FRONTAL stance: after the commit the robot walks (auto-planned path from wherever it stands at the commit, any start
  pose) to a stance WEST of the table FACING the bag (heading stance_yaw_deg, default 0 = east) and reaches forward.
* GRASP FRAME OFF THE HULL SEAM: the planned grasp is the scene grasp frame (grasp_site) moved by grasp_pos_bag (bag
  frame) and tilted by grasp_tilt_deg about the bag y axis: the pad centre sits 15 mm off the bag's centre plane, so
  the strap-side pad presses on ONE convex strap hull (bag_1_21) instead of straddling the seam between bag_1_20 and
  bag_1_21 (test-only gantry probe study, probes/); the 15 deg tilt lets the forearm reach forward with a relaxed arm.
* STOW + FRONT CARRY: after the lift the bag is brought (online IK, standing) to a carry pose IN FRONT of the robot,
  then the left-arm joint targets are frozen for the whole walk (no per-tick arm target steps while walking).
* CARRY PATH turn_walk: turn in place (stepping back from the table), walk straight, end FACING the person; the
  presentation stance is solved so that the bag, held at the frozen carry pose, lands at the presentation point.
* --start X Y YAW_DEG builds a start scene (harness/make_start_scene.py: only keyframe start's pelvis pose changes).

v11 FINAL (folder v11/final, this file): params `headcam` = the ASSUMED head camera mount (headcam/headcam_mount.json)
turns on VISIBILITY-GATED perception (class GatedPerception): the v9 pointing/target rule runs only on keypoints that
are inside the head camera frustum and unoccluded (mj_ray from the camera on the scratch state, read only) and on
objects in view; the per-tick visibility is logged (controller-log x_perc_visible, x_perc_in_frame, x_perc_obj_frac,
x_perc_obj_visible, x_perc_cam).  The camera is not added to the physics model (no mass, contact or actuation).

Everything below this line is the v10 docstring (still accurate for the unchanged parts).

v9 POINT, INFER, FETCH AND PRESENT (no handover) on the v9 body scene (rigid backpack, no handle, no strap).

Started from a copy of v7/variants/v7-present-fallback/present_ctrl.py (approach, carry, hold-out presentation, idle
recipient, smoothing, drop detection).  Changes:

* PERCEPTION in the 50 Hz loop (class Perception): ideal perception of the recipient's skeleton (18 keypoints from his
  replayed motion exactly as the runner poses him: Recipient6.full_pose(t) at the tick time) and of the four tabletop
  objects (bounding-box centres of their collision meshes at their TRUE simulated poses).  Pointing gesture = an arm
  whose elbow is extended (> point_elbow_deg), whose wrist is at least point_reach_m horizontally away from the chest and
  above hip height, held for >= point_hold_s.  Pointing ray = shoulder (upper-arm joint) -> wrist (hand joint), origin
  at the wrist (the steadiest of the definitions the avatar allows, see RAY_NOTE).  Each object is scored by the angle
  between the ray and the direction from the wrist to its centre; the robot COMMITS to a target when one object stays
  best by >= target_margin_deg and within target_max_deg for >= target_hold_s.  Logged per tick: pointing, ray, angles,
  target, confidence, keypoints; events see_person / pointing_detected / target_candidates / target_selected.
* The robot BALANCES IN PLACE (zero velocity command, keyframe arm posture) until the target is committed; only then
  does it plan the stance, the approach path and the grasp around the SELECTED object's pose.  A target other than the
  backpack stops the sequence (event target_not_fetchable; the robot keeps standing).
* GRASP: the v9 direct body grasp (scene site bag_grasp_top_flip for the left arm): the wide-open UMI (153 mm gap)
  comes down over the top of the upright bag, fingers straight down, jaws across the bag's front-back thickness, pad
  centred on the bag's centre-of-mass plane; the force-limited servos close to the 35 N limit; lift, carry, present.
  No strap, no handle.

usage: python point_fetch_ctrl.py OUT_DIR [--params JSON|file] [--xml SCENE.xml] [--max-time 18.0] [--no-check]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS = HERE
sys.path.insert(0, str(HARNESS))
import episode as E  # noqa: E402
import recipient6 as R6  # noqa: E402
from recipient import q2R  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
# v11 final: the ASSUMED head camera (mount + projection + mj_ray occlusion helpers; not part of the physics model)
sys.path.insert(0, str(HERE))
import headcam as HC  # noqa: E402

SCENE_XML = HERE / 'scene' / 'room_point_fetch.xml'
TOOL = 0.0515            # left_grasp_center -> left_fingertip_center along the approach axis (measured in the scene)
NAT = 0.6

# Candidate objects on the table: (id, visitor label, scene.glb node name)
OBJECTS = [('backpack', 'Backpack', 'object_backpack'), ('water_bottle', 'Bottle', 'object_water_bottle'),
           ('cup', 'Cup', 'object_cup'), ('candy_box', 'Candy jar', 'object_candy_box')]
# Keypoints of the recipient skeleton: (name, recording bone whose head is the keypoint; None = head tip)
KEYPOINTS = [('pelvis', 'pelvis'), ('spine', 'spine_02'), ('chest', 'spine_03'), ('neck', 'neck_01'), ('head', 'head'),
             ('head_top', None), ('shoulder_l', 'upperarm_l'), ('elbow_l', 'lowerarm_l'), ('wrist_l', 'hand_l'),
             ('shoulder_r', 'upperarm_r'), ('elbow_r', 'lowerarm_r'), ('wrist_r', 'hand_r'), ('hip_l', 'thigh_l'),
             ('knee_l', 'calf_l'), ('ankle_l', 'foot_l'), ('hip_r', 'thigh_r'), ('knee_r', 'calf_r'), ('ankle_r', 'foot_r')]
SKELETON_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (0, 12),
                  (12, 13), (13, 14), (0, 15), (15, 16), (16, 17)]
RAY_NOTE = ('Pointing ray: from the shoulder (upper-arm joint) through the wrist (hand joint), drawn from the wrist. The '
            'avatar has no finger or eye joints. Of the definitions it allows, the shoulder-to-wrist line was the '
            'steadiest during the recorded gesture (1.6 deg mean frame-to-mean jitter, elbow-to-wrist 2.3 deg); rays from '
            'a head point (neck, head joint or head top to the wrist) changed the best object when that point moved by '
            'a few centimetres, so they were not used.')


def min_jerk(u):
    u = min(1.0, max(0.0, u))
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_angle_deg(R1, R2):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R1.T @ R2) - 1) / 2))))


def slerp_R(R0, R1, u):
    from scipy.spatial.transform import Rotation, Slerp
    s = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, R1])))
    return s([min(1.0, max(0.0, u))]).as_matrix()[0]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def unit(v):
    return v / np.linalg.norm(v)


def angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def spline_path(knots, dt=0.02):
    """knots [(t, x, y, yaw)] -> (t, xy, yaw) sampled at dt; clamped cubic spline (zero end velocities)."""
    from scipy.interpolate import CubicSpline
    k = np.array(knots, float)
    k[:, 3] = np.unwrap(k[:, 3])
    ts = np.arange(k[0, 0], k[-1, 0] + 1e-9, dt)
    out = [CubicSpline(k[:, 0], k[:, i], bc_type='clamped')(ts) for i in (1, 2, 3)]
    return ts, np.stack(out[:2], 1), out[2]


DEFAULT = dict(
    # ---- perception (pointing gesture -> target)
    point_elbow_deg=150.0, point_reach_m=0.45, point_hold_s=0.30,
    target_max_deg=15.0, target_margin_deg=5.0, target_hold_s=0.25,
    # ---- approach (planned at the commit, around the selected object's grasp site)
    grasp_site='bag_grasp_top_flip', t_approach=2.4, approach_mid=None, lead=0.3, stop_tol=0.04, v_max=0.8,
    stop_tol_carry=0.05, stance_yaw_deg=-90.0, stance_f=0.32, stance_l=0.16, t_arm_blend=2.0, t_raise=1.0, raise_rel=[0.0, 0.40, 0.85],
    pre_up=0.16, app_up=0.10, t_pre=0.50, pre_tol=0.015, t_pre_max=0.6, t_desc=1.0, t_stop_max=0.8, pf_kp=1.0, pf_ki=0.2, t_close=0.35,
    finger_open=-0.035, finger_close=0.05, wr_grasp=1.0, align_tol=0.0025, t_align_max=0.30,
    slip_max=0.0, r_shoulder_roll=-0.22, base_h=0.74,
    # ---- post-grasp
    t_lift=0.70, lift=0.08, t_hold=0.10,
    backup=[-0.40, 0.06], t_back=1.3, back_yaw_deg=-125.0, t_turn=1.1, turn_end_d=[-0.07, -0.35],
    carry_rel=[0.32, 0.36, 0.95], back_rel=[0.32, 0.30], t_walk=2.0, v_max_carry=0.7,
    wz_max_carry=0.9, walk_d=[0.0, -0.25], t_walk1=0.6, back_u=0.4, d_final=0.20, t_final=0.8,
    # ---- presentation: the bag's top centre (site bag_top_center) at a point in front of the recipient's idle left
    # shoulder (recorded f200) along yaw nfar_yaw_deg, at height present_z (ideal perception of the person)
    present_site='bag_top_center', nfar_yaw_deg=95.0, present_dist=0.43, present_lat=0.0, present_z=1.00,
    present_rel=[0.32, 0.36], t_present=0.45, hand_yaw_lim=[-150.0, -80.0], present_settle=0.60,
    t_carry_extra=0.8, heading_mode='travel', t_present_hold=5.0,
    damp_k=0.0, damp_leak=0.6, damp_max=0.03,
    arm_kp_scale=2.0, ik_iters=10, wr_carry=0.6, cart_ki=0.06, cart_imax=0.03,
    dq_max_left=0.08, dq_max_right=0.02, dq_max_hold=0.02,
    # drop detection (after the lift): robot-side finger closure beyond the grasp, or the TRUE bag height
    drop_z=0.15, drop_close=0.004, t_after_drop=1.0, t_after_stop=2.0,
    # ---- v10 (right-or-lean): side reach from the west of the table (off the bed)
    nat=0.6,                    # natural-range bound of |shoulder yaw|, |wrist roll|, |wrist yaw| [rad] (v9: 0.6)
    torso_rpy_grasp=[0.0, 0.0, 0.0],   # GR00T torso orientation command (roll, pitch, yaw rel. to the pelvis yaw frame)
    t_torso_in=1.5,             # ramp-in of torso_rpy_grasp after the commit [s]
    t_torso_out=1.5,            # ramp-out after the lift (start of the carry walk) [s]
    carry_axis_only=False,      # lift/hold/carry: constrain only the approach axis (fingers straight down), yaw free
    ik_seeds_extra=None,        # extra left-arm IK seeds (7-vectors) for the global solves
    descent_check=True,         # log the UMI-to-bag clearance along the planned straight descent at the replan
    desc_mode='v9',             # 'v9' | 'v10': descend to desc_hold above the grasp with the integral frozen, align the
                                # closing/pad-width axes there (x/y integral only), then insert the last desc_hold slowly
    carry_accel=[1.5, 1.0],
    arm_kp_scale_carry=None,
    carry_arm_frozen=False,     # carry walk with the left-arm joint targets frozen at the lifted posture (no IK steps)
    carry_mode='segments',      # 'segments' (v9 knots: back, turn, walk) | 'straight' (turn in place, one straight walk)
    dq_max_carry=None,          # left-arm joint-target rate limit during the carry walk [rad/tick] (None: dq_max_left)
    base_h_carry=None,
    finger_hold_mode='saturate',  # 'saturate' (v9: closed target, both servos at the 35 N limit) | 'contact'
    finger_squeeze_m=0.006, t_finger_ramp=0.3,          # GR00T base-height command from the start of the carry walk (None: base_h)
    torso_out_at='carry',       # 'carry' | 'hold': when the torso lean ramps back to zero (t_torso_out >= 100 keeps it)    # left-arm PD gain scale during the carry walk and presentation (None: arm_kp_scale)
    carry_frame_tau=0.0,        # >0: carry hand targets in a low-pass filtered pelvis frame (time constant, s)
    back_z=None,                # tool height during the 'back' segment (None: carry_rel z)     # PathFollower acceleration limits (forward/lateral, m/s^2) for the carry walk
    desc_hold=0.03, t_insert=0.6, align_tol_x=0.003, align_tol_z=0.005,
    # ---- v10 final: egocam watch lean (0 = k04 behaviour): torso pitch command while watching, removed from the commit
    watch_pitch=0.0, watch_ramp=0.3, unlean_T=0.5, plan_upright=True,
    plan_waist=None,            # waist (yaw, roll, pitch) assumed by the upright commit-time plans (None: 0, 0, 0)
    recipient_motion_fps=None,  # rate of recipient-motion.json (None: the harness default, 30 fps)
    reason_back='step back west of the table while turning right, arm holds the bag',
    reason_turn='turn toward the person while bringing the bag to the front-left carry pose',
    reason_carry='walk to the presentation stance carrying the bag at the front-left (PI path follower)',
    reason_carry_path='step back west of the table while turning right, turn toward the person, walk to the presentation stance',
    # ---- v11 frontal grasp (README.md)
    grasp_pos_bag=None,         # grasp-centre position in the bag frame (None: the grasp_site position)
    grasp_tilt_deg=0.0,         # tilt of the grasp frame about the bag y axis (+: fingertips toward bag +x)
    grasp_yaw_deg=0.0,          # turn of the closing axis about bag z (deg)
    grasp_pad_x_mm=0.0,         # disclosure only: pad-tip line offset from the bag centre plane used to compute grasp_pos_bag
    approach_mode='via',        # 'via' (v10: fixed approach_mid knots) | 'auto' (turn toward the stance, walk, turn)
    approach_v=0.30, approach_wz=0.7, approach_min_T=2.0, approach_turn_T=0.8, approach_back=0.15,
    t_arm_delay=0.0,            # the arm starts moving t_arm_delay after the plan (approach)
    stow_rel=None,              # None: no stow (v10). [fwd, left, z]: tool target in the pelvis yaw frame after the lift
    t_stow=1.5, stow_hold=0.3,
    t_turn_carry=1.6, carry_back=0.10, v_carry=0.30,
    carry_back_first=0.0,       # v11: >0: duration [s] of a straight step back (heading kept) before the turn
    present_anchor='shoulder_l',  # 'shoulder_l' (v10) | 'chest'
    face_bone='spine_03',       # turn_walk: the robot ends facing this recorded bone of the person (frame 200)
    reason_stow='bring the lifted bag to the carry pose in front of the body (online IK, standing)',
    ik_order='pre_first',       # 'pre_first' (v10) | 'grasp_first' (v11: grasp posture first, the others seeded from it)
    ik_limits=None,             # {left-arm joint index 0..6: [lo, hi]} tighter IK bounds (v11 natural reach)
    pre_vertical=False,
    raise_mid=None,             # v11: intermediate left-arm posture (7 joints) of the approach raise         # v11: pre-grasp / hover points straight above the grasp (vertical descent)
    ik_self_clear=0.0,
    global_max_jump=None,
    ik_w_post=0.0,
    lift_mode='online',         # 'online' (v10: online IK every tick) | 'joint_path' (v11: IK continuation planned at the close)
    lift_n=12, dq_max_lift=None,
    lift_profile='minjerk',     # v11 joint_path lift timing: 'minjerk' | 'easeout' (fast start, slow stop)              # v11: >0 weight of the pull toward the planned arm posture in the online DLS IK       # v11: reject global-IK fallback solutions more than this [rad] from the current targets
    approach_step_T=0.8,
    # ---- v11 final: head camera and VISIBILITY-GATED perception (README.md). None = ideal perception without any
    # visibility test (v9/v10/v11-grasp behaviour). A dict = the ASSUMED head camera mount {name, body, pos, quat_wxyz,
    # fovy_deg, aspect} (headcam/headcam_mount.json): a keypoint is used only when it is inside that camera's frustum
    # (margin vis_margin of the half width/height) and unoccluded (mj_ray from the camera on the scratch state, read
    # only); pointing needs the shoulder, elbow and wrist of the arm plus the chest and both hips; an object is a
    # candidate only when its centre is in the frame and at least vis_obj_frac of its sampled surface is unoccluded.
    start_note=None,            # v11 final: disclosure-only sentence appended to the start-pose line (why this start)
    headcam=None, vis_margin=0.05, vis_obj_frac=0.5, vis_obj_samples=[48, 24], see_min_kp=6,
    arm_ready_at_Q=False,      # v11 auto2: the arm reaches the ready pose before the final step forward        # v11 auto2: minimum duration of the final step forward to the pickup stance [s]          # >0: global IK solutions with the left arm closer than this to the own body are penalised
)


# ============================================================================================ perception
class Perception:
    """Pointing-gesture perception and target inference, called once per 50 Hz tick (causal: it reads only the state
    at the tick time).  Ideal perception: the recipient skeleton is his replayed motion as the runner poses him, the
    object centres come from the TRUE simulated object poses."""

    def __init__(self, m, rcp, p):
        self.m, self.rcp, self.p = m, rcp, p
        rec = rcp.rec
        self.rec = rec
        self.kp_idx = [rec.idx[b] if b else None for _, b in KEYPOINTS]
        self.head_i = rec.idx['head']
        self.head_off = rec.rest_tail[self.head_i] - rec.rest_head[self.head_i]
        self.K = {n: i for i, (n, _) in enumerate(KEYPOINTS)}
        # object centres: bounding-box centre of the collision vertices in the object's own frame
        self.objs = []
        for oid, label, node in OBJECTS:
            if oid == 'backpack':
                b = m.body('backpack').id
                V = []
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] == b and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] or m.geom_conaffinity[g]):
                        Rg = np.zeros(9)
                        mj.mju_quat2Mat(Rg, m.geom_quat[g])
                        V.append(self._verts(g) @ Rg.reshape(3, 3).T + m.geom_pos[g])
                V = np.vstack(V)
                self.objs.append({'id': oid, 'label': label, 'node': node, 'kind': 'body', 'ref': b,
                                  'c_local': 0.5 * (V.min(0) + V.max(0))})
            else:
                g = m.geom(f'object_{oid}').id
                V = self._verts(g)
                self.objs.append({'id': oid, 'label': label, 'node': node, 'kind': 'geom', 'ref': g,
                                  'c_local': 0.5 * (V.min(0) + V.max(0))})
        self.n_obj = len(self.objs)
        self.t_ext0 = None
        self.ext_side = None
        self.detected = False
        self.lead = -1
        self.lead_t0 = None
        self.target = -1
        self.t_commit = None
        self.seen = False
        self.candidates_logged = False
        self.last = None

    def _verts(self, g):
        mid = self.m.geom_dataid[g]
        return self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]].astype(float)

    def centres(self, ds):
        out = []
        for o in self.objs:
            if o['kind'] == 'body':
                R, p = ds.xmat[o['ref']].reshape(3, 3), ds.xpos[o['ref']]
            else:
                R, p = ds.geom_xmat[o['ref']].reshape(3, 3), ds.geom_xpos[o['ref']]
            out.append(p + R @ o['c_local'])
        return np.array(out)

    def keypoints(self, t):
        P, Q, _, _ = self.rcp.full_pose(t)
        kp = np.zeros((len(KEYPOINTS), 3))
        for k, i in enumerate(self.kp_idx):
            kp[k] = P[i] if i is not None else P[self.head_i] + q2R(Q[self.head_i]) @ self.head_off
        return kp

    def arm_state(self, kp, side):
        K = self.K
        sh, el, wr = kp[K[f'shoulder_{side}']], kp[K[f'elbow_{side}']], kp[K[f'wrist_{side}']]
        elbow = angle_deg(sh - el, wr - el)
        reach = float(np.linalg.norm((wr - kp[K['chest']])[:2]))
        hip_z = 0.5 * (kp[K['hip_l']][2] + kp[K['hip_r']][2])
        above = float(wr[2] - hip_z)
        ok = elbow > self.p['point_elbow_deg'] and reach >= self.p['point_reach_m'] and above > 0.0
        return ok, {'elbow_deg': elbow, 'reach_m': reach, 'above_hip_m': above, 'shoulder': sh, 'wrist': wr}

    def step(self, t, ds, robot_xy, robot_yaw):
        p = self.p
        ev = []
        kp = self.keypoints(t)
        cen = self.centres(ds)
        if not self.seen:
            self.seen = True
            d = kp[self.K['pelvis']][:2] - robot_xy
            bearing = math.degrees(wrap(math.atan2(d[1], d[0]) - robot_yaw))
            ev.append(('see_person', 'person perceived (ideal perception of the replayed recording: 18 skeleton '
                       'keypoints at the tick time); objects perceived at their TRUE simulated poses',
                       dict(distance_m=float(np.linalg.norm(d)), bearing_deg=bearing, n_keypoints=len(KEYPOINTS),
                            objects=[o['id'] for o in self.objs], object_centres=cen)))
        arms = {s: self.arm_state(kp, s) for s in ('l', 'r')}
        ext = [s for s in ('l', 'r') if arms[s][0]]
        side = max(ext, key=lambda s: arms[s][1]['reach_m']) if ext else None
        if side is None or side != self.ext_side:
            self.t_ext0 = t if side is not None else None
            self.ext_side = side
        held = (t - self.t_ext0) if side is not None else 0.0
        pointing = side is not None and held >= p['point_hold_s'] - 1e-9
        ray = np.full(6, np.nan)
        ang = np.full(self.n_obj, np.nan)
        if pointing:
            a = arms[side][1]
            d = unit(a['wrist'] - a['shoulder'])
            ray = np.concatenate([a['wrist'], d])
            ang = np.array([angle_deg(d, c - a['wrist']) for c in cen])
            if not self.detected:
                self.detected = True
                ev.append(('pointing_detected', f"pointing gesture: {'left' if side == 'l' else 'right'} arm extended "
                           f"(elbow {a['elbow_deg']:.0f} deg), wrist {a['reach_m']:.2f} m from the chest and "
                           f"{a['above_hip_m']:.2f} m above the hips, held {held:.2f} s",
                           dict(arm='left' if side == 'l' else 'right', elbow_deg=a['elbow_deg'], wrist_reach_m=a['reach_m'],
                                wrist_above_hip_m=a['above_hip_m'], held_s=held, ray=ray)))
            if not self.candidates_logged:
                self.candidates_logged = True
                order = np.argsort(ang)
                ev.append(('target_candidates', 'angle between the pointing ray and the direction to each object centre: '
                           + ', '.join(f"{self.objs[i]['id']} {ang[i]:.1f} deg" for i in order),
                           dict(angles_deg={self.objs[i]['id']: float(ang[i]) for i in range(self.n_obj)},
                                order=[self.objs[i]['id'] for i in order], ray=ray, object_centres=cen)))
            if self.target < 0:
                order = np.argsort(ang)
                best, second = int(order[0]), int(order[1])
                margin = float(ang[second] - ang[best])
                ok = ang[best] <= p['target_max_deg'] and margin >= p['target_margin_deg']
                if ok and self.lead == best:
                    pass
                elif ok:
                    self.lead, self.lead_t0 = best, t
                else:
                    self.lead, self.lead_t0 = -1, None
                if ok and t - self.lead_t0 >= p['target_hold_s'] - 1e-9:
                    self.target = best
                    self.t_commit = t
                    ev.append(('target_selected', f"target committed: {self.objs[best]['id']} ({ang[best]:.1f} deg off the "
                               f"pointing ray, {margin:.1f} deg closer than the {self.objs[second]['id']}) for "
                               f"{t - self.lead_t0:.2f} s",
                               dict(target=self.objs[best]['id'], target_index=best, angle_deg=float(ang[best]),
                                    runner_up=self.objs[second]['id'], margin_deg=margin, held_s=t - self.lead_t0,
                                    angles_deg={self.objs[i]['id']: float(ang[i]) for i in range(self.n_obj)},
                                    target_centre=cen[best], ray=ray)))
        elif self.target < 0:
            self.lead, self.lead_t0 = -1, None
        if self.target >= 0:
            conf = 1.0
        elif pointing and self.lead >= 0:
            conf = min(1.0, (t - self.lead_t0) / p['target_hold_s'])
        else:
            conf = 0.0
        self.last = {'pointing': float(pointing), 'ray': ray, 'angles': ang, 'target': float(self.target),
                     'confidence': conf, 'keypoints': kp, 'centres': cen, 'arm_extended': float(side is not None),
                     'held': held}
        return ev


class GatedPerception(Perception):
    """v11 final: the same pointing gesture and target rule as Perception, applied ONLY to what the ASSUMED head camera
    sees at the tick.  The camera pose is the live torso_link pose (scratch MjData) composed with the mount; a keypoint
    is usable only when it projects inside the frame (margin vis_margin) and mj_ray from the camera reaches it (the
    headcam.Occlusion rule: nothing in front, or the keypoint's own limb collider / the person's surface around it);
    the torso_link body (the head shell behind the lens) is excluded from the ray.  Objects: deterministic samples of
    their collision-mesh vertices; the fraction whose ray first hits the object itself (or nothing before the point)
    and that lie in the frame; an object is a candidate when its centre is in the frame and that fraction >=
    vis_obj_frac.  Everything is read-only (mj_ray on the scratch copy): the simulation state is never written."""

    def __init__(self, m, rcp, p):
        super().__init__(m, rcp, p)
        self.mt = dict(p['headcam'])
        self.mt.setdefault('aspect', HC.ASPECT)
        self.occ = HC.Occlusion(m)
        self.names = [k for k, _ in KEYPOINTS]
        n_bag, n_obj = (int(x) for x in p['vis_obj_samples'])
        rng = np.random.default_rng(0)
        for o in self.objs:
            if o['kind'] == 'body':
                b = o['ref']
                V = []
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] == b and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] or m.geom_conaffinity[g]):
                        Rg = np.zeros(9)
                        mj.mju_quat2Mat(Rg, m.geom_quat[g])
                        V.append(self._verts(g) @ Rg.reshape(3, 3).T + m.geom_pos[g])
                V, n = np.vstack(V), n_bag
            else:
                V, n = self._verts(o['ref']), n_obj
            o['samples'] = V[np.sort(rng.choice(len(V), n, replace=False))] if len(V) > n else V
        self.vis = None
        self.t_seen = None

    def _obj_pose(self, ds, o):
        if o['kind'] == 'body':
            return ds.xmat[o['ref']].reshape(3, 3), ds.xpos[o['ref']]
        return ds.geom_xmat[o['ref']].reshape(3, 3), ds.geom_xpos[o['ref']]

    def _owned(self, o, g):
        if g < 0:
            return False
        if o['kind'] == 'body':
            return int(self.m.geom_bodyid[g]) == o['ref']
        return g == o['ref']

    def visibility(self, ds, kp, cen):
        p, mt = self.p, self.mt
        C, R = HC.cam_pose(self.m, ds, mt)
        u, v, dep = HC.project(C, R, kp, mt['fovy_deg'], mt['aspect'])
        inf = HC.in_frustum(u, v, dep, margin=p['vis_margin'])
        usable = np.zeros(len(kp), bool)
        for j, n in enumerate(self.names):
            if inf[j]:
                usable[j] = self.occ.point_visible(ds, C, kp[j], surface_r=HC.SURFACE_R.get(n, 0.05), own=HC.OWN.get(n, ()),
                                                   own_r=HC.OWN_R.get(n, HC.OWN_R_DEFAULT))[0]
        frac = np.zeros(self.n_obj)
        ovis = np.zeros(self.n_obj, bool)
        cin = np.zeros(self.n_obj, bool)
        for i, o in enumerate(self.objs):
            Ro, po = self._obj_pose(ds, o)
            W = o['samples'] @ Ro.T + po
            uu, vv, dd = HC.project(C, R, W, mt['fovy_deg'], mt['aspect'])
            ino = HC.in_frustum(uu, vv, dd, margin=p['vis_margin'])
            ok = 0
            for X in W[ino]:
                dist, g = self.occ.first_hit(ds, C, X)
                if dist < 0 or dist >= 1 - 1e-3 or self._owned(o, g):
                    ok += 1
            frac[i] = ok / len(W)
            uc, vc, dc = HC.project(C, R, cen[i], mt['fovy_deg'], mt['aspect'])
            cin[i] = bool(HC.in_frustum(uc, vc, dc, margin=p['vis_margin'])[0])
            ovis[i] = cin[i] and frac[i] >= p['vis_obj_frac']
        return {'cam_pos': C, 'cam_R': R, 'uv': np.stack([u, v], 1), 'in_frame': inf, 'usable': usable,
                'obj_frac': frac, 'obj_centre_in': cin, 'obj_visible': ovis}

    def step(self, t, ds, robot_xy, robot_yaw):
        p = self.p
        K = self.K
        ev = []
        kp = self.keypoints(t)
        cen = self.centres(ds)
        vis = self.visibility(ds, kp, cen)
        self.vis = vis
        use = vis['usable']
        n_use = int(use.sum())
        if not self.seen and n_use >= p['see_min_kp'] and (use[K['head']] or use[K['chest']]):
            self.seen = True
            self.t_seen = t
            d = kp[K['pelvis']][:2] - robot_xy
            bearing = math.degrees(wrap(math.atan2(d[1], d[0]) - robot_yaw))
            ev.append(('see_person', f"person in view of the head camera: {n_use} of {len(KEYPOINTS)} skeleton keypoints "
                       'inside its frame and unoccluded (ideal keypoints of the replayed recording, used only where '
                       'visible); objects in view: ' + (', '.join(o['id'] for o, ok in zip(self.objs, vis['obj_visible']) if ok)
                                                       or 'none'),
                       dict(distance_m=float(np.linalg.norm(d)), bearing_deg=bearing, n_keypoints=len(KEYPOINTS),
                            n_visible=n_use, visible=[n for n, ok in zip(self.names, use) if ok],
                            objects=[o['id'] for o in self.objs], objects_visible=[bool(x) for x in vis['obj_visible']],
                            object_centres=cen)))
        arms = {}
        for s in ('l', 'r'):
            ok, a = self.arm_state(kp, s)
            need = [K[f'shoulder_{s}'], K[f'elbow_{s}'], K[f'wrist_{s}'], K['chest'], K['hip_l'], K['hip_r']]
            a['usable'] = bool(all(use[j] for j in need))
            arms[s] = (bool(ok and a['usable'] and self.seen), a)
        ext = [s for s in ('l', 'r') if arms[s][0]]
        side = max(ext, key=lambda s: arms[s][1]['reach_m']) if ext else None
        if side is None or side != self.ext_side:
            self.t_ext0 = t if side is not None else None
            self.ext_side = side
        held = (t - self.t_ext0) if side is not None else 0.0
        pointing = side is not None and held >= p['point_hold_s'] - 1e-9
        ray = np.full(6, np.nan)
        ang = np.full(self.n_obj, np.nan)
        if pointing:
            a = arms[side][1]
            dvec = unit(a['wrist'] - a['shoulder'])
            ray = np.concatenate([a['wrist'], dvec])
            ang = np.array([angle_deg(dvec, c - a['wrist']) if vis['obj_visible'][i] else np.nan for i, c in enumerate(cen)])
            if not self.detected:
                self.detected = True
                ev.append(('pointing_detected', f"pointing gesture seen by the head camera: {'left' if side == 'l' else 'right'} "
                           f"arm extended (elbow {a['elbow_deg']:.0f} deg), wrist {a['reach_m']:.2f} m from the chest and "
                           f"{a['above_hip_m']:.2f} m above the hips, held {held:.2f} s; shoulder, elbow, wrist, chest and "
                           'hips all in view and unoccluded',
                           dict(arm='left' if side == 'l' else 'right', elbow_deg=a['elbow_deg'], wrist_reach_m=a['reach_m'],
                                wrist_above_hip_m=a['above_hip_m'], held_s=held, ray=ray)))
            seen_obj = [i for i in range(self.n_obj) if np.isfinite(ang[i])]
            if not self.candidates_logged and seen_obj:
                self.candidates_logged = True
                order = sorted(seen_obj, key=lambda i: ang[i])
                hidden = [self.objs[i]['id'] for i in range(self.n_obj) if i not in seen_obj]
                ev.append(('target_candidates', 'angle between the pointing ray and the direction to each object in view: '
                           + ', '.join(f"{self.objs[i]['id']} {ang[i]:.1f} deg" for i in order)
                           + (f"; not in view: {', '.join(hidden)}" if hidden else ''),
                           dict(angles_deg={self.objs[i]['id']: float(ang[i]) for i in seen_obj},
                                order=[self.objs[i]['id'] for i in order], not_in_view=hidden, ray=ray,
                                object_centres=cen)))
            if self.target < 0:
                if seen_obj:
                    order = sorted(seen_obj, key=lambda i: ang[i])
                    best = int(order[0])
                    second = int(order[1]) if len(order) > 1 else None
                    margin = float(ang[second] - ang[best]) if second is not None else float('inf')
                    ok = ang[best] <= p['target_max_deg'] and margin >= p['target_margin_deg']
                else:
                    ok, best, second, margin = False, -1, None, float('nan')
                if ok and self.lead == best:
                    pass
                elif ok:
                    self.lead, self.lead_t0 = best, t
                else:
                    self.lead, self.lead_t0 = -1, None
                if ok and t - self.lead_t0 >= p['target_hold_s'] - 1e-9:
                    self.target = best
                    self.t_commit = t
                    run_up = self.objs[second]['id'] if second is not None else 'none'
                    ev.append(('target_selected', f"target committed: {self.objs[best]['id']} ({ang[best]:.1f} deg off the "
                               f"pointing ray, {margin:.1f} deg closer than the {run_up}) for "
                               f"{t - self.lead_t0:.2f} s; all objects compared were in view of the head camera",
                               dict(target=self.objs[best]['id'], target_index=best, angle_deg=float(ang[best]),
                                    runner_up=run_up, margin_deg=margin, held_s=t - self.lead_t0,
                                    angles_deg={self.objs[i]['id']: float(ang[i]) for i in seen_obj},
                                    target_centre=cen[best], ray=ray)))
        elif self.target < 0:
            self.lead, self.lead_t0 = -1, None
        if self.target >= 0:
            conf = 1.0
        elif pointing and self.lead >= 0:
            conf = min(1.0, (t - self.lead_t0) / p['target_hold_s'])
        else:
            conf = 0.0
        self.last = {'pointing': float(pointing), 'ray': ray, 'angles': ang, 'target': float(self.target),
                     'confidence': conf, 'keypoints': kp, 'centres': cen, 'arm_extended': float(side is not None),
                     'held': held}
        return ev

    def log(self):
        v = self.vis
        return {'perc_visible': v['usable'].astype(float), 'perc_in_frame': v['in_frame'].astype(float),
                'perc_uv': np.where(np.isfinite(v['uv']), v['uv'], np.nan), 'perc_obj_frac': v['obj_frac'], 'perc_obj_visible': v['obj_visible'].astype(float),
                'perc_obj_centre_in': v['obj_centre_in'].astype(float), 'perc_seen': float(self.seen),
                'perc_cam': np.concatenate([v['cam_pos'], v['cam_R'].ravel()])}


# ============================================================================================ runner
class PointFetchRunner(E.EpisodeRunner):
    """EpisodeRunner whose recipient disclosure states what THIS episode does with the recipient (pointing gesture
    from his own recording, then idle only: no adaptation, no curl, no contact)."""

    def disclosures(self):
        base = super().disclosures()
        for i, x in enumerate(base):
            if x.startswith('Robot start pose = '):
                base[i] = self.start_pose_disclosure(x)
            if x.startswith('The recipient is a REPLAYED RECORDING'):
                base[i] = ('The recipient is a REPLAYED RECORDING (kinematic, infinite mass): his clip plays from frame 120 '
                           '(his own pointing gesture, frames 128 to 163, which the robot perceives) into his quietest '
                           'recorded window (frames 183 to 219) and ping-pongs there for the rest of the episode (it also '
                           'plays backward, at most 0.45 x real time) instead of freezing. He is NOT adapted: no reach, no '
                           'arm IK, and the finger curl stays 0. His hands, forearms, pelvis, torso and legs are kinematic '
                           'colliders posed at his recorded bones; they collide with the robot and the bag, not with the '
                           'room. NO HANDOVER: the robot presents the bag in front of his left hand and keeps holding it; '
                           'he never touches it.')
        return base


    def start_pose_disclosure(self, line):
        """The start-pose line for the keyframe actually simulated, compared with the v9 keyframe (the fetcher's
        recorded frame-120 pelvis pose)."""
        import re
        txt = Path(E.V9_XML).read_text()
        rec = [float(v) for v in re.search(r'<key\s+name="start"\s+qpos="([^"]*)"', txt).group(1).split()[:7]]
        q0 = self.m.key(self.keyframe).qpos[:7]
        tail = line.split(';', 1)[1].strip() if ';' in line else ''
        if self.keyframe == 'start' and all(abs(float(a) - b) < 1e-9 for a, b in zip(q0, rec)):
            return line
        yaw = math.degrees(math.atan2(2 * (q0[3] * q0[6] + q0[4] * q0[5]), 1 - 2 * (q0[5] ** 2 + q0[6] ** 2)))
        yaw_rec = math.degrees(math.atan2(2 * (rec[3] * rec[6] + rec[4] * rec[5]), 1 - 2 * (rec[5] ** 2 + rec[6] ** 2)))
        same_pos = all(abs(float(a) - b) < 1e-9 for a, b in zip(q0[:3], rec[:3]))
        assert abs(q0[4]) < 1e-12 and abs(q0[5]) < 1e-12, 'start pose must be upright (yaw only)'
        tail_s = f' The {tail}' if tail.startswith('backpack') else (f' {tail}' if tail else '')
        if same_pos:
            return (f"Robot start pose: the fetcher's recorded frame-120 pelvis position (x {q0[0]:.3f} m, y {q0[1]:.3f} m) "
                    f"and height, with the heading turned toward the table: {yaw:.1f} deg instead of the recorded "
                    f"{yaw_rec:.1f} deg (a free choice, not in the recording; collision free at the start); default standing "
                    'posture.' + tail_s)
        note = (self.controller.p.get('start_note') or '') if hasattr(self.controller, 'p') else ''
        return (f"Robot start pose: pelvis at x {q0[0]:.3f} m, y {q0[1]:.3f} m, height {q0[2]:.3f} m, heading {yaw:.1f} deg "
                f"(a free choice, not in the recording: the fetcher's recorded frame-120 pelvis is at x {rec[0]:.3f} m, "
                f"y {rec[1]:.3f} m, heading {yaw_rec:.1f} deg; collision free at the start); default standing posture."
                + (f' {note}' if note else '') + tail_s)


class PointFetch:
    def __init__(self, params=None):
        self.p = dict(DEFAULT)
        if params:
            self.p.update(params)
        self.loco = E.Gr00tLocomotion()

    # ------------------------------------------------------------------------------------------ helpers
    def site_frame(self, ds, name):
        s = self.m.site(name).id
        return ds.site_xpos[s].copy(), ds.site_xmat[s].reshape(3, 3).copy()

    def tool_pose(self, ds):
        s = self.m.site('left_grasp_center').id
        R = ds.site_xmat[s].reshape(3, 3).copy()
        return ds.site_xpos[s] + R[:, 2] * TOOL, R

    def grasp_target(self, ds):
        """TRUE grasp frame of the selected object (privileged): fingertip-centre position and UMI base axes.  v11: the
        scene grasp site moved to grasp_pos_bag (bag frame) and tilted by grasp_tilt_deg about the bag y axis."""
        ps, Rs = self.site_frame(ds, self.p['grasp_site'])
        p = self.p
        if p['grasp_pos_bag'] is None and not p['grasp_tilt_deg'] and not p['grasp_yaw_deg']:
            return ps + Rs[:, 2] * TOOL, Rs
        b = self.m.body('backpack').id
        Rb, pb = ds.xmat[b].reshape(3, 3), ds.xpos[b]
        R_loc = Rb.T @ Rs
        # the test-only probe's grasp-frame construction (v9/probes/grasp_probe.py grasp_frame): closing axis = the
        # site's closing axis turned by grasp_yaw_deg about bag z; approach = (sin t, 0, -cos t) in the bag frame
        # (+tilt: fingertips toward bag +x) orthogonalised against it; pad width = approach x closing
        t = math.radians(p['grasp_tilt_deg'])
        yw = math.radians(p['grasp_yaw_deg'])
        Rz = np.array([[math.cos(yw), -math.sin(yw), 0.0], [math.sin(yw), math.cos(yw), 0.0], [0.0, 0.0, 1.0]])
        xg = Rz @ R_loc[:, 0]
        za = np.array([math.sin(t), 0.0, -math.cos(t)])
        zg = unit(za - xg * float(za @ xg))
        R = Rb @ np.column_stack([xg, np.cross(zg, xg), zg])
        pos = pb + Rb @ np.asarray(p['grasp_pos_bag'], float) if p['grasp_pos_bag'] is not None else ps
        return pos + R[:, 2] * TOOL, R

    def global_ik(self, qpos_full, pos, R, q_seeds=(), w_ori=0.5, prefer_first=False):
        m = self.m
        d = mj.MjData(m)
        d.qpos[:] = qpos_full
        site = m.site('left_grasp_center').id
        L = self.qadr[15:22]
        lo, hi = self.ik.lo[:7], self.ik.hi[:7]

        def fk(q):
            d.qpos[L] = q
            mj.mj_kinematics(m, d)
            Rs = d.site_xmat[site].reshape(3, 3)
            return d.site_xpos[site] + Rs[:, 2] * TOOL, Rs

        def res(q):
            p, Rs = fk(q)
            o = np.concatenate([w_ori * (Rs[:, 0] - R[:, 0]), w_ori * (Rs[:, 2] - R[:, 2])])
            return np.concatenate([20 * (p - pos), o, 0.02 * q])
        best = None
        ft = np.zeros(6)

        def self_clear():
            # v11: clearance of the left arm / UMI to the robot's own torso, pelvis and legs (IK solutions that put the
            # arm through the body are rejected: the global solve has no collision rows)
            dmin = 0.2
            for a in self.g_arm_self:
                for b in self.g_body_self:
                    if np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b]) > m.geom_rbound[a] + m.geom_rbound[b] + 0.05:
                        continue
                    dmin = min(dmin, mj.mj_geomDistance(m, d, a, b, 0.05, ft))
            return dmin
        sc_min = self.p['ik_self_clear']
        for q0 in q_seeds:
            r = least_squares(res, np.clip(q0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=400)
            p, Rs = fk(r.x)
            pe = float(np.linalg.norm(p - pos))
            ang = rot_angle_deg(R, Rs)
            if sc_min > 0 and self_clear() < sc_min:
                pe += 0.05                       # penalised: arm through the own body
            if best is None or pe + 0.002 * ang < best[0] + 0.002 * best[1]:
                best = (pe, ang, r.x.copy(), Rs.copy())
            if prefer_first and pe < 0.003 and ang < 3.0:
                # v11: keep the solution of the first (posture-continuous) seed when it is accurate enough
                return (pe, ang, r.x.copy(), Rs.copy())
        return best

    def corrected_xy(self, tgt, tpos, R):
        """Cartesian integral on the closing / pad-width axes only (no integral along the approach axis): the tool
        cannot be driven deeper than its target by integral wind-up (v10 descent)."""
        p = self.p
        if p['pre_vertical']:
            # v11 vertical descent: no integral along the world vertical (the descent direction)
            e = tgt - tpos
            e[2] = 0.0
            self.e_int = np.clip(self.e_int + p['cart_ki'] * e, -p['cart_imax'], p['cart_imax'])
            return tgt + self.e_int
        e = R.T @ (tgt - tpos)
        e[2] = 0.0
        self.e_int = np.clip(self.e_int + p['cart_ki'] * (R @ e), -p['cart_imax'], p['cart_imax'])
        return tgt + self.e_int

    def plan_lift_path(self, ctx):
        """v11: joint path of the lift, planned once at the close from the measured pose: damped-least-squares IK
        continuation (each waypoint solved from the previous one, collision rows on) for the tool raised straight up
        in lift_n steps with the grasp orientation fixed in the robot frame.  Followed with a min-jerk time scaling and
        no online IK, so the arm cannot jump to another posture branch while it carries the bag up."""
        p = self.p
        R = self.R_rel(ctx)
        q = self.q_out.copy()
        q[7:] = self.right_cmd
        path = [q[:7].copy()]
        errs = []
        for k in range(1, p['lift_n'] + 1):
            tgt = self.rel_target(ctx, self.rel_grasp + np.array([0, 0, p['lift'] * k / p['lift_n']])) + self.e_int
            tg = {'left': dict(pos=tgt, R=R, axis=None, wp=1.0, wr=p['wr_carry'])}
            q, info = self.ik.solve(ctx.qpos, tg, q0=q, iters=30, fixed=list(range(7, 14)))
            q = q.copy()
            q[7:] = self.right_cmd
            path.append(q[:7].copy())
            errs.append(info['err']['left'])
        self.lift_path = np.array(path)
        steps = np.abs(np.diff(self.lift_path, axis=0)).max(1)
        ctx.event('lift_planned', 'joint path of the lift planned from the measured pose (IK continuation, grasp '
                  'orientation fixed in the robot frame)', n_waypoints=len(path), max_step_rad=float(steps.max()),
                  pos_err_m=np.array([e[0] for e in errs]), ang_err_rad=np.array([e[1] for e in errs]),
                  q_top_left=self.lift_path[-1])

    def back(self, R, dist):
        """offset from the grasp to the pre-grasp / hover points: back along the approach axis (v10), or straight up
        (v11 pre_vertical: the tilted open gripper comes down vertically over the bag top)."""
        return np.array([0.0, 0.0, dist]) if self.p['pre_vertical'] else -R[:, 2] * dist

    def corrected(self, tgt, tpos, freeze=False):
        p = self.p
        if not freeze:
            self.e_int = np.clip(self.e_int + p['cart_ki'] * (tgt - tpos), -p['cart_imax'], p['cart_imax'])
        return tgt + self.e_int

    def solve_ik(self, ctx, pos, R=None, wr=0.3, iters=None, allow_global=True, axis=None):
        tg = {'left': dict(pos=pos, R=R, axis=axis if R is None else None, wp=1.0, wr=wr)}
        q0 = self.q_ik if self.q_ik is not None else ctx.q29[15:].copy()
        q0 = q0.copy()
        q0[7:] = self.right_cmd
        kw = {}
        if self.p['ik_w_post'] > 0 and self.q_nom_cur is not None:
            # v11: weak pull toward the planned (natural) arm posture so that the DLS stays on that posture branch
            qn = q0.copy()
            qn[:7] = self.q_nom_cur
            kw = dict(q_nom=qn, w_post=self.p['ik_w_post'])
        q, info = self.ik.solve(ctx.qpos, tg, q0=q0, iters=iters or self.p['ik_iters'], fixed=list(range(7, 14)), **kw)
        if allow_global and info['err']['left'][0] > 0.01 and R is not None:
            pe, ang, ql, _ = self.global_ik(ctx.qpos, pos, R, [q0[:7], q[:7], self.q_pre_plan[:7]])
            gmj = self.p['global_max_jump']
            if gmj is not None and float(np.max(np.abs(ql - q0[:7]))) > gmj:
                # v11: a global solution on another arm-posture branch would swing the arm (and the bag) through a
                # large joint jump; keep the continuous DLS solution instead
                pe = np.inf
            if pe < info['err']['left'][0]:
                q = q.copy()
                q[:7] = ql
                info['err']['left'] = (pe, math.radians(ang))
                self.n_global += 1
        q[7:] = self.right_cmd
        self.q_ik = q
        self.ik_err = info['err']['left']
        return q

    def update_frame(self, ctx):
        """low-pass filtered pelvis pose (x, y, yaw) for the carry targets: the hand target follows the body's walk
        but not its step-to-step sway, which the arm then cancels (carry_frame_tau > 0)."""
        tau = self.p['carry_frame_tau']
        z = np.array([ctx.base_pos[0], ctx.base_pos[1], ctx.base_yaw])
        rxy, ryaw, _, _ = self.pf.ref(ctx.t)
        ref = np.array([rxy[0], rxy[1], ryaw])
        dev = z - ref
        dev[2] = wrap(dev[2])
        if self.fdev is None or tau <= 0:
            self.fdev = dev
        else:
            # filter the deviation from the planned (smooth) path, not the pose itself: no lag of the mean motion
            self.fdev = self.fdev + min(1.0, 0.02 / tau) * (dev - self.fdev)
        self.fbase = ref + self.fdev

    def rel_target(self, ctx, rel):
        pb = ctx.base_pos
        R = rotz(ctx.base_yaw)
        if self.use_fbase and self.fbase is not None:
            pb = np.array([self.fbase[0], self.fbase[1], ctx.base_pos[2]])
            R = rotz(self.fbase[2])
        g = pb + R @ np.array([rel[0], rel[1], 0.0])
        g[2] = rel[2]
        return g

    def R_rel(self, ctx):
        if self.use_fbase and self.fbase is not None:
            return rotz(self.fbase[2]) @ self.R_grasp_rel
        return rotz(ctx.base_yaw) @ self.R_grasp_rel

    def carry_ik(self, ctx, tgt, tpos, wr):
        """carry-phase IK: full grasp orientation fixed in the robot frame (v9), or (carry_axis_only) only the approach
        axis straight down with the yaw left free (the bag keeps hanging under the pads; the DLS solve starts from the
        current joints, so the yaw stays near the current one)."""
        if self.p['carry_axis_only']:
            return self.solve_ik(ctx, self.corrected(tgt, tpos), R=None, axis=np.array([0.0, 0.0, -1.0]), wr=wr)
        return self.solve_ik(ctx, self.corrected(tgt, tpos), R=self.R_rel(ctx), wr=wr)

    def seeds(self):
        s = [np.array([-1.070, 0.212, 0.116, 1.113, -0.058, 1.517, -0.087]),
             np.array([-1.71, 0.1, -0.07, 1.94, 0.29, 1.25, -0.3]), np.array([-1.39, 0.39, -0.33, 1.4, 0.31, 1.47, -0.58]),
             np.array([-1.0, 0.3, 0.0, 1.2, 0.0, 0.5, 0.0]), np.array([-0.8, 0.2, 0.0, 0.6, 0.0, 1.2, 0.0])]
        return [np.array(x, float) for x in (self.p['ik_seeds_extra'] or [])] + s

    def descent_clearance(self, qpos_full, q_seed, n=8):
        """min distance of the open left UMI (base, holders, fingers) and wrist links to the bag hulls along the straight
        descent p_pre -> p_grasp (global IK at n points, measured pelvis pose); planning check, logged only."""
        m = self.m
        d = mj.MjData(m)
        d.qpos[:] = qpos_full
        L = self.qadr[15:22]
        ft = np.zeros(6)
        out = []
        q = q_seed
        for u in np.linspace(0.0, 1.0, n):
            pt = (1 - u) * self.p_pre + u * self.p_grasp
            pe, ang, q, _ = self.global_ik(qpos_full, pt, self.R_grasp, [q])
            d.qpos[L] = q
            mj.mj_forward(m, d)
            best = 1.0
            for a in self.g_umi:
                for b in self.g_bag:
                    best = min(best, mj.mj_geomDistance(m, d, a, b, 0.2, ft))
            out.append((float(u), float(best), float(pe), float(ang)))
        return out

    def swing(self, ds, tpos):
        """horizontal offset of the TRUE bag centre of mass from the fingertip centre, minus its value at the lift
        (privileged); the anti-sway term moves the hand toward it (pendulum pivot control)."""
        if self.bag_id is None:
            return np.zeros(2)
        off = (ds.xipos[self.bag_id] - tpos)[:2].copy()
        if self.swing0 is None:
            return np.zeros(2)
        return off - self.swing0

    def slip_comp(self, ds, tpos):
        if self.p['slip_max'] <= 0:
            return np.zeros(3)
        lt = ds.site_xpos[self.m.site(self.p['grasp_site']).id][2]
        off = lt - tpos[2]
        if self.slip_ref is None:
            self.slip_ref = off
        self.slip = float(np.clip(self.slip_ref - off, 0.0, self.p['slip_max']))
        return np.array([0, 0, self.slip])

    def damped(self, ctx, tgt, ds, tpos, on=True):
        p = self.p
        dt = 0.02
        if on and p['damp_k'] > 0:
            sw = self.swing(ds, tpos)
            self.damp = self.damp + dt * (p['damp_k'] * sw - p['damp_leak'] * self.damp)
        else:
            self.damp = self.damp * max(0.0, 1 - dt * 4.0)
        n = np.linalg.norm(self.damp)
        if n > p['damp_max']:
            self.damp *= p['damp_max'] / n
        return tgt + np.array([self.damp[0], self.damp[1], 0.0])

    def robot_finger_load(self, ds):
        """Vertical force the robot's LEFT UMI fingers exert on the bag + normal force per finger: the robot's OWN
        contacts (what an ideal wrist force/torque sensor and the finger pads would measure).  Logged only."""
        m = self.m
        f6 = np.zeros(6)
        fz = 0.0
        pads = [0.0, 0.0]
        for i in range(ds.ncon):
            c = ds.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 in self.lfing and b2 in self.bag_bodies:
                sgn, fb = 1.0, b1
            elif b2 in self.lfing and b1 in self.bag_bodies:
                sgn, fb = -1.0, b2
            else:
                continue
            mj.mj_contactForce(m, ds, i, f6)
            F2 = c.frame.reshape(3, 3).T @ f6[:3]
            fz += float(F2[2]) * sgn
            pads[self.lfing[fb]] += float(f6[0])
        return fz, pads

    def finger_q(self, ctx):
        return float(np.mean(ctx.qpos[self.fq_adr]))

    # ------------------------------------------------------------------------------------------ reset
    def reset(self, ctx):
        p = self.p
        self.m = ctx.m
        m = self.m
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in E.JOINT_NAMES_29])
        self.fq_adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in E.FINGER_JOINTS[:2]])
        self.bag_bodies = E.bag_subtree(m)
        self.bag_id = None
        self.lfing = {}
        for b in range(m.nbody):
            n = m.body(b).name
            if n.startswith('left_umi_left_finger'):
                self.lfing[b] = 0
            elif n.startswith('left_umi_right_finger'):
                self.lfing[b] = 1
        self.loco.reset()
        coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        bname = lambda g: m.body(int(m.geom_bodyid[g])).name  # noqa: E731
        self.g_umi = [g for g in coll if bname(g).startswith('left_umi') or bname(g) in ('left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link')]
        self.g_bag = [g for g in coll if int(m.geom_bodyid[g]) in self.bag_bodies]
        self.g_arm_self = [g for g in coll if bname(g).startswith(('left_shoulder_yaw', 'left_elbow', 'left_wrist', 'left_umi'))]
        self.g_body_self = [g for g in coll if bname(g) in ('torso_link', 'pelvis', 'pelvis_contour_link', 'head_link')
                            or any(k in bname(g) for k in ('hip_', 'knee', 'ankle'))]
        self.ik = E.OnlineArmIK(m, natural=True, natural_limit=p['nat'], tool_offset={'left': TOOL, 'right': 0.0},
                                pair_spec=E.DEFAULT_PAIRS + [('left_hand', 'table', 0.02), ('left_forearm', 'table', 0.03),
                                                              ('right_hand', 'table', 0.03)])
        for j, (lo_, hi_) in (p['ik_limits'] or {}).items():
            # v11: tighter left-arm bounds for a natural reach (no elbow hyperextension, no arm across the chest)
            j = int(j)
            self.ik.lo[j] = max(self.ik.lo[j], lo_)
            self.ik.hi[j] = min(self.ik.hi[j], hi_)
        self.arm_key = ctx.q29[15:].copy()
        self.right_target = self.arm_key[7:].copy()
        self.right_target[1] = p['r_shoulder_roll']
        self.right_cmd = self.arm_key[7:].copy()
        self.rcp = ctx.recipient
        rec = self.rcp.rec
        # the recipient's left shoulder and hand in his idle window (he idles in f183-f219 after his gesture)
        self.rcp_shoulder = rec.P[200, rec.idx['upperarm_l']].copy()
        self.rcp_hand = rec.P[200, rec.idx['hand_l']].copy()
        if not any(d.get("op") == "idle" for d in self.rcp.decisions):   # run.py already set it
            self.rcp.idle(0.0)
        ctx.event('recipient_idle', 'the recipient plays his recording on (his own pointing gesture f128-f163) into his '
                  'quietest recorded window (f183-f219) and ping-pongs there for the whole episode (no reach, no curl)')
        self.perc = GatedPerception(m, self.rcp, p) if p['headcam'] else Perception(m, self.rcp, p)
        gate = {}
        if p['headcam']:
            gate = dict(headcam=dict(self.perc.mt), visibility_gated=True, vis_margin=p['vis_margin'],
                        vis_obj_frac=p['vis_obj_frac'], vis_obj_samples=list(p['vis_obj_samples']),
                        see_min_kp=p['see_min_kp'],
                        visibility_rule=('a keypoint is used only when it projects inside the head camera frame (margin '
                                         f"{p['vis_margin']:.2f} of the half width and height) and mj_ray from the camera "
                                         'reaches it (nothing in front of it, or only its own limb collider or the person\'s '
                                         'surface around it); pointing needs the shoulder, elbow and wrist of the arm plus '
                                         'the chest and both hips; an object is a candidate only when its centre is in the '
                                         f"frame and at least {p['vis_obj_frac'] * 100:.0f}% of its sampled surface is "
                                         'unoccluded; the robot body behind the lens (torso_link) is ignored by the ray'))
        ctx.event('perception_config', 'pointing gesture and target rule (see describe()); ray definition: ' + RAY_NOTE,
                  point_elbow_deg=p['point_elbow_deg'], point_reach_m=p['point_reach_m'], point_hold_s=p['point_hold_s'],
                  target_max_deg=p['target_max_deg'], target_margin_deg=p['target_margin_deg'],
                  target_hold_s=p['target_hold_s'], objects=[o['id'] for o in self.perc.objs],
                  keypoints=[k for k, _ in KEYPOINTS], skeleton_edges=np.array(SKELETON_EDGES),
                  ray_definition=RAY_NOTE, object_centre='bounding-box centre of the collision mesh at the TRUE pose',
                  **gate)
        self.ph = 'watch'
        self.ph_t0 = 0.0
        self.q_arm_cmd = self.arm_key.copy()
        self.q_out = self.arm_key.copy()
        self.q_ik = None
        self.q_pre_plan = self.arm_key.copy()
        self.q_gr_plan = self.arm_key[:7].copy()
        self.q_nom_cur = None
        self.fing = p['finger_open']
        self.v = np.zeros(3)
        self.ik_err = (np.nan, np.nan)
        self.e_int = np.zeros(3)
        self.damp = np.zeros(2)
        self.swing0 = None
        self.replanned = False
        self.slip_ref = None
        self.slip = 0.0
        self.torso = (0.0, 0.0, 0.0)
        self.lean_at_commit = 0.0
        self.t_carry0 = None
        self.t_hold0 = None
        self.t_stow0 = None
        self.face_pt = None
        self.q_contact = None
        self.t_fhold0 = None
        self.inserting = False
        self.fbase = None
        self.fdev = None
        self.use_fbase = False
        self.n_global = 0
        self.load = 0.0
        self.pads = [0.0, 0.0]
        self.lifted = False
        self.dropped = False
        self.fq_pinch = None
        self.planned = False
        self.grasp_log = np.full(12, np.nan)       # planned grasp target (pos 3 + R 9) at the plan
        self.t_plan = np.nan

    # ------------------------------------------------------------------------------------------ planning (at commit)
    def plan_after_commit(self, ctx, ds):
        """Stance, approach path and grasp around the SELECTED object's pose (TRUE pose: privileged)."""
        p = self.p
        m = self.m
        obj = self.perc.objs[self.perc.target]
        if obj['id'] != 'backpack':
            ctx.event('target_not_fetchable', f"the selected target ({obj['id']}) has no grasp in this controller (only "
                      'the backpack grasp is modelled): the robot stops and keeps standing', target=obj['id'])
            self.set_phase(ctx, 'stand')
            self.q_stand = self.q_out.copy()
            self.stand_tgt = self.tool_pose(ds)[0]
            self.t_done = ctx.t + p['t_after_stop']
            return
        self.bag_id = m.body('backpack').id
        self.p_grasp, self.R_grasp = self.grasp_target(ds)
        self.p_pre = self.p_grasp + self.back(self.R_grasp, p['pre_up'])
        yaw = math.radians(p['stance_yaw_deg'])
        self.R_grasp_rel = rotz(-yaw) @ self.R_grasp
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        g0 = self.p_grasp - self.R_grasp[:, 2] * TOOL           # grasp-centre site position (the scene's grasp frame)
        self.stance = (g0 - p['stance_f'] * fwd - p['stance_l'] * left)[:2]
        qs = ctx.qpos.copy()
        q_now = ctx.qpos.copy()
        if p['watch_pitch'] != 0.0 and p['plan_upright']:
            # plan with the upright torso (keyframe waist 0, 0, 0; yaw-only pelvis): the watch lean is removed before
            # the arm reaches these postures
            w_up = np.array(p['plan_waist'] if p['plan_waist'] is not None else [0.0, 0.0, 0.0], float)
            qs[self.qadr[12:15]] = w_up
            q_now[self.qadr[12:15]] = w_up
            q_now[3:7] = [math.cos(ctx.base_yaw / 2), 0, 0, math.sin(ctx.base_yaw / 2)]
        qs[0:2] = self.stance
        qs[2] = p['base_h'] + 0.005
        qs[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        seeds = self.seeds()
        if p['ik_order'] == 'grasp_first':
            # v11: grasp posture first (natural seeds), then the pre-grasp and ready postures seeded from it, so the
            # whole reach stays on one posture branch
            pe2, ang2, q_gr_l, _ = self.global_ik(qs, self.p_grasp, self.R_grasp, seeds, prefer_first=True)
            pe, ang, q_pre_l, _ = self.global_ik(qs, self.p_pre, self.R_grasp, [q_gr_l] + seeds, prefer_first=True)
        else:
            pe, ang, q_pre_l, _ = self.global_ik(qs, self.p_pre, self.R_grasp, seeds)
            pe2, ang2, q_gr_l, _ = self.global_ik(qs, self.p_grasp, self.R_grasp, [q_pre_l] + seeds)
        self.q_gr_plan = q_gr_l.copy()
        self.q_pre_plan = np.concatenate([q_pre_l, self.right_cmd])
        pe3, ang3, q_app_l, _ = self.global_ik(qs, self.p_pre + self.back(self.R_grasp, p['app_up']), self.R_grasp, [q_pre_l] + seeds,
                                               prefer_first=p['ik_order'] == 'grasp_first')
        self.q_app_plan = np.concatenate([q_app_l, self.right_cmd])
        # ready point for the walk: app_up above the pre-grasp point, in the frame of the planned stance
        Rs_ = rotz(yaw)
        pr_ = self.p_pre + np.array([0, 0, p['app_up']])
        self.rel_ready = np.r_[(Rs_.T @ (pr_ - np.r_[self.stance, 0.0]))[:2], pr_[2]]
        tpos0, tR0 = self.tool_pose(ds)
        self.rel_tool0 = np.r_[(rotz(ctx.base_yaw).T @ (tpos0 - ctx.base_pos))[:2], tpos0[2]]
        self.R_tool0_rel = rotz(-ctx.base_yaw) @ tR0
        self.q_ik = self.q_out.copy()
        self.q_blend0 = self.q_out.copy()
        rr = p['raise_rel']
        praise = np.r_[ctx.base_pos[:2] + (rotz(ctx.base_yaw) @ np.array([rr[0], rr[1], 0.0]))[:2], rr[2]]
        pe4, ang4, q_raise_l, _ = self.global_ik(q_now, praise, rotz(ctx.base_yaw) @ self.R_grasp_rel,
                                                 [q_app_l] + seeds, prefer_first=p['ik_order'] == 'grasp_first')
        self.q_raise = np.concatenate([q_raise_l, self.right_cmd])
        t0 = ctx.t
        start = ctx.base_pos[:2].copy()
        self.arm_sched = None
        if p['approach_mode'] == 'auto':
            k = self.auto_approach_knots(t0, start, ctx.base_yaw, self.stance, yaw)
        elif p['approach_mode'] == 'auto2':
            k = self.auto2_approach_knots(t0, start, ctx.base_yaw, self.stance, yaw)
        else:
            k = [(t0, start[0], start[1], ctx.base_yaw)]
            for fr, x_, y_, yw_ in (p['approach_mid'] or []):
                k.append((t0 + p['t_approach'] * fr, x_, y_, math.radians(yw_)))
            k.append((t0 + p['t_approach'], self.stance[0], self.stance[1], yaw))
        self.t_app_total = k[-1][0] - t0
        self.app_t, xy, yw = spline_path(k)
        self.pf = E.PathFollower(self.app_t, xy, yw, lead_s=p['lead'], v_max=(p['v_max'], 0.3), wz_max=1.5,
                                 stop_tol=(p['stop_tol'], math.radians(5)), kp_pos=p['pf_kp'], ki_pos=p['pf_ki'])
        self.planned = True
        self.t_plan = ctx.t
        self.grasp_log = np.concatenate([self.p_grasp, self.R_grasp.ravel()])
        ctx.event('plan', 'after the target commit: pickup stance, approach path and direct body grasp planned around the '
                  f"SELECTED object's TRUE pose (privileged; scene grasp frame {p['grasp_site']}); IK with shoulder yaw, wrist roll and wrist yaw bounded to "
                  f"+/-{p['nat']:.2f} rad",
                  target=obj['id'], stance_xy=self.stance, stance_yaw_deg=p['stance_yaw_deg'], grasp_fingertip=self.p_grasp,
                  R_grasp=self.R_grasp, grasp_site=p['grasp_site'], pre_err_m=pe, pre_err_deg=ang, grasp_err_m=pe2,
                  grasp_err_deg=ang2, q_pre_left=q_pre_l, q_grasp_left=q_gr_l, approach_knots=np.array(k),
                  raise_point=praise, raise_err_m=pe4, q_raise_left=q_raise_l, ready_err_m=pe3, q_ready_left=q_app_l,
                  recipient_shoulder_l_f200=self.rcp_shoulder, recipient_hand_l_f200=self.rcp_hand)
        self.set_phase(ctx, 'approach')

    def auto_approach_knots(self, t0, start, yaw0, stance, yaw1):
        """v11 approach path from ANY measured pose at the commit: turn toward the stance while starting to walk,
        walk straight (heading = travel direction), turn to the stance heading while arriving.  Durations from the
        distance (approach_v) and the two turns (approach_wz), at least approach_min_T."""
        p = self.p
        d = stance - start
        dist = float(np.linalg.norm(d))
        if dist < 0.12:
            T = max(p['approach_min_T'], abs(wrap(yaw1 - yaw0)) / p['approach_wz'] + 0.6)
            return [(t0, start[0], start[1], yaw0), (t0 + T, stance[0], stance[1], yaw1)]
        yt = math.atan2(d[1], d[0])
        y_a = yaw0 + wrap(yt - yaw0)
        y_b = y_a + wrap(yaw1 - y_a)
        T1 = max(p['approach_turn_T'], abs(y_a - yaw0) / p['approach_wz'])
        T3 = max(p['approach_turn_T'], abs(y_b - y_a) / p['approach_wz'])
        T2 = max(0.6, 0.7 * dist / p['approach_v'])
        T = max(p['approach_min_T'], T1 + T2 + T3)
        s = T / (T1 + T2 + T3)
        T1, T2, T3 = T1 * s, T2 * s, T3 * s
        pa = start + 0.15 * d
        pb = start + 0.85 * d
        return [(t0, start[0], start[1], yaw0), (t0 + T1, pa[0], pa[1], y_a), (t0 + T1 + T2, pb[0], pb[1], y_a),
                (t0 + T1 + T2 + T3, stance[0], stance[1], y_b)]

    def auto2_approach_knots(self, t0, start, yaw0, stance, yaw1):
        """v11 approach in four parts from ANY measured pose at the commit: (1) turn toward a pre-stance point Q,
        approach_back m behind the pickup stance (along the stance heading), (2) walk straight to Q, (3) turn in place at
        Q to face the bag, (4) step forward to the stance.  The arm stays in its standing posture during (1)-(2), is
        raised in front of the body during (3) and brought over the bag during (4) (self.arm_sched)."""
        p = self.p
        fwd1 = np.array([math.cos(yaw1), math.sin(yaw1)])
        Q = stance - p['approach_back'] * fwd1
        d = Q - start
        dist = float(np.linalg.norm(d))
        yt = math.atan2(d[1], d[0]) if dist > 0.05 else yaw0
        y_a = yaw0 + wrap(yt - yaw0)
        y_b = y_a + wrap(yaw1 - y_a)
        T1 = max(p['approach_turn_T'], abs(y_a - yaw0) / p['approach_wz']) if dist > 0.05 else 0.0
        T2 = max(0.6, dist / p['approach_v']) if dist > 0.05 else 0.0
        T3 = max(p['approach_turn_T'], abs(y_b - y_a) / p['approach_wz'])
        T4 = max(p['approach_step_T'], p['approach_back'] / (0.6 * p['approach_v']))
        k = [(t0, start[0], start[1], yaw0)]
        if dist > 0.05:
            pa = start + min(0.10, 0.3 * dist) * d / dist
            k += [(t0 + T1, pa[0], pa[1], y_a), (t0 + T1 + T2, Q[0], Q[1], y_a)]
        k += [(t0 + T1 + T2 + T3, Q[0], Q[1], y_b), (t0 + T1 + T2 + T3 + T4, stance[0], stance[1], y_b)]
        if p['arm_ready_at_Q']:
            # v11: raise and reach the ready pose (high above the bag) while turning at Q; the step forward to the
            # stance is made with the arm still at the ready pose
            self.arm_sched = (T1 + T2, 0.5 * T3, T3)
        else:
            self.arm_sched = (T1 + T2, T3, T3 + T4)
        return k

    def replan_grasp(self, ctx, ds):
        qs = ctx.qpos.copy()
        self.p_grasp, self.R_grasp = self.grasp_target(ds)
        gf = self.p['ik_order'] == 'grasp_first'
        seeds = ([self.q_gr_plan] if gf else []) + [ctx.q29[15:22].copy(), self.q_pre_plan[:7]] + self.seeds()
        pe, ang, ql, Rs = self.global_ik(qs, self.p_grasp, self.R_grasp, seeds, prefer_first=gf)
        self.p_pre = self.p_grasp + self.back(self.R_grasp, self.p['pre_up'])
        pe_p, ang_p, qlp, _ = self.global_ik(qs, self.p_pre, self.R_grasp, ([ql] if gf else []) + [ctx.q29[15:22].copy(), ql] + self.seeds(),
                                             prefer_first=gf)
        self.R_grasp_rel = rotz(-ctx.base_yaw) @ self.R_grasp
        self.replanned = True
        if self.p['ik_w_post'] > 0:
            self.q_nom_cur = np.asarray(ql, float).copy()
        tpos, tR = self.tool_pose(ds)
        self.pre_from = (tpos.copy(), tR.copy(), ctx.t)
        dc = self.descent_clearance(qs, qlp) if self.p['descent_check'] else []
        ctx.event('grasp_replanned', 'grasp re-solved from the measured pelvis pose and the TRUE bag grasp frame '
                  '(privileged); the pre-grasp motion restarts from the current tool pose; planned straight descent '
                  'checked for UMI-to-bag clearance (fingers open)',
                  pelvis=ctx.base_pos, yaw_deg=math.degrees(ctx.base_yaw), pos_err_m=pe, ang_err_deg=ang, q_left=ql,
                  pre_pos_err_m=pe_p, pre_ang_err_deg=ang_p, q_pre_left=qlp,
                  descent_min_clearance_m=min((c[1] for c in dc), default=float('nan')),
                  descent_clearance=np.array(dc) if dc else np.zeros((0, 4)),
                  grasp_fingertip=self.p_grasp, stance_err_m=float(np.linalg.norm(ctx.base_pos[:2] - self.stance)))

    def set_phase(self, ctx, name):
        self.ph = name
        self.ph_t0 = ctx.t

    def present_point(self):
        p = self.p
        nf = math.radians(p['nfar_yaw_deg'])
        n_far = np.array([math.cos(nf), math.sin(nf), 0.0])
        w = np.cross(n_far, [0, 0, 1.0])
        anchor = self.rcp_shoulder
        if p['present_anchor'] == 'chest':
            rec = self.rcp.rec
            anchor = rec.P[200, rec.idx['spine_03']].copy()
        L = anchor + p['present_dist'] * n_far + p['present_lat'] * w
        L[2] = p['present_z']
        return L, n_far

    def build_carry_path(self, ctx):
        p = self.p
        t = ctx.t
        self.t_carry0 = t
        ds = ctx.scratch()
        L, n_far = self.present_point()
        self.seat_des = L
        self.n_far_des = n_far
        tpos, tR = self.tool_pose(ds)
        self.site_top = self.m.site(p['present_site']).id
        self.off_tool = tR.T @ (tpos - ds.site_xpos[self.site_top])
        pr = np.array(p['present_rel'])
        if p['carry_arm_frozen']:
            # the arm joint targets stay frozen for the whole carry: the bag stays where it is relative to the body
            pr = (rotz(ctx.base_yaw).T @ (tpos - ctx.base_pos))[:2]
            self.q_freeze = self.q_out.copy()
        P0 = ctx.base_pos[:2].copy()
        y0 = ctx.base_yaw
        P1 = P0 + np.array(p['backup'])
        P2 = P1 + np.array(p['turn_end_d'])
        P3 = P2 + np.array(p['walk_d'])

        def stance_for(yaw):
            tool_des_ = L + rotz(yaw) @ self.R_grasp_rel @ self.off_tool
            return tool_des_, (tool_des_ - rotz(yaw) @ np.array([pr[0], pr[1], 0.0]))[:2]
        y2 = math.radians(-90.0)
        if p['heading_mode'] == 'travel':
            # face the direction of the last (straight) walk segment: the GR00T walk tracks forward motion well but
            # lateral offsets poorly
            for _ in range(20):
                _, H_ = stance_for(y2)
                dv = H_ - P3
                y2 = math.atan2(dv[1], dv[0])
        if p['carry_mode'] == 'straight':
            # one straight walk from the pickup stance to the presentation stance, facing the walking direction
            for _ in range(20):
                _, H_ = stance_for(y2)
                dv = H_ - P0
                y2 = math.atan2(dv[1], dv[0])
        if p['carry_mode'] == 'turn_walk':
            # v11: end FACING the person (his recorded bone face_bone at frame 200, ideal perception of the person)
            rec = self.rcp.rec
            face = rec.P[200, rec.idx[p['face_bone']]][:2].copy()
            for _ in range(30):
                _, H_ = stance_for(y2)
                dv = face - H_
                y2 = math.atan2(dv[1], dv[0])
            self.face_pt = face
        self.hand_yaw = y2
        tool_des, H = stance_for(y2)
        self.tool_des_z = float(tool_des[2])
        if p['carry_arm_frozen']:
            self.tool_des_z = float(tpos[2])
        self.hand_stance = H
        t1 = t + p['t_back']
        t2 = t1 + p['t_turn']
        t3 = t2 + p['t_walk1']
        t4 = t3 + p['t_walk']
        self.t_back_end, self.t_turn_end = t1, t2
        if p['carry_mode'] == 'turn_walk':
            # v11: turn in place toward the presentation stance while stepping back carry_back from the table, walk
            # straight (heading = walking direction), then turn to face the person while arriving
            Pb = P0 - p['carry_back'] * np.array([math.cos(y0), math.sin(y0)])
            dv = H - Pb
            y_w = y0 + wrap(math.atan2(dv[1], dv[0]) - y0)
            y_e = y_w + wrap(y2 - y_w)
            t1 = t + p['t_turn_carry']
            dist = float(np.linalg.norm(dv))
            t4 = t1 + max(1.2, dist / p['v_carry']) + p['t_final']
            self.t_back_end, self.t_turn_end = t1, t1
            fwd_e = np.array([math.cos(y2), math.sin(y2)])
            Hf = H - p['d_final'] * fwd_e
            k = [(t, P0[0], P0[1], y0), (t1, Pb[0], Pb[1], y_w), (t4 - p['t_final'], Hf[0], Hf[1], y_w + 0.5 * (y_e - y_w)),
                 (t4, H[0], H[1], y_e)]
            if p['carry_back_first'] > 0:
                # v11: step straight back from the table first (heading kept), then turn in place: the bag held in
                # front does not sweep over the tabletop objects while the robot turns
                tb = p['carry_back_first']
                k = [(t, P0[0], P0[1], y0), (t + tb, Pb[0], Pb[1], y0)] + [(tk + tb, xk, yk, ywk) for tk, xk, yk, ywk in k[1:]]
            self.carry_t, xy, yw = spline_path(k)
            self.pf = E.PathFollower(self.carry_t, xy, yw, lead_s=p['lead'], v_max=(p['v_max_carry'], 0.3),
                                     wz_max=p['wz_max_carry'], stop_tol=(p['stop_tol_carry'], math.radians(6)),
                                     accel=tuple(p['carry_accel']))
            ctx.event('carry_path', 'carry path planned from the measured pelvis pose after the stow: ' + p['reason_carry_path'] +
                      ' (PI path follower, simulator pelvis pose); the robot ends facing the person, the bag held at the '
                      'frozen carry pose lands at the presentation point',
                      knots=np.array(k), present_stance=H, heading_deg=math.degrees(y2), top_des=L, n_far=n_far,
                      tool_des=tool_des, off_tool=self.off_tool, face_point=self.face_pt,
                      shoulder_to_top_m=float(np.linalg.norm(L - self.rcp_shoulder)),
                      hand_to_top_m=float(np.linalg.norm(L - self.rcp_hand)))
            return
        if p['carry_mode'] == 'straight':
            # one clamped spline from the pickup stance to the presentation stance (the heading turns toward the
            # walking direction along the way); 'back' / 'turn' / 'carry' are phase labels only
            t1 = t + p['t_back']
            t2 = t1 + p['t_turn']
            t4 = t + p['t_walk']
            self.t_back_end, self.t_turn_end = t1, t2
            k = [(t, P0[0], P0[1], y0)]
        else:
            k = [(t, P0[0], P0[1], y0), (t1, P1[0], P1[1], math.radians(p['back_yaw_deg'])),
                 (t2, P2[0], P2[1], math.radians(-90.0))]
        if p['t_walk1'] > 0 and p['carry_mode'] != 'straight':
            k += [(t3, P3[0], P3[1], math.radians(-90.0))]
        if p['d_final'] > 0:
            fwd2 = np.array([math.cos(y2), math.sin(y2)])
            Hf = H - p['d_final'] * fwd2
            k += [(t4 - p['t_final'], Hf[0], Hf[1], y2)]
        k += [(t4, H[0], H[1], y2)]
        self.carry_t, xy, yw = spline_path(k)
        self.pf = E.PathFollower(self.carry_t, xy, yw, lead_s=p['lead'], v_max=(p['v_max_carry'], 0.3),
                                 wz_max=p['wz_max_carry'], stop_tol=(p['stop_tol_carry'], math.radians(6)),
                                 accel=tuple(p['carry_accel']))
        ctx.event('carry_path', 'carry path planned from the measured pelvis pose after the lift: ' + p['reason_carry_path'] +
                  ' (PI path follower, simulator pelvis pose); the presentation point is in front of the person\'s idle '
                  'left shoulder / hand',
                  knots=np.array(k), present_stance=H, heading_deg=math.degrees(y2), top_des=L, n_far=n_far,
                  tool_des=tool_des, off_tool=self.off_tool,
                  shoulder_to_top_m=float(np.linalg.norm(L - self.rcp_shoulder)),
                  hand_to_top_m=float(np.linalg.norm(L - self.rcp_hand)))

    # ------------------------------------------------------------------------------------------ phases
    def detect_drop(self, ctx, ds, tpos, bag_z):
        p = self.p
        if not self.lifted or self.dropped:
            return
        fq = self.finger_q(ctx)
        why = None
        if self.fq_pinch is not None and fq > self.fq_pinch + p['drop_close']:
            why = ('finger_closure', ['umi_finger_position'], False)
        elif bag_z < p['drop_z']:
            why = ('bag_height', ['true_bag_height'], True)
        if why is None:
            return
        self.dropped = True
        ctx.event('drop_detected', f'drop detected ({why[0]}) after the lift: the sequence stops, the robot stands still '
                  'with its arm where it is', signals=why[1], privileged=why[2], finger_q_m=fq,
                  finger_q_pinch_m=self.fq_pinch, bag_z_m=bag_z, phase_at_drop=self.ph)
        self.set_phase(ctx, 'stand')
        self.q_stand = self.q_out.copy()
        self.stand_tgt = tpos.copy()
        self.t_done = ctx.t + p['t_after_drop']

    def policy_step(self, ctx):
        p = self.p
        t = ctx.t
        ds = ctx.scratch()
        tpos, tR = self.tool_pose(ds)
        reason = ''
        bh = p['base_h'] if (p['base_h_carry'] is None or self.t_carry0 is None) else p['base_h_carry']
        arm_target = None
        self.load, self.pads = self.robot_finger_load(ds)
        bag_b = self.m.body('backpack').id
        bag_z = float(ds.xpos[bag_b][2])
        # ---------------- perception (every tick, before any decision)
        for name, why, data in self.perc.step(t, ds, ctx.base_pos[:2], ctx.base_yaw):
            ctx.event(name, why, to_phase=name, **data)       # to_phase: the trace shows perception events
        self.detect_drop(ctx, ds, tpos, bag_z)
        if self.ph == 'watch':
            self.v = np.zeros(3)
            self.q_arm_cmd = self.arm_key.copy()
            reason = ('balance in place and watch the person: no motion until a pointed-at object is committed '
                      '(perception every tick)')
            if p['watch_pitch'] != 0.0:
                reason = ('balance in place and watch the person, the torso leaned back a little so that the chest '
                          'camera looks up toward him: no walking until a pointed-at object is committed (perception '
                          'every tick)')
            if self.perc.target >= 0:
                self.plan_after_commit(ctx, ds)
        if self.ph == 'approach':
            # two-stage joint-space arm motion (planned at the commit): raise the open gripper beside the body first
            # (q_raise, clear of the bag), then bring it over the bag top to app_up above the pre-grasp point (q_ready)
            a_delay, a_raise, a_blend = self.arm_sched or (p['t_arm_delay'], p['t_raise'], p['t_arm_blend'])
            tt = max(0.0, t - self.t_plan - a_delay)
            if p['raise_mid'] is not None and tt < a_raise:
                # v11: swing the arm forward first (raise_mid: shoulder pitch forward, elbow bent, upper arm away from
                # the torso), then turn the hand into the grasp orientation: no joint-space path through the own body
                q_mid = self.q_blend0.copy()
                q_mid[:7] = np.asarray(p['raise_mid'], float)
                if tt < 0.5 * a_raise:
                    u = min_jerk(tt / (0.5 * a_raise))
                    self.q_arm_cmd = (1 - u) * self.q_blend0 + u * q_mid
                else:
                    u = min_jerk((tt - 0.5 * a_raise) / (0.5 * a_raise))
                    self.q_arm_cmd = (1 - u) * q_mid + u * self.q_raise
            elif tt < a_raise:
                u = min_jerk(tt / a_raise)
                self.q_arm_cmd = (1 - u) * self.q_blend0 + u * self.q_raise
            else:
                u = min_jerk((tt - a_raise) / (a_blend - a_raise))
                self.q_arm_cmd = (1 - u) * self.q_raise + u * self.q_app_plan
            w = tt / a_blend
            self.v = np.array(self.pf.command(t, ctx.base_pos, ctx.base_yaw)[:3])
            reason = 'walk the approach path to the pickup stance (PI path follower); open gripper raised beside the body'
            if t >= self.app_t[-1] - 0.25 and w >= 1.0:
                self.set_phase(ctx, 'pre_grasp')
                self.q_ik = self.q_out.copy()
                self.pre_from = (tpos.copy(), tR.copy(), t)
        if self.ph == 'pre_grasp' and not self.replanned:
            # the arm waits at the ready posture until the base has stopped (or t_stop_max after the path end); then the
            # grasp is re-solved from the measured pelvis pose and the walk command stays zero
            self.v = np.array(self.pf.command(t, ctx.base_pos, ctx.base_yaw)[:3])
            self.q_arm_cmd = self.q_app_plan.copy()
            reason = 'settle at the pickup stance (PI path follower), open gripper held above the bag'
            if not any(self.v) or t > self.app_t[-1] + p['t_stop_max']:
                self.v = np.zeros(3)
                self.replan_grasp(ctx, ds)
                self.q_ik = self.q_out.copy()
        elif self.ph == 'pre_grasp':
            self.v = np.zeros(3)
            p0, R0, t0 = self.pre_from
            u = min_jerk((t - t0) / p['t_pre'])
            tgt = (1 - u) * p0 + u * self.p_pre
            Rt = slerp_R(R0, self.R_grasp, u)
            self.q_arm_cmd = self.solve_ik(ctx, self.corrected(tgt, tpos) if u >= 1 else tgt, R=Rt, wr=p['wr_grasp'],
                                           allow_global=u >= 1)
            arm_target = (tgt, Rt)
            reason = 'open gripper to the pre-grasp point above the bag top (online IK from the measured pose)'
            err_now = float(np.linalg.norm(tpos - self.p_pre))
            if t - t0 >= p['t_pre'] and (err_now < p['pre_tol'] or t - t0 >= p['t_pre'] + p['t_pre_max']):
                ctx.event('pre_grasp_reached', 'measured tool error at the pre-grasp point', err_m=err_now)
                self.set_phase(ctx, 'descend')
        v10d = p['desc_mode'] == 'v10'
        if self.ph == 'descend' and not self.inserting:
            self.v = np.zeros(3)
            u = min_jerk((t - self.ph_t0) / p['t_desc'])
            p_end = self.p_grasp + self.back(self.R_grasp, p['desc_hold']) if v10d else self.p_grasp
            tgt = (1 - u) * self.p_pre + u * p_end
            self.q_arm_cmd = self.solve_ik(ctx, self.corrected(tgt, tpos, freeze=v10d), R=self.R_grasp, wr=p['wr_grasp'])
            arm_target = (tgt, self.R_grasp)
            reason = 'straight descent: the wide-open fingers come down on both sides of the bag top'
            if t - self.ph_t0 >= p['t_desc']:
                self.set_phase(ctx, 'align')
        if self.ph == 'align' and v10d:
            self.v = np.zeros(3)
            tg_, Rg = self.grasp_target(ds)               # TRUE grasp frame now (privileged)
            tgt = tg_ + self.back(Rg, p['desc_hold'])
            self.q_arm_cmd = self.solve_ik(ctx, self.corrected_xy(tgt, tpos, Rg), R=Rg, wr=p['wr_grasp'])
            arm_target = (tgt, Rg)
            reason = ("centre the open fingers on the bag's centre-of-mass plane a few centimetres above the grasp depth "
                      '(closing and pad-width axes)')
            e = tR.T @ (tpos - tgt)
            if (abs(e[1]) < p['align_tol'] and abs(e[0]) < p['align_tol_x'] and abs(e[2]) < p['align_tol_z'] and
                    t - self.ph_t0 >= 0.06) or t - self.ph_t0 >= p['t_align_max']:
                ctx.event('aligned_above', 'measured fingertip-centre offset from the hover point above the grasp '
                          '(x closing axis, y pad width, z approach)', offset_m=e)
                self.R_grasp = Rg.copy()
                self.inserting = True                     # second 'descend' segment: the slow final insertion
                self.set_phase(ctx, 'descend')
        elif self.ph == 'descend' and self.inserting:
            self.v = np.zeros(3)
            tg_, Rg = self.grasp_target(ds)
            u = min_jerk((t - self.ph_t0) / p['t_insert'])
            tgt = tg_ + self.back(Rg, p['desc_hold'] * (1 - u))
            self.q_arm_cmd = self.solve_ik(ctx, self.corrected_xy(tgt, tpos, Rg), R=Rg, wr=p['wr_grasp'])
            arm_target = (tgt, Rg)
            reason = 'slow final descent to the grasp depth (no integral along the approach axis)'
            if t - self.ph_t0 >= p['t_insert']:
                e = tR.T @ (tpos - tg_)
                ctx.event('grasp_pose_reached', 'measured fingertip-centre offset from the grasp frame before '
                          'closing (x closing axis, y pad width = offset from the bag centre-of-mass plane, z approach)',
                          offset_m=e, tool=tpos, grasp_fingertip=tg_)
                self.p_grasp_c = tg_.copy()
                self.R_grasp = Rg.copy()
                self.R_grasp_rel = rotz(-ctx.base_yaw) @ self.R_grasp
                self.set_phase(ctx, 'close')
        if self.ph == 'align' and not v10d:
            self.v = np.zeros(3)
            tgt, Rg = self.grasp_target(ds)               # TRUE grasp frame now (privileged)
            self.q_arm_cmd = self.solve_ik(ctx, self.corrected(tgt, tpos), R=Rg, wr=p['wr_grasp'])
            arm_target = (tgt, Rg)
            reason = "centre the pads on the bag's centre-of-mass plane before closing"
            e = tR.T @ (tpos - tgt)                         # x closing, y pad width, z approach
            if (abs(e[1]) < p['align_tol'] and abs(e[0]) < 0.004 and abs(e[2]) < 0.004 and t - self.ph_t0 >= 0.06) or \
                    t - self.ph_t0 >= p['t_align_max']:
                ctx.event('grasp_pose_reached', 'aligned; measured fingertip-centre offset from the grasp frame before '
                          'closing (x closing axis, y pad width = offset from the bag centre-of-mass plane, z approach)',
                          offset_m=e, tool=tpos, grasp_fingertip=tgt)
                self.p_grasp_c = tgt.copy()
                self.R_grasp = Rg.copy()
                self.R_grasp_rel = rotz(-ctx.base_yaw) @ self.R_grasp
                self.set_phase(ctx, 'close')
        if self.ph == 'close':
            self.v = np.zeros(3)
            u = min_jerk((t - self.ph_t0) / (p['t_close'] * 0.6))
            self.fing = p['finger_open'] + (p['finger_close'] - p['finger_open']) * u
            self.q_arm_cmd = self.solve_ik(ctx, self.p_grasp_c + self.e_int, R=self.R_grasp, wr=p['wr_grasp'])
            arm_target = (self.p_grasp_c, self.R_grasp)
            reason = 'fingers close on the bag top (force-limited servos, 35 N per finger)'
            if t - self.ph_t0 >= p['t_close']:
                ctx.event('fingers_closed', 'finger servo targets at the close value', fingers_q=ctx.fingers_q,
                          pads_N=np.array(self.pads))
                if p['finger_hold_mode'] == 'contact':
                    # robot-side proprioception: each finger's measured joint position at contact; from here the targets
                    # move to finger_squeeze_m inside it, so neither force-limited servo saturates and the pair centres
                    # the bag along the closing axis (saturated servos leave it free to slide between the stops)
                    self.q_contact = np.array(ctx.fingers_q[:2], float)
                    self.t_fhold0 = t
                    ctx.event('finger_hold', 'finger targets set from the measured finger positions at contact: '
                              f"{p['finger_squeeze_m'] * 1000:.1f} mm inside each (about "
                              f"{p['finger_squeeze_m'] * 5000:.0f} N each with the 5000 N/m servo), ramped over "
                              f"{p['t_finger_ramp']:.2f} s", signals=['umi_finger_position'],
                              finger_q_contact_m=self.q_contact, finger_targets_m=self.q_contact + p['finger_squeeze_m'])
                Rb = rotz(ctx.base_yaw)
                self.rel_grasp = Rb.T @ (self.p_grasp_c - ctx.base_pos)
                self.rel_grasp[2] = self.p_grasp_c[2]
                if p['ik_w_post'] > 0:
                    # v11: posture at the top of the lift (global IK seeded from the current grasp posture), used as
                    # the DLS nominal posture during the lift and hold
                    top = self.rel_target(ctx, self.rel_grasp + np.array([0, 0, p['lift']]))
                    pe_l, ang_l, q_lift, _ = self.global_ik(ctx.qpos, top,
                                                            self.R_rel(ctx), [ctx.q29[15:22].copy(), self.q_nom_cur] + self.seeds(),
                                                            prefer_first=True)
                    self.q_nom_cur = q_lift.copy()
                    ctx.event('lift_planned', 'arm posture at the top of the lift (IK from the measured pose), the nominal '
                              'posture of the lift', pos_err_m=pe_l, ang_err_deg=ang_l, q_lift_left=q_lift)
                if p['lift_mode'] == 'joint_path':
                    self.plan_lift_path(ctx)
                self.set_phase(ctx, 'lift')
        if self.ph in ('back', 'turn', 'carry'):
            self.v = np.array(self.pf.command(t, ctx.base_pos, ctx.base_yaw)[:3])
        elif self.ph in ('lift', 'hold', 'present', 'present_hold', 'stand'):
            self.v = np.zeros(3)
        if self.ph == 'lift':
            self.fing = p['finger_close']
            u = min_jerk((t - self.ph_t0) / p['t_lift'])
            rel = self.rel_grasp + np.array([0, 0, p['lift'] * u])
            tgt = self.rel_target(ctx, rel)
            if p['lift_mode'] == 'joint_path':
                # v11: follow the joint path planned at the close (IK continuation), no online IK during the lift
                if p['lift_profile'] == 'easeout':
                    # quick lift-off (the bag leaves the table at once instead of bouncing on it while the load moves
                    # to the pads), then slowing to rest at the top
                    w_ = min(1.0, max(0.0, (t - self.ph_t0) / p['t_lift']))
                    u = 1.0 - (1.0 - w_) ** 2
                x = u * (len(self.lift_path) - 1)
                i0 = min(int(x), len(self.lift_path) - 2)
                a_ = x - i0
                self.q_arm_cmd = self.q_out.copy()
                self.q_arm_cmd[:7] = (1 - a_) * self.lift_path[i0] + a_ * self.lift_path[i0 + 1]
                self.q_arm_cmd[7:] = self.right_cmd
            else:
                self.q_arm_cmd = self.carry_ik(ctx, tgt, tpos, p['wr_carry'])
            arm_target = (tgt, None)
            reason = 'lift the bag straight up (robot standing)'
            if t - self.ph_t0 >= p['t_lift']:
                self.rel_a = rel.copy()
                self.lifted = True
                self.fq_pinch = self.finger_q(ctx)
                self.swing0 = (ds.xipos[self.bag_id] - tpos)[:2].copy()
                ctx.event('lifted', 'lift done; grasp finger position recorded for the robot-side drop check',
                          signals=['umi_finger_position'], finger_q_pinch_m=self.fq_pinch, bag_z_m=bag_z,
                          pads_N=np.array(self.pads))
                self.set_phase(ctx, 'hold')
                self.t_hold0 = t
        if self.ph == 'hold':
            self.fing = p['finger_close']
            tgt = self.damped(ctx, self.rel_target(ctx, self.rel_a), ds, tpos)
            if p['lift_mode'] == 'joint_path':
                self.q_arm_cmd = self.q_out.copy()
                self.q_arm_cmd[:7] = self.lift_path[-1]
                self.q_arm_cmd[7:] = self.right_cmd
            else:
                self.q_arm_cmd = self.carry_ik(ctx, tgt, tpos, p['wr_carry'])
            arm_target = (tgt, None)
            reason = 'hold the lifted bag still before walking'
            if t - self.ph_t0 >= p['t_hold']:
                self.slip_comp(ds, tpos)
                if p['stow_rel'] is not None:
                    # v11: bring the bag to the carry pose in front of the body first (standing, online IK)
                    self.t_stow0 = t
                    self.set_phase(ctx, 'stow')
                else:
                    self.build_carry_path(ctx)
                    # straight mode: one walk, labelled 'carry' from the start (no step-back or turn segments)
                    self.set_phase(ctx, 'carry' if p['carry_mode'] in ('straight', 'turn_walk') else 'back')
        if self.ph == 'stow':
            self.v = np.zeros(3)
            self.fing = p['finger_close']
            u = min_jerk((t - self.ph_t0) / p['t_stow'])
            rel = (1 - u) * self.rel_a + u * np.array(p['stow_rel'], float)
            tgt = self.rel_target(ctx, rel)
            self.q_arm_cmd = self.carry_ik(ctx, tgt, tpos, p['wr_carry'])
            arm_target = (tgt, None)
            reason = p['reason_stow']
            if t - self.ph_t0 >= p['t_stow'] + p['stow_hold']:
                ctx.event('stowed', 'the bag is at the carry pose in front of the body; the left-arm joint targets are '
                          'frozen from here (no IK while walking)', tool=tpos, stow_rel=np.array(p['stow_rel'], float),
                          tool_rel=np.r_[(rotz(ctx.base_yaw).T @ (tpos - ctx.base_pos))[:2], tpos[2]],
                          pads_N=np.array(self.pads))
                self.build_carry_path(ctx)
                self.set_phase(ctx, 'carry')
        if self.ph in ('back', 'turn', 'carry'):
            self.fing = p['finger_close']
            self.update_frame(ctx)
            self.use_fbase = p['carry_frame_tau'] > 0
            back_rel = np.array(p['back_rel'])
            if self.ph == 'back':
                u = p['back_u'] * min_jerk((t - self.ph_t0) / p['t_back'])
                if t >= self.t_back_end:
                    self.set_phase(ctx, 'turn')
                    self.u_turn0 = u
                reason = p['reason_back']
            elif self.ph == 'turn':
                u = self.u_turn0 + (1 - self.u_turn0) * min_jerk((t - self.ph_t0) / p['t_turn'])
                if t >= self.t_turn_end:
                    self.set_phase(ctx, 'carry')
                reason = p['reason_turn']
            else:
                u = 1.0
                reason = p['reason_carry']
            cr = np.array(p['carry_rel'])
            if self.ph in ('back', 'turn'):
                mid_rel = np.array([back_rel[0], back_rel[1], cr[2] if p['back_z'] is None else p['back_z']])
                rel = (1 - u) * self.rel_a + u * mid_rel if self.ph == 'back' else \
                    (1 - u) * self.rel_a + u * (mid_rel + (cr - mid_rel) * min_jerk((t - self.ph_t0) / p['t_turn']))
            else:
                rel = cr
            tgt = self.damped(ctx, self.rel_target(ctx, rel), ds, tpos) + self.slip_comp(ds, tpos)
            if self.ph == 'carry':
                tl = self.carry_t[-1]
                v = min_jerk((t - (tl - p['t_present'])) / p['t_present'])
                if v > 0:
                    pr = np.array([p['present_rel'][0], p['present_rel'][1], self.tool_des_z])
                    tgt = self.damped(ctx, self.rel_target(ctx, (1 - v) * rel + v * pr), ds, tpos) + self.slip_comp(ds, tpos)
            self.last_carry_tgt = tgt.copy()
            if p['carry_arm_frozen']:
                self.q_arm_cmd = self.q_freeze.copy()
                reason = reason + '; arm joint targets frozen at the lifted posture (no IK while walking)'
            else:
                self.q_arm_cmd = self.carry_ik(ctx, tgt, tpos, p['wr_carry'])
            arm_target = (tgt, None)
            if self.ph == 'carry' and ((t >= self.carry_t[-1] and not any(self.v)) or
                                       t >= self.carry_t[-1] + p['t_carry_extra']):
                self.set_phase(ctx, 'present')
                self.use_fbase = False
                self.q_hold = self.q_out.copy()
                self.q_ik = self.q_out.copy()
                self.hold_tgt = self.last_carry_tgt.copy()
        if self.ph == 'present':
            self.fing = p['finger_close']
            self.q_arm_cmd = self.q_hold.copy()
            arm_target = (self.hold_tgt, None)
            reason = ("stopped with the bag held out in front of the person's left hand: arm joint targets frozen (no IK), "
                      'Balance model stands')
            if t - self.ph_t0 >= p['present_settle']:
                top = ds.site_xpos[self.site_top].copy()
                rp = self.rcp.rec
                Pn, _ = rp.at(self.rcp.tm(t))
                hand_now = Pn[rp.idx['hand_l']]
                self.set_phase(ctx, 'present_hold')
                self.t_done = t + p['t_present_hold']
                ctx.event('bag_presented', 'the robot holds the bag out in front of the person and keeps holding it (arm '
                          'joint targets frozen, fingers closed, Balance model stands); no handover follows',
                          bag_top=top, bag_top_des=self.seat_des, bag_top_err_m=float(np.linalg.norm(top - self.seat_des)),
                          hand_l_now=hand_now, bag_top_to_hand_l_m=float(np.linalg.norm(top - hand_now)),
                          robot_finger_load_N=self.load, pads_N=np.array(self.pads), pelvis=ctx.base_pos,
                          pelvis_yaw_deg=math.degrees(ctx.base_yaw),
                          stance_err_m=float(np.linalg.norm(ctx.base_pos[:2] - self.hand_stance)))
        if self.ph == 'present_hold':
            self.fing = p['finger_close']
            self.q_arm_cmd = self.q_hold.copy()
            arm_target = (self.hold_tgt, None)
            reason = 'keep holding the bag out, standing still (Balance), fingers closed'
        if self.ph == 'stand':
            self.v = np.zeros(3)
            self.q_arm_cmd = self.q_stand.copy()
            arm_target = (self.stand_tgt, None)
            reason = 'stand still (Balance)'
        # ---------------- arm joint-target rate limits (G9): right arm ramps to its relaxed posture
        self.right_cmd = self.right_cmd + np.clip(self.right_target - self.right_cmd, -p['dq_max_right'], p['dq_max_right'])
        qa = self.q_arm_cmd.copy()
        qa[7:] = self.right_cmd
        dq = qa - self.q_out
        dq_l = p['dq_max_hold'] if self.ph == 'present_hold' else p['dq_max_left']
        if p['dq_max_lift'] is not None and self.ph in ('lift', 'hold'):
            dq_l = p['dq_max_lift']
        if p['dq_max_carry'] is not None and self.ph in ('back', 'turn', 'carry'):
            dq_l = p['dq_max_carry']
        dq[:7] = np.clip(dq[:7], -dq_l, dq_l)
        dq[7:] = np.clip(dq[7:], -p['dq_max_right'], p['dq_max_right'])
        self.q_out = self.q_out + dq
        tr = np.array(p['torso_rpy_grasp'], float)
        t_out0 = {'hold': self.t_hold0, 'stow': self.t_stow0}.get(p['torso_out_at'], self.t_carry0)
        if self.planned and np.any(tr):
            if t_out0 is None:
                u_t = min_jerk((t - self.t_plan) / p['t_torso_in'])
            else:
                u_t = 1.0 - min_jerk((t - t_out0) / p['t_torso_out'])
            self.torso = tuple(float(x) for x in tr * u_t)
        if p['watch_pitch'] != 0.0:
            # egocam watch lean: GR00T torso pitch command while watching, removed with a min-jerk ramp from the plan
            if not self.planned:
                self.lean_at_commit = p['watch_pitch'] * min_jerk(t / p['watch_ramp'])
                lean = self.lean_at_commit
                base_t = (0.0, 0.0, 0.0)
            else:
                lean = self.lean_at_commit * (1.0 - min_jerk((t - self.t_plan) / p['unlean_T']))
                base_t = tuple(float(x) for x in tr * u_t) if np.any(tr) else (0.0, 0.0, 0.0)
            self.torso = (base_t[0], base_t[1] + lean, base_t[2])
        q15 = self.loco.step(ctx, *self.v, base_height=bh, torso_rpy=self.torso)
        ks = p['arm_kp_scale']
        if p['arm_kp_scale_carry'] is not None and self.t_carry0 is not None:
            # softer LEFT arm from the start of the carry walk (ramped over 0.5 s): the arm filters the body's step jolts
            u_k = min_jerk((t - self.t_carry0) / 0.5)
            ks_l = ks + (p['arm_kp_scale_carry'] - ks) * u_k
            sc = np.concatenate([np.full(7, ks_l), np.full(7, ks)])
        else:
            sc = np.full(14, ks)
        q29, kp, kd, g = E.compose_gr00t(q15, self.q_out, arm_kp=E.ARM_KP_14 * sc, arm_kd=E.ARM_KD_14 * np.sqrt(sc))
        ik_t = np.full(14, np.nan)
        if arm_target is not None:
            ik_t[:3] = arm_target[0]
        fingers = np.array([self.fing, self.fing, p['finger_open'], p['finger_open']])
        if self.q_contact is not None and self.ph != 'stand':
            u_f = min_jerk((t - self.t_fhold0) / p['t_finger_ramp'])
            fingers[:2] = (1 - u_f) * p['finger_close'] + u_f * (self.q_contact + p['finger_squeeze_m'])
        done = self.ph in ('present_hold', 'stand') and t >= getattr(self, 't_done', 1e9)
        pc = self.perc.last
        gsite = ds.site_xpos[self.m.site(p['grasp_site']).id].copy()
        return E.ControlOutput(q_target=q29, kp=kp, kd=kd, gravcomp=g, fingers=fingers, cmd=self.loco.cmd,
                               policy_obs=self.loco.obs, policy_action=self.loco.action, phase=self.ph,
                               phase_reason=reason, done=done, arm_ik_targets=ik_t,
                               log={'tool_pos': tpos, 'ik_err': np.array(self.ik_err, float), 'e_int': self.e_int,
                                    'damp': self.damp, 'tool_R': tR.ravel(), 'slip': self.slip,
                                    'robot_finger_load': self.load, 'robot_pads': np.array(self.pads),
                                    'finger_q': self.finger_q(ctx), 'bag_z': bag_z, 'bag_com': ds.xipos[bag_b].copy(),
                                    'bag_quat': ds.xquat[bag_b].copy(), 'grasp_site_true': gsite,
                                    'grasp_plan': self.grasp_log,
                                    'perc_pointing': pc['pointing'], 'perc_arm_extended': pc['arm_extended'],
                                    'perc_ray': pc['ray'], 'perc_angles': pc['angles'], 'perc_target': pc['target'],
                                    'perc_confidence': pc['confidence'], 'perc_keypoints': pc['keypoints'],
                                    'perc_centres': pc['centres'], 'perc_held': pc['held'],
                                    **(self.perc.log() if isinstance(self.perc, GatedPerception) else {})})

    def policy_info(self):
        return self.loco.info()

    def describe(self):
        p = self.p
        return ('v11 frontal fetch. GR00T WBC (Balance/Walk ONNX, its own trained observations) drives legs+waist from velocity / base-height '
                'commands; a PI path follower turns planned pelvis paths into velocity commands from the simulator '
                'pelvis pose. PERCEPTION every 50 Hz tick: the recipient skeleton (18 keypoints, ideal perception of his '
                'replayed motion) and the four tabletop objects (TRUE poses)'
                + ('' if not p['headcam'] else
                   ', each used only while it is inside the ASSUMED head camera frame and unoccluded (mj_ray from the '
                   'camera; pointing needs shoulder, elbow, wrist, chest and hips in view)')
                + '; pointing gesture = elbow > '
                f"{p['point_elbow_deg']:.0f} deg, wrist >= {p['point_reach_m']:.2f} m from the chest and above the hips, "
                f"held >= {p['point_hold_s']:.2f} s; ray shoulder -> wrist; target = the object that stays best by >= "
                f"{p['target_margin_deg']:.0f} deg and within {p['target_max_deg']:.0f} deg of the ray for >= "
                f"{p['target_hold_s']:.2f} s. The robot balances in place until the commit"
                + ('' if p['watch_pitch'] == 0.0 else
                   f" (torso pitch command {p['watch_pitch']:.2f} rad while watching, removed over {p['unlean_T']:.1f} s "
                   'from the commit)')
                + ", then plans the stance, the "
                f"approach and the grasp around the selected object. Left arm: joint PD ({p['arm_kp_scale']:g}x upstream deploy gains"
                + ('' if p['arm_kp_scale_carry'] is None else f", {p['arm_kp_scale_carry']:g}x during the carry") + ') + '
                'robot-only gravity compensation, tracking online damped-least-squares IK targets ('
                + ('natural per-joint bounds' if p['ik_limits'] else
                   f"shoulder yaw, wrist roll and wrist yaw bounded to +/-{p['nat']:.2f} rad") + ') '
                'for the left UMI fingertip centre, with a Cartesian integral; the 50 Hz arm joint targets are rate '
                f"limited (left <= {p['dq_max_left']:g} rad/tick"
                + ('' if p['dq_max_carry'] is None else f", {p['dq_max_carry']:g} rad/tick during the carry walk")
                + f", right <= {p['dq_max_right']:g} rad/tick); "
                + ('while walking with the bag and while presenting, the left-arm joint targets ' if p['carry_arm_frozen']
                   else 'while presenting, the left-arm joint targets ')
                + 'are frozen. Right arm ramps to a relaxed posture at its side. Fingers: force-limited position servos '
                '(35 N), wide open until the grasp, closed from the grasp to the end. Recipient: replayed recording '
                '(pointing gesture, then idle ping-pong). FETCH AND PRESENT ONLY: no handover.')

    def disclosures(self):
        p = self.p
        where = 'in front of him' if p['present_anchor'] == 'chest' else 'in front of his idle left hand'
        return ['NO HANDOVER (fallback episode): the robot picks the bag up, carries it to the person, holds it out '
                f'{where} and keeps holding it until the episode ends; he never reaches or touches it '
                '(gates G5 handover and G6 pace fail by design).',
                self.perception_disclosure(),
                'The robot balances in place (zero velocity command) until the target is committed; the stance, approach '
                'path and grasp are planned only after the commit, around the selected object\'s pose. Only the backpack '
                'has a modelled grasp; any other target would stop the sequence.',
                'Ideal localization: the path follower and the arm IK read the pelvis pose from the simulator.',
                f"Privileged object state: the grasp, pre-grasp and lift targets, the grasp re-solve and the alignment use "
                f"the TRUE simulated bag pose (scene grasp frame {p['grasp_site']}"
                + ('' if p['grasp_pos_bag'] is None and not p['grasp_tilt_deg'] else
                   ', moved and tilted as described under Grasp') +
                f"); the presentation target uses the TRUE "
                f"bag top ({p['present_site']}); the drop check uses the TRUE bag height.",
                'Drop detection: robot-side, the left UMI finger position closing more than '
                f"{p['drop_close'] * 1000:.0f} mm beyond its grasp position; privileged, the TRUE bag height below "
                f"{p['drop_z']:.2f} m after the lift. Either stops the sequence (the robot stands still).",
                self.present_disclosure(),
                'Right arm: kept relaxed at the side; the robot carries one-handed.',
                f"Arm PD gains are {p['arm_kp_scale']:g}x the upstream deploy gains (stiffness; damping scaled by the square root)"
                + ('' if p['arm_kp_scale_carry'] is None else
                   f"; from the start of the carry walk the left arm ramps to {p['arm_kp_scale_carry']:g}x within 0.5 s")
                + '; a Cartesian integral correction on the measured tool '
                'error is added to the IK target; while presenting, the left-arm joint targets are frozen at their last '
                'value (no IK).',
                self.grasp_disclosure()] + self.v10_disclosures() + self.v11_disclosures()

    def perception_disclosure(self):
        p = self.p
        rule = ('pointing gesture = elbow extended over '
                f"{p['point_elbow_deg']:.0f} deg, wrist at least {p['point_reach_m']:.2f} m from the chest and above the "
                f"hips for {p['point_hold_s']:.2f} s; each object is scored by the angle between the pointing ray and the "
                f"direction to its centre; commit when one object stays best by {p['target_margin_deg']:.0f} deg and "
                f"within {p['target_max_deg']:.0f} deg for {p['target_hold_s']:.2f} s. " + RAY_NOTE)
        if p['headcam']:
            mt = p['headcam']
            return ('HEAD CAMERA (ASSUMED SENSOR) AND VISIBILITY-GATED IDEAL PERCEPTION: the stock G1 head camera (an '
                    'RGB-D camera) is pitched 47.6 deg down and cannot see the arms or head of a standing person a few '
                    'metres away; this run assumes a '
                    f"forward-looking RGB camera in the head (on the centre line of the face, pitched {mt['pitch_down_deg']:.0f} "
                    f"deg down, 16:9, {mt['hfov_deg']:.0f} deg horizontal field of view; pose from the torso link, no mass, "
                    'not part of the physics model). The robot reads the person\'s skeleton keypoints from his replayed '
                    'recording and the object centres from their TRUE simulated poses (no camera images, detector or '
                    'noise), but uses a keypoint only while it is inside that camera\'s frame and unoccluded (a ray from '
                    'the camera, against the simulated scene), and compares only objects in view. Pointing needs the '
                    'shoulder, elbow and wrist of the arm, the chest and both hips in view. The target inference runs in '
                    'the control loop and decides when and where the robot moves: ' + rule)
        return ('IDEAL PERCEPTION: the robot reads the person\'s skeleton (18 keypoints) directly from his replayed '
                'recording as the simulator poses him at each 50 Hz tick, and the four tabletop object centres from their '
                'TRUE simulated poses; no camera images, detector or noise. The target inference itself runs in the '
                'control loop and decides when and where the robot moves: ' + rule)

    def present_disclosure(self):
        p = self.p
        if p['carry_mode'] == 'turn_walk' and p['present_anchor'] == 'chest':
            return ('Presentation point (ideal perception of the person): the top of the bag is placed '
                    f"{p['present_dist']:.2f} m in front of the person's recorded chest (frame 200) along his facing direction "
                    f"(yaw {p['nfar_yaw_deg']:.0f} deg); the robot ends its walk facing his chest, and its stance is solved so "
                    'that the bag, held at the frozen carry pose, lands at that point horizontally; the bag is presented at '
                    'the height it was carried at; after the stop the arm is not moved.')
        return ('Presentation point (ideal perception of the person): the top of the bag is placed '
                f"{p['present_dist']:.2f} m in front of the person's recorded idle left shoulder (frame 200) along yaw "
                f"{p['nfar_yaw_deg']:.0f} deg, at {p['present_z']:.2f} m height, i.e. in front of his hanging left hand; "
                'the robot heading is the direction of its last, straight walk segment, the stance is solved so that the '
                'bag, held at the carry pose, lands at that point; after the stop the arm is not moved.')

    def v10_disclosures(self):
        p = self.p
        out = []
        if p['watch_pitch'] != 0.0:
            out.append(f"Watch lean: while the robot watches the person, before the target commit, the controller ramps the "
                       f"GR00T torso pitch command to {p['watch_pitch']:.2f} rad within {p['watch_ramp']:.1f} s (leaning "
                       'back; an input of the trained policy, which itself moves the waist and legs) so that the chest '
                       'camera looks up toward the person; the walk velocity command stays zero. From the commit the '
                       f"lean is removed within {p['unlean_T']:.1f} s, and the arm motions planned at the commit are "
                       'planned for the upright torso (the waist angles GR00T stands with when it does not lean). The '
                       'camera still cannot see his shoulder, elbow or head; '
                       'perception does not use the camera image.')
        if p['approach_mid']:
            vias = '; '.join(f"at {fr * 100:.0f}% of the walk time: x {x_:.3f} m, y {y_:.3f} m, heading {yw_:.0f} deg"
                             for fr, x_, y_, yw_ in p['approach_mid'])
            out.append('Approach path: the walk from the measured position at the commit to the stance passes '
                       f"{len(p['approach_mid'])} fixed via point(s) ({vias}), set in the parameters for this start pose "
                       'so that the robot finishes turning toward the stance heading before it arrives; the path is '
                       'planned and walked only after the target commit.')
        if p['ik_limits']:
            names = ['shoulder pitch', 'shoulder roll', 'shoulder yaw', 'elbow', 'wrist roll', 'wrist pitch', 'wrist yaw']
            lim = '; '.join(f"{names[int(j)]} {lo:+.2f} to {hi:+.2f}" for j, (lo, hi) in sorted(p['ik_limits'].items(),
                                                                                              key=lambda kv: int(kv[0])))
            out.append('Arm IK bounds (rad, within the joint limits): ' + lim + '. They keep a natural forward reach '
                       '(the upper arm away from the torso, no hyperextended elbow) while the gripper is turned so that its '
                       'jaws close across the bag\'s front-back thickness. Planning solutions closer than '
                       f"{p['ik_self_clear'] * 1000:.0f} mm to the robot's own body, and fallback solutions more than "
                       f"{p['global_max_jump'] or 0:.1f} rad away from the current joint targets, are rejected."
                       if p['global_max_jump'] else 'Arm IK bounds (rad, within the joint limits): ' + lim + '.')
        elif p['nat'] > 0.6 + 1e-9:
            out.append(f"Arm IK range: shoulder yaw, wrist roll and wrist yaw are bounded to +/-{p['nat']:.2f} rad (wider "
                       'than the +/-0.6 rad natural range of earlier versions) so that the left arm, reaching forward, can '
                       'turn the gripper until its jaws close across the bag\'s front-back thickness.' if p['stow_rel'] is not None
                       else f"Arm IK range: shoulder yaw, wrist roll and wrist yaw are bounded to +/-{p['nat']:.2f} rad (wider "
                       'than the +/-0.6 rad natural range of earlier versions) so that the left arm can reach the top of the '
                       'bag sideways from a stance west of the table, off the bed, with the fingers straight down.')
        if np.any(np.array(p['torso_rpy_grasp'], float)):
            r, pt, y = p['torso_rpy_grasp']
            keep = p['t_torso_out'] >= 100
            out.append(f"Torso lean: after the target commit the controller ramps the GR00T torso orientation command to roll "
                       f"{r:.2f} rad, pitch {pt:.2f} rad, yaw {y:.2f} rad (an input of the trained policy; the policy itself "
                       'moves the waist and legs)' + (', and keeps it until the end of the episode.' if keep else
                                                      (', and ramps it back to zero after the lift, before walking.'
                                                       if p['torso_out_at'] == 'hold' else
                                                       ', and ramps it back to zero during the carry.')))
        if p['finger_hold_mode'] == 'contact':
            out.append('Finger hold: once the fingers have closed on the bag, each finger servo target is set '
                       f"{p['finger_squeeze_m'] * 1000:.1f} mm inside that finger's measured contact position (robot-side "
                       f"finger encoders), about {p['finger_squeeze_m'] * 5000:.0f} N per finger instead of the 35 N limit, "
                       'so that the two position servos centre the bag between the pads; with both servos at their force '
                       'limit the bag can slide back and forth along the closing axis during the walk.')
        if p['base_h_carry'] is not None:
            out.append(f"Carry walk: the GR00T base-height command is {p['base_h_carry']:.2f} m from the start of the carry "
                       f"(standing and grasping: {p['base_h']:.2f} m).")
        if p['desc_mode'] == 'v10' and p['pre_vertical']:
            out.append(f"Grasp descent: the open, tilted gripper comes straight down from {p['pre_up'] * 100:.0f} cm above the grasp to "
                       f"{p['desc_hold'] * 100:.0f} cm above it without integral correction, centres the pads there (closing and "
                       f"pad-width axes), then makes the last {p['desc_hold'] * 100:.0f} cm slowly; all targets come from the TRUE "
                       'bag grasp frame (privileged).')
        elif p['desc_mode'] == 'v10':
            out.append(f"Grasp descent: the open gripper descends to {p['desc_hold'] * 100:.0f} cm above the grasp depth without "
                       'integral correction, centres the pads there (closing and pad-width axes), then makes the last '
                       f"{p['desc_hold'] * 100:.0f} cm slowly; all targets come from the TRUE bag grasp frame (privileged).")
        if p['carry_arm_frozen'] and (p['stow_rel'] is not None or p['carry_mode'] == 'turn_walk'):
            pass                                    # v11 carry text: v11_disclosures()
        elif p['carry_arm_frozen']:
            out.append('Carry: while walking, the left-arm joint targets stay frozen at the posture the arm had after the '
                       'lift (no IK while walking, so no per-tick arm target steps); the bag is carried at the robot\'s left '
                       'side, the presentation stance is solved for that fixed hand position, and the bag is presented at the '
                       'height it was carried at.')
        elif p['carry_axis_only']:
            out.append('Carry: the arm IK keeps the gripper fingers pointing straight down and leaves the yaw of the bag free; '
                       'the bag is carried at the robot\'s left side.')
        if p['carry_frame_tau'] > 0 and not p['carry_arm_frozen']:
            out.append('Carry: the hand target is placed relative to the planned walking path plus a low-pass filtered '
                       f"(time constant {p['carry_frame_tau']:.2f} s) deviation of the measured pelvis pose from it, so the "
                       'arm cancels the step-to-step sway of the body instead of passing it to the bag.')
        return out


def _grasp_disclosure(self):
    p = self.p
    if p['grasp_pos_bag'] is None and not p['grasp_tilt_deg']:
        return ('Grasp: direct body grasp of the rigid bag (no handle, no strap): the wide-open gripper comes down over '
                'the top of the upright bag, fingers straight down, jaws closing across the bag\'s front-back thickness '
                '(about 110 mm under the pads), pads centred on its centre-of-mass plane; the bag can pivot a few degrees '
                'between the pads.')
    gx, gy, gz = p['grasp_pos_bag'] if p['grasp_pos_bag'] is not None else (0.0, 0.017, 0.38648)
    return ('Grasp: direct body grasp of the rigid bag (no handle, no strap): the wide-open gripper comes down over the top '
            'of the upright bag from the front of the robot, jaws closing across the bag\'s front-back thickness (about '
            f"110 mm under the pads). The planned grasp frame is the scene grasp frame {p['grasp_site']} moved to the bag-frame "
            f"point ({gx:.4f}, {gy:.4f}, {gz:.4f}) m and tilted {p['grasp_tilt_deg']:.0f} deg about the bag's front-back axis "
            '(fingertips leaning away from the robot, so that the forearm reaches forward and down with a relaxed elbow)'
            + (f", its closing axis turned {p['grasp_yaw_deg']:.1f} deg about the vertical" if p['grasp_yaw_deg'] else '')
            + ': the pad-tip line sits '
            f"{abs(p['grasp_pad_x_mm']):.1f} mm beside the bag's centre-of-mass plane and, with the tilt, the pads press "
            'about 2 to 4 cm from the middle of the bag, so that the back pad presses on ONE convex hull of the '
            'scanned shoulder straps instead of the seam between two hull pieces (bag_1_20 and bag_1_21), where the contact '
            'switched between the pieces and the pad force dropped for single physics steps in earlier runs. Chosen with the '
            'test-only gantry probe (replays of recorded carry motions); the bag can pivot a few degrees between the pads.')


def _v11_disclosures(self):
    p = self.p
    out = []
    if p['approach_mode'] == 'auto2':
        out.append('Approach path (planned at the commit from the measured pelvis pose, so it works from any start pose): '
                   f"turn toward a point {p['approach_back']:.2f} m behind the pickup stance, walk straight to it, turn in "
                   'place to face the bag, step forward to the stance. The pickup stance is west of the low table facing '
                   f"the bag (heading {p['stance_yaw_deg']:.0f} deg), {p['stance_f']:.2f} m behind and {p['stance_l']:.2f} m "
                   'to the right of the grasp point, off the bed. The arm keeps its standing posture while walking and is '
                   'raised in front of the body only once the robot turns to face the bag.')
    if p['approach_mode'] == 'auto':
        out.append('Approach path (planned at the commit from the measured pelvis pose, so it works from any start pose): '
                   'turn toward the pickup stance while starting to walk, walk straight, turn to face the bag while '
                   f"arriving; pickup stance west of the low table facing the bag (heading {p['stance_yaw_deg']:.0f} deg), "
                   f"{p['stance_f']:.2f} m behind and {p['stance_l']:.2f} m to the right of the grasp point, off the bed.")
    if p['lift_mode'] == 'joint_path':
        out.append(f"Lift: {p['lift'] * 100:.0f} cm straight up in {p['t_lift']:.1f} s along a joint path planned when the fingers "
                   'have closed (inverse kinematics continued in small steps from the measured arm pose, gripper orientation '
                   'kept fixed relative to the robot), with no online IK during the lift and the arm targets limited to '
                   f"{p['dq_max_lift'] * 50:.1f} rad/s." if p['dq_max_lift'] else
                   f"Lift: {p['lift'] * 100:.0f} cm straight up along a joint path planned when the fingers have closed.")
    if p['stow_rel'] is None and p['carry_arm_frozen'] and p['carry_mode'] == 'turn_walk':
        out.append('Carry: the bag stays at the pose the lift ends in, in front of the robot (the gripper about 0.4 m ahead '
                   'of the pelvis); the left-arm joint targets are frozen for the whole walk (no IK while walking, so no '
                   'per-tick arm target steps). The robot first steps straight back '
                   f"{p['carry_back']:.2f} m from the table, turns in place, walks straight and turns to face the person while "
                   'arriving; the presentation stance is solved for the fixed hand position.')
    if p['stow_rel'] is not None:
        f_, l_, z_ = p['stow_rel']
        out.append('Carry: after the lift the robot, standing, brings the bag with online IK to a carry pose in front of its '
                   f"body (gripper {f_:.2f} m ahead of the pelvis, {z_:.2f} m high), then freezes the left-arm joint targets "
                   'for the whole walk (no IK while walking, so no per-tick arm target steps). It turns in place while stepping '
                   'back from the table, walks straight and stops facing the person; the presentation stance is solved so '
                   'that the bag, held at the frozen carry pose, lands at the presentation point, and it is presented at the '
                   'height it was carried at.')
    return out


PointFetch.grasp_disclosure = _grasp_disclosure
PointFetch.v11_disclosures = _v11_disclosures


class RecipientMotionRate(R6.Recipient6):
    """Recipient6 whose recipient-motion.json is written at motion_fps (output only: the colliders are posed by the
    same full_pose(t) at every physics step either way).  At the 50 Hz control-tick rate every exported frame of the
    website clip is a stored frame, so the drawn avatar equals the physics colliders exactly instead of being linearly
    interpolated between 30 fps frames."""
    motion_fps = None

    def write_motion_json(self, path, t_end, fps=None, extra=None):
        return super().write_motion_json(path, t_end, fps=fps or self.motion_fps, extra=extra)


def make_recipient(params, xml):
    if params.get('recipient_motion_fps'):
        r = RecipientMotionRate.for_scene(xml)
        r.motion_fps = float(params['recipient_motion_fps'])
        return r
    return R6.Recipient6.for_scene(xml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--params', default='{}')
    ap.add_argument('--xml', default=str(SCENE_XML))
    ap.add_argument('--max-time', type=float, default=18.0)
    ap.add_argument('--no-check', action='store_true')
    a = ap.parse_args()
    params = json.loads(a.params) if not a.params.endswith('.json') else json.loads(Path(a.params).read_text())
    ctrl = PointFetch(params)
    rcp = make_recipient(ctrl.p, a.xml)
    runner = PointFetchRunner(ctrl, a.out, xml=a.xml, recipient=rcp, max_time=a.max_time, check=not a.no_check,
                              description='v11 frontal point-fetch-present: GR00T WBC; the robot perceives the recipient\'s '
                                          'pointing gesture, infers the pointed-at object, then walks to the backpack, faces '
                                          'it, grasps the bag body directly with the left UMI, carries it in front of its '
                                          'body to the recipient and holds it out to him, facing him (FETCH AND PRESENT, '
                                          'no handover)')
    s = runner.run()
    print(json.dumps(s['outcome'], default=str))
    (Path(a.out) / 'params.json').write_text(json.dumps(ctrl.p, indent=1) + '\n')


if __name__ == '__main__':
    main()
