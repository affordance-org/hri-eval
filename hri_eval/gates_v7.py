#!/usr/bin/env python3
"""v7 gates G7 (visual honesty), G8 (controlled handover), G9 (smoothness), used by check_gates.py.

v7 changes (copy of gates_v6.py; everything not listed is v6):
  * G7a samples robot cylinders/capsules (skin.GeomSampler int(geom_type) fix) and never reduces an empty sample set.
  * G7c also checks the v7 BODY colliders (recipient_<pelvis|spine_0x|thigh_x|calf_x|foot_x>) against the json bones.
  * G8c (pinch kept until the recipient carries >= 0.6 W) counts only the recipient's HAND/ARM support (`rpad`); the
    bag resting on his torso/leg colliders is reported separately (fz_recipient_body) and never counts as taken.
  * G8d is STRUCTURED: an event uses recipient forces iff its `signals` list holds a signal outside ROBOT_SIGNALS, or a
    payload KEY names a recipient/hook/rpad force; such an event must carry privileged: true (+ summary disclosure).
    Free-text reasons are not scanned.
  * G8e is STRUCTURED: the release decision events are the events in [t_open - 1 s, t_open + 1 tick] whose event NAME
    is one of RELEASE_EVENTS ('release', 'open', 'let_go', 'release_decision'), or that carry `decision: 'release'`;
    `phase` events are never decision events, whatever their reason says.  Each decision event must list only
    robot-side `signals` or be privileged + disclosed; at least one must exist.
  * G9b is SPLIT: UMI tool speed <= 0.6 m/s while the tool is within 10 cm of the bag-recipient contact region (every
    physics-step bag<->recipient contact point of the trial; if there is none, the recipient's hand bodies at that step)
    and during the GRASP (0.5 s before the first finger-bag contact .. 0.5 s after lift-off) and the RELEASE (0.5 s
    before the opening command .. 0.5 s after the last finger-bag contact); <= 1.2 m/s elsewhere in the window
    [0.5 s before the first finger-bag contact, t_hc (or the end)].

v6 notes:

v6 changes (everything else is v5, below):
  * G7 measures against the CURLED visible skin: recipient-motion.json frames carry `curl_l` and
    adaptation.curl_l.rig holds the rig (curl.py formula + parameters); the skin of a frame is LBS(V0 + delta(curl_l))
    (= the website's skinned mesh with its morph).  Frames without curl_l (v4/v5 trials) are curl 0 = the flat hand.
    At physics steps the curl is the json curl interpolated linearly in time (the website interpolates its keys).
  * G7c also checks every finger-segment collider body (recipient_hand_l_<segment>) against the json hand bone +
    json curl (<= 0.5 mm, <= 0.5 deg).  A curl scene whose json has no curl rig fails G7c (invisible fingers).
  * G7d treats left-hand frames with curl > 0 (or changing) as adapted (the skin speed includes the fingers).
  * G8e release provenance: the robot's release (first controller tick after the pickup at which the holding UMI's
    finger target opens by > 2 mm) must be preceded (<= 1 s before, <= 1 tick after) by a release/open event that
    either lists only robot-side signals in `signals` (ROBOT_SIGNALS) or carries `privileged: true` (and summary.json
    discloses privileged force use).  No opening command -> not applicable.

Data sources (all independent of the controller's self-reports, except where noted):
  * the re-simulation of check_gates.py (contacts and forces at EVERY physics step via StepCollector);
  * recipient-motion.json (what the website shows: the person's bones at 30 fps) + the avatar skin (skin.py);
  * trajectory.npz mocap rows (the simulated recipient colliders);
  * controller-log.npz q_target (the arm joint targets the PD tracked; G9);
  * events.json / summary.json only for the privileged-signal disclosure check of G8.

G7 visual honesty
  a) penetration at every 30 fps frame (state k = round(t/dt), mocap row k+1, skin = LBS of the json frame):
       robot geometry (collision AND visual geoms, ~2 mm surface samples) inside the skin <= 2 mm;
       bag + strap geometry inside the skin <= 3 mm.
  b) every physics-step contact between the bag and a recipient collider lies within 3 mm of the visible skin surface
     (|signed distance| of the contact point; skin at that step = mocap poses for collider-backed bones, json
     interpolation for the others) -> no invisible proxy carries the bag;
     every physics-step robot<->recipient contact: robot geom samples inside the skin at that step <= 2 mm.
  c) colliders follow the visible bones: at frames that fall on a physics step (t multiple of 0.1 s) each
     recipient_<bone> mocap pose equals the json bone pose (<= 0.5 mm, <= 0.5 deg).  (N/A for scenes without
     skin-matched bodies; then (b) uses the json bones only.)
  d) recipient hand motion while ADAPTED (json, both hands, 30 fps; frames where that arm's adaptation weight > 0):
     max hand-skin vertex speed <= 1.0 m/s, hand bone angular speed <= 300 deg/s, no frame-to-frame rotation jump
     > 12 deg; also at every physics step from the hand mocap bodies.  (His unadapted recorded motion reaches
     1.8 m/s (left hand) / 3.5 m/s (right hand, f167) in the clip; it is reported, not gated.)
  e) anatomy on ADAPTED frames (adaptation weight > 0, per arm): wrist flexion/extension within +-70, radial <= 20,
     ulnar <= 30, elbow flexion 0-145 (skin.HandAnatomy definitions, = v4 auditor's); recorded frames are reported.
  f) no full-body freeze > 1.0 s: a frame is still when no skin vertex moved >= 0.3 mm since the previous frame
     (~1 cm/s); idle ping-pong counts as motion when the skin moves.
  g) while a recipient body carries load (recipient vertical support >= 0.25 W) every contacting recipient body point
     moves <= 2 cm/s and, when moving faster than 2 mm/s, within 20 deg of vertical (physics steps, mocap).
G8 controlled handover (after the G3 pickup start)
  a) the bag is never without any contact (normal force > EPS) for more than 10 ms;
  b) the NET contact force on the bag from each contact class (and in total) never exceeds 3 W (squeeze forces of a
     pinch cancel; impacts and drops do not);
  c) the holding UMI keeps its pinch (both finger pads >= 5 N on the bag) until the recipient carries >= 0.6 W
     (first pinch loss after pickup: recipient vertical support at the previous step >= 0.6 W);
  d) robot-side decisions that use recipient contact forces are logged as privileged: every event whose payload
     names a recipient/hook force must carry privileged: true, and summary.json must disclose it.
G9 smoothness
  a) no 50 Hz arm joint-target jump > 0.25 rad (controller-log q_target, all 14 arm joints);
  b) (v6) UMI tool speed (<side>_grasp_center site) <= 0.6 m/s from 0.5 s before the first finger-bag contact to the
     handover completion (t_hc, or the end);  v7: 0.6 m/s only near the bag-recipient contact region (10 cm) and
     during grasp / release, 1.2 m/s elsewhere (see the v7 notes above);
  c) right arm changes ramped: per-tick right-arm target change <= 0.05 rad (2.5 rad/s).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

import skin as SKN
import curl as CU
import curl_colliders as CC

EPS_N = 0.01
BODY_BONES = ('pelvis', 'spine_01', 'spine_02', 'spine_03', 'thigh_l', 'calf_l', 'foot_l', 'thigh_r', 'calf_r', 'foot_r')
SKIN_BODY_BONES = ('upperarm_l', 'lowerarm_l', 'hand_l', 'upperarm_r', 'lowerarm_r', 'hand_r') + BODY_BONES
RELEASE_EVENTS = {'release', 'open', 'let_go', 'release_decision'}
G9B_NEAR_LIMIT, G9B_FAR_LIMIT, G9B_NEAR_RADIUS = 0.6, 1.2, 0.10
ROBOT_SIGNALS = {'umi_finger_position', 'umi_finger_force', 'finger_actuator_force', 'joint_position', 'joint_velocity',
                 'joint_torque', 'arm_joint_torque', 'wrist_force_torque', 'tactile', 'ego_camera', 'time',
                 'robot_proprioception', 'qfrc_actuator', 'base_imu', 'controller_state'}


def curl_info(rm, sk):
    """(rig or None, per-frame curl array) from a recipient-motion.json dict."""
    ad = rm.get('adaptation') or {}
    params = (ad.get('curl_l') or {}).get('rig')
    c = np.array([float(f.get('curl_l', 0.0)) for f in rm['frames']])
    rig = CU.CurlRig.from_params(params, sk) if params else None
    return rig, c


def curled_skin(sk, rig, P, Q, c):
    V = sk.pose(P, Q)
    if rig is not None and c > 0:
        V = rig.pose(sk, P, Q, c, V=V)
    return V


class StepCollector:
    """Per-physics-step bag / recipient contact data from the checker's re-simulation (call after every mj_step with
    the pre-step state index k)."""

    def __init__(self, m, C, weight):
        self.m, self.C, self.W = m, C, weight
        self.bag_set = C.bag_bodies
        self.rec_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith('recipient_')}
        # v7: the torso/pelvis/leg colliders are recipient BODY support (never "taken by the hand")
        self.rec_body_bodies = {b for b in self.rec_bodies
                                if m.body(b).name.replace('recipient_', '') in BODY_BONES}
        self.rec_hand_pts = []         # (k, positions of recipient_hand_* bodies) for G9b without bag contact
        self.hand_bodies = [b for b in self.rec_bodies if m.body(b).name in ('recipient_hand_l', 'recipient_hand_r')]
        self.fz_rec_body = []
        self.robot_cls = {'foot', 'finger_L', 'finger_R', 'umi_base', 'robot_other'}
        self.finger_body = {}
        for b in range(m.nbody):
            n = m.body(b).name
            for side in ('left', 'right'):
                for f in ('left', 'right'):
                    if n.startswith(f'{side}_umi_{f}_finger'):
                        self.finger_body[b] = f'{side}_{f}'
        self.sites = {s: m.site(f'{s}_grasp_center').id for s in ('left', 'right')}
        self.any_contact, self.net = [], []
        self.net_cls = {}
        self.pad_fn = []
        self.fz_rec = []
        self.rec_contacts = []         # (k, recipient geom, other geom, other class, pos, fn)
        self.tool = []
        self.f6 = np.zeros(6)

    def collect(self, d, k):
        m, C = self.m, self.C
        net = np.zeros(3)
        cls_f = {}
        pads = {'left_left': 0.0, 'left_right': 0.0, 'right_left': 0.0, 'right_right': 0.0}
        anyc = False
        fz_rec = 0.0
        fz_body = 0.0
        ncon = d.ncon
        if ncon:
            gg = d.contact.geom[:ncon]
            b1 = m.geom_bodyid[gg[:, 0]]
            b2 = m.geom_bodyid[gg[:, 1]]
            for i in range(ncon):
                i1, i2 = int(b1[i]) in self.bag_set, int(b2[i]) in self.bag_set
                r1, r2 = int(b1[i]) in self.rec_bodies, int(b2[i]) in self.rec_bodies
                if not (i1 or i2 or r1 or r2) or (i1 and i2):
                    continue
                c = d.contact[i]
                mj.mj_contactForce(m, d, i, self.f6)
                fn = float(self.f6[0])
                if fn <= EPS_N:
                    continue
                F2 = c.frame.reshape(3, 3).T @ self.f6[:3]
                if i1 or i2:
                    anyc = True
                    other_b = int(b1[i]) if i2 else int(b2[i])
                    Fb = F2 if i2 else -F2
                    oc = C.geom_cls(int(gg[i, 0]) if i2 else int(gg[i, 1]))
                    net += Fb
                    cls_f[oc] = cls_f.get(oc, np.zeros(3)) + Fb
                    if other_b in self.finger_body:
                        pads[self.finger_body[other_b]] += fn
                    if other_b in self.rec_body_bodies:
                        fz_body += Fb[2]
                    elif other_b in self.rec_bodies:
                        fz_rec += Fb[2]
                if r1 or r2:
                    gr, go = (int(gg[i, 0]), int(gg[i, 1])) if r1 else (int(gg[i, 1]), int(gg[i, 0]))
                    ob = int(m.geom_bodyid[go])
                    ocl = 'bag' if ob in self.bag_set else C.geom_cls(go)
                    self.rec_contacts.append((k, gr, go, ocl, c.pos.copy(), fn))
        self.any_contact.append(anyc)
        self.net.append(net)
        for kk in set(cls_f) | set(self.net_cls):
            self.net_cls.setdefault(kk, [np.zeros(3)] * (len(self.net) - 1)).append(cls_f.get(kk, np.zeros(3)))
        self.pad_fn.append([pads['left_left'], pads['left_right'], pads['right_left'], pads['right_right']])
        self.fz_rec.append(fz_rec)
        self.fz_rec_body.append(fz_body)
        self.rec_hand_pts.append(np.array([d.xpos[b] for b in self.hand_bodies]) if self.hand_bodies else np.zeros((0, 3)))
        self.tool.append(np.concatenate([d.site_xpos[self.sites['left']], d.site_xpos[self.sites['right']]]))

    def arrays(self):
        n = len(self.net)
        return {'any_contact': np.array(self.any_contact, bool), 'net': np.array(self.net),
                'net_cls': {k: np.array(v[:n]) for k, v in self.net_cls.items()},
                'pad_fn': np.array(self.pad_fn), 'fz_rec': np.array(self.fz_rec), 'tool': np.array(self.tool),
                'fz_rec_body': np.array(self.fz_rec_body)}


# ------------------------------------------------------------------------------------------------ helpers
def _ang_deg(R1, R2):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R1.T @ R2) - 1) / 2))))


def skin_bones_at_step(m, mocap_pos_row, mocap_quat_row, rm_arr, t):
    """(P, Q) of the person at a physics step: collider-backed bones from the mocap row, others from the json."""
    P, Q = SKN.motion_pose(rm_arr, t)
    backed = []
    for bone in SKIN_BODY_BONES:
        try:
            bid = m.body(f'recipient_{bone}').id
        except KeyError:
            continue
        mid = m.body_mocapid[bid]
        if mid < 0:
            continue
        i = NAMES_IDX[bone]
        P[i], Q[i] = mocap_pos_row[mid], mocap_quat_row[mid]
        backed.append(bone)
    return P, Q, backed


NAMES_IDX = {}


def runs(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


# ------------------------------------------------------------------------------------------------ G7
def gate_g7(trial, m, tr, coll: StepCollector, C, weight, frame_stride=1, verbose=False):
    trial = Path(trial)
    rm = json.loads((trial / 'recipient-motion.json').read_text())
    sk = SKN.Skin(rm)
    NAMES_IDX.clear()
    NAMES_IDX.update(sk.idx)
    tk, Pk, Qk = SKN.motion_arrays(rm)
    rm_arr = (tk, Pk, Qk)
    rig, curl_k = curl_info(rm, sk)
    curl_bodies = {m.body(b).name: CC.seg_of_body(m.body(b).name) for b in range(m.nbody)
                   if m.body(b).name.startswith('recipient_hand_l_') and m.body_mocapid[b] >= 0}
    res_curl = {'json_has_curl_rig': rig is not None, 'scene_curl_bodies': len(curl_bodies),
                'curl_range_json': [float(curl_k.min()), float(curl_k.max())] if len(curl_k) else None}
    dt = m.opt.timestep
    T = tr['time']
    N = len(T)
    qpos, mpos, mquat = tr['qpos'], tr['mocap_pos'], tr['mocap_quat']
    d = mj.MjData(m)
    root = m.body_rootid
    pel = m.body('pelvis').id
    robot_g = [g for g in range(m.ngeom) if root[m.geom_bodyid[g]] == root[pel]]
    bag_g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in C.bag_bodies]
    gs = SKN.GeomSampler(m, spacing=0.002)
    res = {'frames_checked': 0}
    worst_robot = (0.0, None, None)
    worst_bag = (0.0, None, None)
    robot_rows, bag_rows = [], []
    still = []
    prevV = None
    sub = np.arange(0, len(sk.V0), 3)
    hand_ix = {s: sk.bone_vertices(f'hand_{s}', 0.5) for s in ('l', 'r')}
    anat = {s: SKN.HandAnatomy(sk, s) for s in ('l', 'r')}
    hand_speed, hand_ang, hand_jump = [], [], []
    rec_speed, rec_ang = [], []
    anat_rows = []
    prev_hand = None
    w_l = np.array([f.get('left_arm_adaptation_weight', 0.0) for f in rm['frames']])
    w_r = np.array([f.get('right_arm_adaptation_weight', 0.0) for f in rm['frames']])
    consistency = []
    A_LIMIT = {'robot': 0.002, 'bag': 0.003}
    failed_at = {'robot': None, 'bag': None}
    for j in range(0, len(tk), frame_stride):
        t = float(tk[j])
        k = int(round(t / dt))
        if k >= N - 1:
            break
        d.qpos[:] = qpos[k]
        d.mocap_pos[:] = mpos[k + 1]
        d.mocap_quat[:] = mquat[k + 1]
        mj.mj_kinematics(m, d)
        V = curled_skin(sk, rig, Pk[j], Qk[j], float(curl_k[j]))
        surf = sk.surface(V) if None in failed_at.values() else None
        res['frames_checked'] += 1
        for name, geoms, rows in (('robot', robot_g, robot_rows), ('bag', bag_g, bag_rows)):
            if failed_at[name] is not None:
                # part a has failed for this side: the rest cannot make it pass, and sampling every geom near the
                # skin costs seconds per frame for a robot that leans on the person, so stop checking this side
                continue
            depth = 0.0
            who = None
            for g in geoms:
                c = d.geom_xpos[g]
                rb = m.geom_rbound[g] if m.geom_rbound[g] > 0 else 0.5
                dn = surf.tree.query(c)[0]
                if dn > rb + 0.02:
                    continue
                pts = gs.world(d, g)
                if not len(pts):                       # v7: a geom type without samples is skipped, never crashes
                    continue
                sd, _, _ = surf.signed_distance(pts, max_dist=0.05)
                dpt = float(max(0.0, -sd.min()))
                if dpt > depth:
                    depth, who = dpt, (m.geom(g).name or f'{m.body(m.geom_bodyid[g]).name}#g{g}',
                                       sk.names[sk.dom[surf.used[surf.tree.query(pts[int(np.argmin(sd))])[1]]]])
            rows.append((t, depth, who))
            if depth > A_LIMIT[name] + 1e-9:
                failed_at[name] = t
        # colliders follow the visible bones (frames on a physics step)
        if abs(t / dt - k) < 1e-6:
            for bone in SKIN_BODY_BONES:
                try:
                    bid = m.body(f'recipient_{bone}').id
                except KeyError:
                    continue
                mid = m.body_mocapid[bid]
                i = sk.idx[bone]
                ep = float(np.linalg.norm(mpos[k + 1, mid] - Pk[j, i]))
                ea = SKN.quat_angle_deg(mquat[k + 1, mid], Qk[j, i])
                consistency.append((t, bone, ep, ea))
            ih = sk.idx['hand_l']
            for bn, sg in curl_bodies.items():
                mid = m.body_mocapid[m.body(bn).id]
                if rig is not None:
                    pe, qe = CC.body_pose(rig, sg, Pk[j, ih], SKN.q2R(Qk[j, ih]), float(curl_k[j]))
                else:                                   # no visible curl: the fingers must be the flat hand
                    pe, qe = Pk[j, ih], Qk[j, ih]
                ep = float(np.linalg.norm(mpos[k + 1, mid] - pe))
                ea = SKN.quat_angle_deg(mquat[k + 1, mid], qe)
                consistency.append((t, bn, ep, ea))
        # hands: skin vertex speed, bone angular speed
        hv = {s: V[hand_ix[s]] for s in ('l', 'r')}
        hq = {s: Qk[j, sk.idx[f'hand_{s}']] for s in ('l', 'r')}
        if prev_hand is not None and j > 0:
            dtj = t - prev_hand[2]
            for s, wv in (('l', w_l), ('r', w_r)):
                v = float(np.linalg.norm(hv[s] - prev_hand[0][s], axis=1).max() / dtj)
                a = SKN.quat_angle_deg(hq[s], prev_hand[1][s])
                adapted = wv[j] > 0 or wv[j - frame_stride] > 0
                if s == 'l':
                    adapted = adapted or curl_k[j] > 0 or curl_k[j - frame_stride] > 0
                (hand_speed if adapted else rec_speed).append((t, s, v))
                (hand_ang if adapted else rec_ang).append((t, s, a / dtj))
                if adapted:
                    hand_jump.append((t, s, a))
        prev_hand = (hv, hq, t)
        for s, wv in (('l', w_l), ('r', w_r)):
            an = anat[s].angles(Qk[j])
            anat_rows.append((t, s, float(wv[j]), an))
        Vs = V[sub]
        if prevV is not None:
            still.append(float(np.linalg.norm(Vs - prevV, axis=1).max()) < 0.0003)
        prevV = Vs
    rr = max(robot_rows, key=lambda r: r[1]) if robot_rows else (None, 0.0, None)
    br = max(bag_rows, key=lambda r: r[1]) if bag_rows else (None, 0.0, None)
    res['a_robot_inside_skin_max_mm'] = rr[1] * 1000
    res['a_robot_inside_skin_worst'] = {'t': rr[0], 'geom_and_skin_bone': rr[2]}
    res['a_robot_frames_over_2mm'] = [round(r[0], 3) for r in robot_rows if r[1] > 0.002][:40]
    res['a_bag_inside_skin_max_mm'] = br[1] * 1000
    res['a_bag_inside_skin_worst'] = {'t': br[0], 'geom_and_skin_bone': br[2]}
    res['a_bag_frames_over_3mm'] = [round(r[0], 3) for r in bag_rows if r[1] > 0.003][:40]
    res['a_stopped_after_failure_at_s'] = dict(failed_at)     # robot / bag: None = every frame was checked
    ok_a = rr[1] <= 0.002 + 1e-9 and br[1] <= 0.003 + 1e-9
    # (b) physics-step contacts on the recipient
    bag_c = [c for c in coll.rec_contacts if c[3] == 'bag']
    rob_c = [c for c in coll.rec_contacts if c[3] in ('foot', 'finger_L', 'finger_R', 'umi_base', 'robot_other')]
    off_skin = []
    by_step = {}
    for c in bag_c + rob_c:
        by_step.setdefault(c[0], []).append(c)
    worst_c = 0.0
    worst_info = None
    rob_depth = 0.0
    cache_V = {}
    stopped_at = None
    for k, cs in sorted(by_step.items()):
        tkk = float(T[k])
        if not ok_a:
            # part a has failed, so G7 fails whatever part b finds: part b is not checked (each contact step costs a
            # skin surface build and thousands of sampled robot points)
            break
        if off_skin or rob_depth > 0.002 + 1e-9:
            # part b has failed: the rest cannot make it pass, and every further contact step costs a skin surface
            # build (minutes for a robot that leans on the person), so stop; the reported values are up to here
            stopped_at = tkk
            break
        P, Q, backed = skin_bones_at_step(m, mpos[k + 1], mquat[k + 1], rm_arr, tkk)
        V = curled_skin(sk, rig, P, Q, float(np.interp(tkk, tk, curl_k)))
        surf = sk.surface(V)
        pts = np.array([c[4] for c in cs])
        sd, _, _ = surf.signed_distance(pts, max_dist=0.05)
        for c, s_ in zip(cs, sd):
            if c[3] == 'bag':
                if abs(s_) > worst_c:
                    worst_c, worst_info = abs(float(s_)), (tkk, m.geom(c[1]).name, m.geom(c[2]).name, float(c[5]))
                if abs(s_) > 0.003:
                    off_skin.append((round(tkk, 4), m.geom(c[1]).name, round(float(s_) * 1000, 1), round(float(c[5]), 2)))
        if any(c[3] != 'bag' for c in cs):
            mj_d = d
            mj_d.qpos[:] = qpos[k]
            mj_d.mocap_pos[:] = mpos[k + 1]
            mj_d.mocap_quat[:] = mquat[k + 1]
            mj.mj_kinematics(m, mj_d)
            for c in cs:
                if c[3] == 'bag':
                    continue
                sdp, _, _ = surf.signed_distance(gs.world(mj_d, c[2]), max_dist=0.05)
                rob_depth = max(rob_depth, float(max(0.0, -sdp.min())))
    res['b_bag_recipient_contact_steps'] = len({c[0] for c in bag_c})
    res['b_bag_contact_max_offskin_mm'] = worst_c * 1000
    res['b_bag_contact_worst'] = worst_info
    res['b_bag_contacts_off_skin_first40'] = off_skin[:40]
    res['b_n_bag_contacts_off_skin'] = len(off_skin)
    res['b_robot_recipient_contact_steps'] = len({c[0] for c in rob_c})
    res['b_robot_inside_skin_at_contact_steps_max_mm'] = rob_depth * 1000
    res['b_stopped_after_failure_at_s'] = stopped_at      # None: every contact step was checked
    ok_b = (len(off_skin) == 0) and rob_depth <= 0.002 + 1e-9
    if not ok_a:
        ok_b = None
        res['b_not_checked'] = 'part a failed, so G7 fails; part b was not checked'
    # (c) colliders follow the bones
    res['curl'] = res_curl
    if consistency:
        res['c_collider_vs_json_bone_max_pos_mm'] = max(c[2] for c in consistency) * 1000
        res['c_collider_vs_json_bone_max_deg'] = max(c[3] for c in consistency)
        worst_c = max(consistency, key=lambda c: c[2] + c[3] * 1e-3)
        res['c_worst'] = {'t': worst_c[0], 'body_or_bone': worst_c[1], 'pos_mm': worst_c[2] * 1000, 'deg': worst_c[3]}
        ok_c = res['c_collider_vs_json_bone_max_pos_mm'] <= 0.5 and res['c_collider_vs_json_bone_max_deg'] <= 0.5
        if curl_bodies and rig is None:
            res['c_note'] = 'curl scene but recipient-motion.json has no curl rig: the fingers are not visible'
        res['c_applicable'] = True
    else:
        res['c_applicable'] = False
        res['c_note'] = 'no skin-matched recipient bodies (recipient_<bone>) in this scene: the colliders are proxies'
        ok_c = True
    # (d) hand motion (json) + physics-step hand mocap
    vmax = max(hand_speed, key=lambda r: r[2]) if hand_speed else (None, None, 0.0)
    amax = max(hand_ang, key=lambda r: r[2]) if hand_ang else (None, None, 0.0)
    jmax = max(hand_jump, key=lambda r: r[2]) if hand_jump else (None, None, 0.0)
    res['d_hand_skin_speed_max_mps'] = {'t': vmax[0], 'side': vmax[1], 'value': vmax[2]}
    res['d_hand_ang_speed_max_dps'] = {'t': amax[0], 'side': amax[1], 'value': amax[2]}
    res['d_hand_frame_jump_max_deg'] = {'t': jmax[0], 'side': jmax[1], 'value': jmax[2]}
    res['d_scope'] = ('frames where that arm is adapted (adaptation weight > 0 at either end of the frame pair); the '
                      'unadapted recording is the person\'s real motion and is only reported')
    res['d_recorded_hand_speed_max_mps'] = max((r[2] for r in rec_speed), default=0.0)
    res['d_recorded_hand_ang_speed_max_dps'] = max((r[2] for r in rec_ang), default=0.0)
    phys_ang = 0.0
    phys_v = 0.0
    for s in ('l', 'r'):
        try:
            mid = m.body_mocapid[m.body(f'recipient_hand_{s}').id]
        except KeyError:
            continue
        p = mpos[1:, mid]
        q = mquat[1:, mid]
        wv = w_l if s == 'l' else w_r
        wstep = np.interp(T[:len(p)], tk, wv) > 0
        ad = wstep[1:] | wstep[:-1]
        if not ad.any():
            continue
        phys_v = max(phys_v, float((np.linalg.norm(np.diff(p, axis=0), axis=1) / dt)[ad].max()))
        dots = np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1)
        phys_ang = max(phys_ang, float((np.degrees(2 * np.arccos(dots)) / dt)[ad].max()))
    res['d_physics_step_hand_bone_speed_max_mps'] = phys_v
    res['d_physics_step_hand_ang_speed_max_dps'] = phys_ang
    ok_d = vmax[2] <= 1.0 and amax[2] <= 300.0 and jmax[2] <= 12.0 and phys_v <= 1.0 and phys_ang <= 300.0
    # (e) anatomy on adapted frames
    lim = {'wrist_flex_deg': (-70, 70), 'wrist_dev_deg': (-30, 20), 'elbow_flex_deg': (0, 145)}
    viol = []
    ranges_ad, ranges_rec = {}, {}
    for t, s, w, an in anat_rows:
        tgt = ranges_ad if w > 0 else ranges_rec
        for kk, v in an.items():
            lo_hi = tgt.setdefault(f'{s}:{kk}', [v, v])
            lo_hi[0], lo_hi[1] = min(lo_hi[0], v), max(lo_hi[1], v)
        if w > 0:
            for kk, (lo, hi) in lim.items():
                if not (lo - 0.05 <= an[kk] <= hi + 0.05):     # 0.05 deg: json quaternions are rounded to 1e-7
                    viol.append((round(t, 3), s, kk, round(an[kk], 1)))
    res['e_limits_deg'] = lim
    res['e_adapted_ranges'] = {k: [round(a, 1), round(b, 1)] for k, (a, b) in ranges_ad.items()}
    res['e_recorded_ranges'] = {k: [round(a, 1), round(b, 1)] for k, (a, b) in ranges_rec.items()}
    res['e_violations_first40'] = viol[:40]
    res['e_n_violations'] = len(viol)
    ok_e = not viol
    # (f) freeze
    rs = runs(still)
    longest = max(((b - a) / 30.0 for a, b in rs), default=0.0) * frame_stride
    res['f_longest_full_body_freeze_s'] = longest
    res['f_freeze_intervals'] = [[round(float(tk[a + 1]), 3), round(float(tk[min(b, len(tk) - 1)]), 3)]
                                 for a, b in rs if (b - a) / 30.0 > 1.0][:20]
    ok_f = longest <= 1.0 + 1e-9
    # (g) static or slow while carrying load
    A = coll.arrays()
    loaded = A['fz_rec'] >= 0.25 * weight
    bad_g = []
    vmax_g = 0.0
    for (k, gr, go, ocl, pos, fn) in coll.rec_contacts:
        if ocl != 'bag' or not loaded[k] or k + 2 >= N:
            continue
        bid = int(m.geom_bodyid[gr])
        mid = m.body_mocapid[bid]
        p0, p1 = mpos[k + 1, mid], mpos[k + 2, mid]
        R0, R1 = SKN.q2R(mquat[k + 1, mid]), SKN.q2R(mquat[k + 2, mid])
        x = R0.T @ (pos - p0)
        v = (R1 @ x + p1 - pos) / dt
        sp = float(np.linalg.norm(v))
        vmax_g = max(vmax_g, sp)
        if sp > 0.002:
            ang = math.degrees(math.acos(min(1.0, abs(v[2]) / sp)))
            if sp > 0.02 or ang > 20.0:
                bad_g.append((round(float(T[k]), 4), m.geom(gr).name, round(sp, 4), round(ang, 1)))
    res['g_loaded_steps'] = int(loaded.sum())
    res['g_max_contact_point_speed_while_loaded_mps'] = vmax_g
    res['g_violations_first40'] = bad_g[:40]
    res['g_n_violations'] = len(bad_g)
    ok_g = not bad_g
    res['parts_pass'] = {'a_penetration': ok_a, 'b_contacts_on_visible_skin': ok_b, 'c_colliders_follow_bones': ok_c,
                         'd_hand_motion': ok_d, 'e_anatomy': ok_e, 'f_no_freeze': ok_f, 'g_load_static_or_slow': ok_g}
    res['pass'] = bool(all(res['parts_pass'].values()))
    return res


# ------------------------------------------------------------------------------------------------ G8
def g8_from_series(t, any_contact, net, net_cls, pad_fn, fz_rec, weight, t_pick, dt=0.002):
    """pure function on per-step series (unit-testable)."""
    res = {}
    if t_pick is None:
        res['applicable'] = False
        res['pass'] = True
        res['note'] = 'no G3 pickup: nothing to hand over'
        return res
    res['applicable'] = True
    k0 = int(np.searchsorted(t, t_pick))
    nc = ~any_contact[k0:]
    gaps = runs(nc)
    longest = max(((b - a) * dt for a, b in gaps), default=0.0)
    res['a_longest_no_contact_s'] = longest
    res['a_no_contact_intervals'] = [[round(float(t[k0 + a]), 4), round(float(t[min(k0 + b, len(t) - 1)]), 4)]
                                     for a, b in gaps if (b - a) * dt > 0.010][:20]
    ok_a = longest <= 0.010 + 1e-9
    peak_total = float(np.linalg.norm(net[k0:], axis=1).max()) if len(net) > k0 else 0.0
    peak_cls = {c: float(np.linalg.norm(v[k0:], axis=1).max()) for c, v in net_cls.items() if len(v) > k0}
    kpk = int(np.argmax(np.linalg.norm(net[k0:], axis=1))) + k0 if len(net) > k0 else k0
    res['b_peak_net_contact_force_N'] = peak_total
    res['b_peak_net_force_t'] = float(t[kpk])
    res['b_peak_net_force_by_class_N'] = peak_cls
    res['b_limit_N'] = 3 * weight
    ok_b = peak_total <= 3 * weight and all(v <= 3 * weight for v in peak_cls.values())
    # holding UMI = the side with the larger pad force at pickup
    side = 0 if pad_fn[k0, :2].sum() >= pad_fn[k0, 2:].sum() else 2
    pads = pad_fn[k0:, side:side + 2]
    lost = np.flatnonzero(pads.min(1) < 5.0)
    if len(lost):
        kl = k0 + int(lost[0])
        fz_before = float(fz_rec[max(kl - 1, 0)])
        res['c_pinch_lost_t'] = float(t[kl])
        res['c_recipient_support_before_N'] = fz_before
        res['c_recipient_support_frac_W'] = fz_before / weight
        ok_c = fz_before >= 0.6 * weight
    else:
        res['c_pinch_lost_t'] = None
        res['c_note'] = 'pinch never released'
        ok_c = True
    res['c_holding_umi'] = 'left' if side == 0 else 'right'
    res['parts_pass'] = {'a_contact_continuity': ok_a, 'b_peak_force': ok_b, 'c_pinch_until_60pct': ok_c}
    res['pass'] = bool(ok_a and ok_b and ok_c)
    return res


def _load_events(trial):
    ev = json.loads((trial / 'events.json').read_text()) if (trial / 'events.json').exists() else []
    return ev.get('events', ev) if isinstance(ev, dict) else ev


def _force_key(k):
    k = k.lower()
    return any(w in k for w in ('recipient', 'hook', 'rpad')) and any(f in k for f in ('force', '_n', 'support', 'load', 'weight'))


def event_uses_privileged_signals(e):
    """STRUCTURED (v7): True iff `signals` names a signal outside ROBOT_SIGNALS or a payload key names a
    recipient/hook/rpad force.  The free-text reason is never read."""
    sig = e.get('signals')
    if isinstance(sig, str):
        sig = [sig]
    if isinstance(sig, (list, tuple)) and any(str(x) not in ROBOT_SIGNALS for x in sig):
        return True
    return any(_force_key(k) for k in e if k not in ('time', 'event', 'reason', 'signals', 'privileged', 'decision'))


def privileged_check(trial):
    trial = Path(trial)
    ev = _load_events(trial)
    summ = json.loads((trial / 'summary.json').read_text()) if (trial / 'summary.json').exists() else {}
    flagged, undeclared = [], []
    for e in ev:
        if event_uses_privileged_signals(e) or e.get('privileged') is True:
            row = {'t': e.get('time'), 'event': e.get('event'), 'signals': e.get('signals'),
                   'reason': str(e.get('reason', ''))[:160]}
            (flagged if e.get('privileged') is True else undeclared).append(row)
    disc = ' '.join(map(str, summ.get('disclosures', []))).lower()
    disclosed = ('privileged' in disc) and ('force' in disc or 'load' in disc)
    ok = not undeclared and (not flagged or disclosed)
    return {'method': 'structured: event.signals outside ROBOT_SIGNALS or recipient-force payload keys; reasons not read',
            'events_using_recipient_forces_privileged': flagged[:20], 'events_using_recipient_forces_UNDECLARED': undeclared[:20],
            'summary_discloses_privileged_force_use': disclosed, 'pass': bool(ok)}


def release_provenance(trial, t_pick, holding='left', disclosed=False):
    """G8e: the robot's release decision uses robot-side sensing (event `signals` subset of ROBOT_SIGNALS) or is
    logged privileged (event privileged: true + summary disclosure)."""
    trial = Path(trial)
    if t_pick is None:
        return {'applicable': False, 'pass': True, 'note': 'no pickup'}
    lg = np.load(trial / 'controller-log.npz', allow_pickle=True)
    lt, ft = lg['time'], lg['finger_target']
    cols = (0, 1) if holding == 'left' else (2, 3)
    k0 = int(np.searchsorted(lt, t_pick))
    hold = ft[k0:, cols].mean(1)
    if not len(hold):
        return {'applicable': False, 'pass': True, 'note': 'no ticks after pickup'}
    ref = np.maximum.accumulate(hold)
    op = np.flatnonzero(hold < ref - 0.002)
    if not len(op):
        return {'applicable': False, 'pass': True, 'note': 'the holding UMI never opened its fingers'}
    t_open = float(lt[k0 + op[0]])
    ev = _load_events(trial)
    # v7: STRUCTURED decision events (event name or decision field); phase events and reasons are not used
    cand = [e for e in ev if t_open - 1.0 <= float(e.get('time', -1)) <= t_open + 0.021 and
            (str(e.get('event', '')).lower() in RELEASE_EVENTS or str(e.get('decision', '')).lower() == 'release')]
    rows, ok = [], bool(cand)
    for e in cand:
        sig = e.get('signals')
        sig = [sig] if isinstance(sig, str) else sig
        robot_only = isinstance(sig, list) and len(sig) > 0 and set(map(str, sig)) <= ROBOT_SIGNALS
        priv = e.get('privileged') is True
        good = robot_only or (priv and disclosed)
        ok &= good
        rows.append({'t': e.get('time'), 'event': e.get('event'), 'signals': sig, 'privileged': priv,
                     'robot_side_only': robot_only, 'ok': good})
    return {'applicable': True, 't_open_command_s': t_open, 'decision_events': rows[:10],
            'note': None if cand else 'no release/open decision event logged before the opening command',
            'decision_event_names': sorted(RELEASE_EVENTS), 'allowed_robot_signals': sorted(ROBOT_SIGNALS), 'pass': bool(ok)}


def gate_g8(trial, tr, coll: StepCollector, weight, t_pick):
    A = coll.arrays()
    t = tr['time'][:len(A['net'])]
    res = g8_from_series(t, A['any_contact'], A['net'], A['net_cls'], A['pad_fn'], A['fz_rec'], weight, t_pick)
    fb = A.get('fz_rec_body')
    res['fz_recipient_body_max_N'] = float(fb.max()) if fb is not None and len(fb) else 0.0
    res['c_note_v7'] = ("recipient support for G8c = hand/forearm colliders only (rpad); torso/leg support is "
                        "reported (fz_recipient_body_max_N), not counted")
    pc = privileged_check(trial)
    res['d_privileged'] = pc
    res.setdefault('parts_pass', {})['d_privileged_logged'] = pc['pass']
    rp = release_provenance(trial, t_pick, res.get('c_holding_umi', 'left'), pc['summary_discloses_privileged_force_use'])
    res['e_release_provenance'] = rp
    res['parts_pass']['e_release_robot_sensing_or_privileged'] = rp['pass']
    res['pass'] = bool(res['pass'] and pc['pass'] and rp['pass'])
    return res


# ------------------------------------------------------------------------------------------------ G9
def gate_g9(trial, tr, coll: StepCollector, t_hc, dt=0.002, t_pick=None, t_open=None):
    trial = Path(trial)
    res = {}
    lg = np.load(trial / 'controller-log.npz', allow_pickle=True)
    qt = lg['q_target'][:, 15:29]
    lt = lg['time']
    dq = np.abs(np.diff(qt, axis=0))
    res['a_max_arm_target_jump_rad'] = float(dq.max()) if len(dq) else 0.0
    if len(dq):
        i, j = np.unravel_index(int(np.argmax(dq)), dq.shape)
        res['a_worst'] = {'t': float(lt[i + 1]), 'joint_index_in_arm14': int(j)}
    res['a_ticks_over_0.25rad'] = [round(float(lt[i + 1]), 3) for i in np.flatnonzero(dq.max(1) > 0.25)][:30]
    ok_a = res['a_max_arm_target_jump_rad'] <= 0.25
    dqr = dq[:, 7:14] if len(dq) else np.zeros((0, 7))
    res['c_max_right_arm_target_step_rad'] = float(dqr.max()) if len(dqr) else 0.0
    res['c_right_ticks_over_0.05rad'] = [round(float(lt[i + 1]), 3) for i in np.flatnonzero(dqr.max(1) > 0.05)][:30] if len(dqr) else []
    ok_c = res['c_max_right_arm_target_step_rad'] <= 0.05
    A = coll.arrays()
    tool = A['tool']
    t = tr['time'][:len(tool)]
    pads = A['pad_fn']
    first = np.flatnonzero(pads.sum(1) > EPS_N)
    if len(first):
        t_fc = float(t[first[0]])
        t0 = t_fc - 0.5
        t1 = t_hc if t_hc is not None else float(t[-1])
        t_lc = float(t[first[-1]])
        # grasp window: 0.5 s before the first finger-bag contact .. 0.5 s after lift-off (G3 pickup start, else +1 s)
        g_end = (t_pick + 0.5) if t_pick is not None else t_fc + 1.0
        # release window: 0.5 s before the opening command (G8e) .. 0.5 s after the last finger-bag contact
        r_beg = (t_open - 0.5) if t_open is not None else None
        n = len(tool)
        vL = np.linalg.norm(np.diff(tool[:, :3], axis=0), axis=1) / dt
        vR = np.linalg.norm(np.diff(tool[:, 3:], axis=0), axis=1) / dt

        def ma(x):     # 5-step (10 ms) moving average removes single-step contact jitter of the measured site
            return np.convolve(x, np.ones(5) / 5, mode='same')
        vL, vR = ma(vL), ma(vR)
        ts = t[1:]
        sel = (ts >= t0) & (ts <= t1)
        win = (ts >= t0) & (ts <= g_end)
        if r_beg is not None:
            win |= (ts >= r_beg) & (ts <= max(t_lc, r_beg) + 0.5)
        # bag-recipient contact region: all physics-step contact points between the bag and a recipient collider
        reg = np.array([c[4] for c in coll.rec_contacts if c[3] == 'bag']).reshape(-1, 3)
        from scipy.spatial import cKDTree
        near = {}
        for side, cols in (('left', slice(0, 3)), ('right', slice(3, 6))):
            P = tool[1:, cols]
            if len(reg):
                dmin = cKDTree(reg).query(P)[0]
            else:                                   # no bag-recipient contact: his hand bodies at each step
                H = coll.rec_hand_pts[1:n]
                dmin = np.array([np.min(np.linalg.norm(h - p, axis=1)) if len(h) else np.inf for h, p in zip(H, P)]) \
                    if len(H) == len(P) else np.full(len(P), np.inf)
            near[side] = dmin <= G9B_NEAR_RADIUS
        res['b_window_s'] = [t0, t1]
        res['b_grasp_window_s'] = [t0, g_end]
        res['b_release_window_s'] = [r_beg, max(t_lc, r_beg) + 0.5] if r_beg is not None else None
        res['b_contact_region'] = {'n_points': int(len(reg)), 'radius_m': G9B_NEAR_RADIUS,
                                   'source': 'bag-recipient contact points' if len(reg) else
                                   'recipient hand bodies (no bag-recipient contact in the trial)'}
        res['b_limits_mps'] = {'near_region_or_grasp_release': G9B_NEAR_LIMIT, 'elsewhere': G9B_FAR_LIMIT}
        res['b_max_umi_speed_mps'] = {}
        res['b_max_umi_speed_strict_zone_mps'] = {}
        res['b_max_umi_speed_elsewhere_mps'] = {}
        worst, ok_b = None, True
        for side, v_ in (('left', vL), ('right', vR)):
            strict = sel & (win | near[side])
            loose = sel & ~strict
            vs = float(v_[strict].max()) if strict.any() else 0.0
            vl = float(v_[loose].max()) if loose.any() else 0.0
            res['b_max_umi_speed_mps'][side] = float(v_[sel].max()) if sel.any() else 0.0
            res['b_max_umi_speed_strict_zone_mps'][side] = vs
            res['b_max_umi_speed_elsewhere_mps'][side] = vl
            if sel.any():
                excess = np.where(strict, v_ - G9B_NEAR_LIMIT, v_ - G9B_FAR_LIMIT)
                excess[~sel] = -np.inf
                k = int(np.argmax(excess))
                if worst is None or excess[k] > worst[0]:
                    worst = (float(excess[k]), side, float(ts[k]), bool(strict[k]), float(v_[k]))
            ok_b &= vs <= G9B_NEAR_LIMIT + 1e-9 and vl <= G9B_FAR_LIMIT + 1e-9
        if worst is not None:
            res['b_worst'] = {'side': worst[1], 't': worst[2], 'speed_mps': worst[4],
                              'limit_mps': G9B_NEAR_LIMIT if worst[3] else G9B_FAR_LIMIT, 'strict_zone': worst[3]}
            res['b_worst_t'] = worst[2]
    else:
        res['b_note'] = 'no finger-bag contact'
        ok_b = True
    res['parts_pass'] = {'a_arm_target_jumps': ok_a, 'b_umi_speed': ok_b, 'c_right_arm_ramped': ok_c}
    res['pass'] = bool(ok_a and ok_b and ok_c)
    return res
