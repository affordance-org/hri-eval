#!/usr/bin/env python3
"""Independent checker for one HRI Eval trial folder (definitions and thresholds: docs/scoring.md).

usage: python hri_eval/check_gates.py TRIAL_DIR [--stride 5] [--frame-stride 1] [--reference-xml FILE] [--v4-only]
                                      [--out FILE]

It loads TRIAL_DIR/trial.xml, starts from row 0 of trajectory.npz and re-simulates the whole episode with only the
recorded motor inputs (ctrl) and the recorded person poses (mocap_pos, mocap_quat). Contacts, forces, distances and
poses come from that re-simulation, never from the policy or the runner's telemetry. It writes gates.json into
TRIAL_DIR (or FILE with --out), prints a pass/fail summary and the details of every failing gate, and exits with 0 only
if all gates pass.

  G0_invariants           an unmodified closed-loop episode: no state, force or model writes, plain torque motors at
                          the URDF limits, robot, bag and physics equal to the reference scene (default
                          hri_eval/scene/room_point_fetch.xml, the scene run.py simulates), start = keyframe 'start'
                          with the seeded variation of variation.json recomputed from its seed
  G1_motor_replay         the re-simulated qpos equals the recorded qpos to < 1e-6 at every physics step
  G2_no_fall              pelvis >= 0.55 m, torso tilt < 25 deg, only allowed contacts            (score: no fall)
  G3_pickup               bag >= 5 cm up, held only by the finger pads, for >= 0.5 s             (score: pickup)
  G4_carry                while held by the pads only: bag moves >= 0.4 m, pelvis >= 0.3 m      (score: carry)
  G5_handover             the person alone holds the bag for the last >= 1.0 s                   (score: handover)
  G6_pace                 handover complete within 3.0 x 4.4 s = 13.2 s                          (score: pace)
  G7_visual_honesty       the simulated person colliders match the visible skin
  G8_controlled_handover  no drop or throw, peak force, pinch until taken, privileged and release events logged
  G9_smoothness           arm targets and gripper speeds are smooth

--stride N analyses every Nth physics step for G2 to G6, --frame-stride N every Nth 30 fps frame in G7, --v4-only
skips G7 to G9.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gates_v7 as G5V  # noqa: E402  (v7 gates; module alias kept)
REFERENCE_XML = HERE / 'scene' / 'room_point_fetch.xml'     # the scene run.py simulates
URDF_CAPS = {'hip_pitch': 88.0, 'hip_roll': 139.0, 'hip_yaw': 88.0, 'knee': 139.0, 'ankle_pitch': 35.0,
             'ankle_roll': 35.0, 'waist_yaw': 88.0, 'waist_roll': 35.0, 'waist_pitch': 35.0, 'shoulder_pitch': 25.0,
             'shoulder_roll': 25.0, 'shoulder_yaw': 25.0, 'elbow': 25.0, 'wrist_roll': 25.0, 'wrist_pitch': 5.0,
             'wrist_yaw': 5.0}
JOINTS_29 = [f'{s}_{j}_joint' for s in ('left', 'right') for j in
             ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')] + \
            ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'] + \
            [f'{s}_{j}_joint' for s in ('left', 'right') for j in
             ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
FINGERS = ['left_umi_left_finger_joint', 'left_umi_right_finger_joint', 'right_umi_left_finger_joint',
           'right_umi_right_finger_joint']
STRAP_JOINTS = ['bag_strap_anchor', 'bag_strap_bend']     # v3-v7 names (kept for importers; v9 G0 uses bag_joints())
RPAD_BODIES = ('recipient_palm_pad', 'recipient_thumb_pad', 'recipient_hook_hand')
SKIN_RPAD = tuple(f'recipient_{b}' for b in ('upperarm_l', 'lowerarm_l', 'hand_l', 'upperarm_r', 'lowerarm_r', 'hand_r'))
sys_path_added = False


def bag_subtree(m, root='backpack'):
    r0 = m.body(root).id
    out = set()
    for b in range(m.nbody):
        r = b
        while r > 0 and r != r0:
            r = m.body_parentid[r]
        if r == r0:
            out.add(b)
    return out


def bag_joints(m):
    """v9: names of the joints in the bag subtree (v9: backpack_free; v3-v7 + bag_strap_anchor, bag_strap_bend)."""
    bs = bag_subtree(m)
    return [m.joint(j).name for j in range(m.njnt) if int(m.jnt_bodyid[j]) in bs]


def bag_sites(m):
    bs = bag_subtree(m)
    return [m.site(s).name for s in range(m.nsite) if int(m.site_bodyid[s]) in bs]


def handle_geoms(m, bag_bodies, bag):
    """v9: G5 clearance reference.  Strap links' collision geoms if the bag has strap links (v3-v7), else every
    collision geom of the bag body (v9: no handle, no strap)."""
    coll = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bag_bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]
    strap = [g for g in coll if int(m.geom_bodyid[g]) != bag]
    return (strap, 'strap links (bag subtree minus the bag body)') if strap else (coll, 'every bag-body collision geom (v9: no handle, no strap)')
G = 9.81
BAG_MASS = 1.2
WEIGHT = BAG_MASS * G
EPS_N = 0.01
HUMAN_DURATION_S = 4.4
PACE_FACTOR = 3.0              # v7 (v6: 2.5)
MAX_PRE_ROLL = 1.0


def model_fingerprint(m):
    """Same digest as episode.model_fingerprint (kept standalone so the checker does not import the runner)."""
    h = hashlib.sha256()
    for f in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_gravcomp', 'jnt_range', 'jnt_actfrcrange',
              'jnt_stiffness', 'dof_damping', 'dof_armature', 'dof_frictionloss', 'actuator_ctrlrange',
              'actuator_forcerange', 'actuator_gainprm', 'actuator_biasprm', 'actuator_gear', 'geom_friction',
              'geom_solref', 'geom_solimp', 'geom_size', 'geom_contype', 'geom_conaffinity', 'geom_margin', 'eq_active0',
              'jnt_actgravcomp', 'body_pos', 'body_quat'):
        h.update(np.ascontiguousarray(getattr(m, f)).tobytes())
    for f in ('timestep', 'impratio', 'noslip_iterations', 'iterations', 'integrator', 'cone', 'gravity', 'disableflags',
              'enableflags', 'o_margin', 'o_solref', 'o_solimp'):
        h.update(np.ascontiguousarray(getattr(m.opt, f)).tobytes())
    return h.hexdigest()


def cap_of(j):
    return URDF_CAPS[j.replace('left_', '').replace('right_', '').replace('_joint', '')]


def tilt_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))


def runs(mask):
    """[(start, end_exclusive)] of consecutive True."""
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


class Classes:
    def __init__(self, m):
        self.m = m
        pelvis = m.body('pelvis').id
        self.robot = set()
        for b in range(m.nbody):
            r = b
            while r > 0:
                if r == pelvis:
                    self.robot.add(b)
                    break
                r = m.body_parentid[r]
        self.bag = m.body('backpack').id
        self.bag_bodies = bag_subtree(m)
        self.body_cls = {}
        names = {m.body(b).name for b in range(m.nbody)}
        rpad_names = set(RPAD_BODIES) | (set(SKIN_RPAD) if 'recipient_hand_l' in names else set()) | \
            {n for n in names if n.startswith('recipient_hand_l_')}     # v6 curl scene: finger segment bodies
        for b in range(m.nbody):
            n = m.body(b).name
            if b in self.robot:
                if n in ('left_ankle_roll_link', 'right_ankle_roll_link'):
                    c = 'foot'
                elif n.startswith('left_umi_') and 'finger' in n:
                    c = 'finger_L'
                elif n.startswith('right_umi_') and 'finger' in n:
                    c = 'finger_R'
                elif n.startswith('left_umi') or n.startswith('right_umi'):
                    c = 'umi_base'
                else:
                    c = 'robot_other'
            elif b in self.bag_bodies:
                c = 'bag'
            elif n in rpad_names:
                c = 'rpad'
            elif n.startswith('recipient_') and n.replace('recipient_', '') in G5V.BODY_BONES:
                c = 'recipient_body'             # v7 torso / pelvis / leg colliders
            elif n.startswith('recipient_'):
                c = 'recipient_other'
            elif b == 0:
                c = 'world'
            else:
                c = 'other_body'
            self.body_cls[b] = c

    def geom_cls(self, g):
        b = int(self.m.geom_bodyid[g])
        c = self.body_cls[b]
        if c == 'world':
            n = self.m.geom(g).name
            if n == 'floor':
                return 'floor'
            if n.startswith('table_'):
                return 'table'
            return 'world_other'
        return c


ROBOT_CLS = {'foot', 'finger_L', 'finger_R', 'umi_base', 'robot_other'}
UMI_CLS = {'finger_L', 'finger_R', 'umi_base'}
WORLD_CLS = {'floor', 'table', 'world_other', 'other_body'}


def model_checks(m, ref, ref_key=None):
    """ref_key: the expected keyframe 'start' qpos (default: the reference's; run.py variations: the reference's with
    the seeded variation applied)."""
    out = {}
    mocap = [m.body(b).name for b in range(m.nbody) if m.body_mocapid[b] >= 0]
    out['neq'] = int(m.neq)
    out['mocap_bodies'] = mocap
    out['mocap_only_recipient'] = all(n.startswith('recipient_') for n in mocap)
    out['mocap_world_children_without_children'] = all(
        m.body_parentid[m.body(n).id] == 0 and not any(m.body_parentid[c] == m.body(n).id for c in range(m.nbody))
        for n in mocap)
    out['gravcomp_bodies'] = int(np.count_nonzero(m.body_gravcomp))
    out['actuator_gravcomp_joints'] = int(np.count_nonzero(m.jnt_actgravcomp))
    out['options_ok'] = bool(abs(m.opt.timestep - 0.002) < 1e-12 and m.opt.integrator == mj.mjtIntegrator.mjINT_IMPLICITFAST
                             and m.opt.cone == mj.mjtCone.mjCONE_ELLIPTIC and abs(m.opt.impratio - 10) < 1e-12
                             and m.opt.noslip_iterations == 0 and np.allclose(m.opt.gravity, [0, 0, -9.81]))
    act_ok = True
    caps = []
    for j in JOINTS_29:
        jid = m.joint(j).id
        acts = [a for a in range(m.nu) if m.actuator_trnid[a, 0] == jid and m.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT]
        cap = cap_of(j)
        caps.append(cap)
        if len(acts) != 1:
            act_ok = False
            continue
        a = acts[0]
        act_ok &= bool(m.actuator_gaintype[a] == mj.mjtGain.mjGAIN_FIXED and abs(m.actuator_gainprm[a, 0] - 1) < 1e-12
                       and m.actuator_biastype[a] == mj.mjtBias.mjBIAS_NONE and abs(m.actuator_gear[a, 0] - 1) < 1e-12
                       and np.allclose(m.actuator_ctrlrange[a], [-cap, cap]) and np.allclose(m.actuator_forcerange[a], [-cap, cap])
                       and m.actuator_dyntype[a] == mj.mjtDyn.mjDYN_NONE)
    for f in FINGERS:
        jid = m.joint(f).id
        acts = [a for a in range(m.nu) if m.actuator_trnid[a, 0] == jid]
        act_ok &= len(acts) == 1 and bool(np.all(np.abs(m.actuator_forcerange[acts[0]]) <= 35.0 + 1e-9)) and \
            bool(m.actuator_forcelimited[acts[0]])
    out['actuators_ok'] = act_ok
    out['nu_expected'] = bool(m.nu == 33)
    # physical parameters of robot + bag identical to the harness reference scene (by name)
    diffs = []
    if ref is not None:
        def bodies(mm):
            pel = mm.body('pelvis').id
            res = []
            for b in range(mm.nbody):
                r = b
                while r > 0 and r != pel:
                    r = mm.body_parentid[r]
                rb = b
                while rb > 0 and mm.body(rb).name != 'backpack':
                    rb = mm.body_parentid[rb]
                if r == pel or rb > 0:
                    res.append(mm.body(b).name)
            return res
        for n in bodies(ref):
            try:
                b, br = m.body(n), ref.body(n)
            except KeyError:
                diffs.append(f'missing body {n}')
                continue
            for f in ('mass', 'inertia', 'ipos', 'iquat', 'pos', 'quat'):
                if not np.allclose(getattr(b, f), getattr(br, f), rtol=0, atol=1e-12):
                    diffs.append(f'body {n}.{f}')
            if m.body_gravcomp[b.id] != ref.body_gravcomp[br.id]:
                diffs.append(f'body {n}.gravcomp')
            gm = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b.id]
            gr = [g for g in range(ref.ngeom) if ref.geom_bodyid[g] == br.id]
            if len(gm) != len(gr):
                diffs.append(f'body {n} geom count')
                continue
            for g1, g2 in zip(gm, gr):
                for f in ('geom_size', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_condim', 'geom_contype',
                          'geom_conaffinity', 'geom_pos', 'geom_quat', 'geom_type', 'geom_margin', 'geom_gap', 'geom_priority'):
                    if not np.allclose(getattr(m, f)[g1], getattr(ref, f)[g2], rtol=0, atol=1e-12):
                        diffs.append(f'geom {g1} of {n}: {f}')
        bj_ref, bj = bag_joints(ref), bag_joints(m)
        if sorted(bj_ref) != sorted(bj):                    # v9: bag joints of the REFERENCE, by name (listed first)
            diffs.insert(0, f'bag joints {sorted(bj)} != reference {sorted(bj_ref)}')
        for s_ in bag_sites(ref):
            try:
                a_, b_ = m.site(s_), ref.site(s_)
            except KeyError:
                diffs.append(f'missing bag site {s_}')
                continue
            if m.body(int(m.site_bodyid[a_.id])).name != ref.body(int(ref.site_bodyid[b_.id])).name:
                diffs.append(f'bag site {s_}.body')
            if not (np.allclose(a_.pos, b_.pos, rtol=0, atol=1e-12) and np.allclose(a_.quat, b_.quat, rtol=0, atol=1e-12)):
                diffs.append(f'bag site {s_}.pose')
        for j in JOINTS_29 + FINGERS + bj_ref:
            try:
                a, b = m.joint(j), ref.joint(j)
            except KeyError:
                diffs.append(f'missing joint {j}')
                continue
            if m.jnt_type[a.id] != ref.jnt_type[b.id]:
                diffs.append(f'joint {j}.type')
                continue
            for f in ('range', 'armature', 'damping', 'frictionloss', 'stiffness', 'solref', 'solimp', 'limited'):
                if not np.allclose(getattr(a, f), getattr(b, f)):
                    diffs.append(f'joint {j}.{f}')
        for j in FINGERS:
            aa = [x for x in range(m.nu) if m.actuator_trnid[x, 0] == m.joint(j).id]
            ab = [x for x in range(ref.nu) if ref.actuator_trnid[x, 0] == ref.joint(j).id]
            for f in ('actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange'):
                if len(aa) != 1 or len(ab) != 1 or not np.allclose(getattr(m, f)[aa[0]], getattr(ref, f)[ab[0]]):
                    diffs.append(f'finger actuator {j}.{f}')
        for k in ('table_top',):
            if not (np.allclose(m.geom(k).pos, ref.geom(k).pos) and np.allclose(m.geom(k).size, ref.geom(k).size)):
                diffs.append(f'geom {k}')
        if m.nq != ref.nq:                                  # v9: a strap bag and a strapless bag differ in nq
            diffs.insert(0, f'keyframe start qpos: nq {m.nq} != reference {ref.nq}')
        elif not np.allclose(m.key('start').qpos, ref.key('start').qpos if ref_key is None else ref_key, rtol=0, atol=1e-12):
            diffs.append('keyframe start qpos')
    out['physical_param_diffs_vs_reference'] = diffs[:30]
    out['physical_params_match_reference'] = not diffs
    return out, np.array(caps)


def check(trial_dir, stride=5, write=True, reference_xml=REFERENCE_XML, verbose=False, v5=True, frame_stride=1):
    trial = Path(trial_dir)
    res = {'trial': str(trial), 'gates': {}, 'checker': 'hri_eval/check_gates.py', 'mujoco': mj.__version__,
           'reference_xml': str(reference_xml),
           'stride_steps': stride, 'eps_contact_N': EPS_N, 'bag_weight_N': WEIGHT}
    m = mj.MjModel.from_xml_path(str(trial / 'trial.xml'))
    ref = mj.MjModel.from_xml_path(str(reference_xml)) if Path(reference_xml).exists() else None
    tr = np.load(trial / 'trajectory.npz')
    t, qpos, qvel, ctrl = tr['time'], tr['qpos'], tr['qvel'], tr['ctrl']
    mpos, mquat = tr['mocap_pos'], tr['mocap_quat']
    act = tr['act'] if 'act' in tr.files else np.zeros((len(t), 0))
    summ = json.loads((trial / 'summary.json').read_text()) if (trial / 'summary.json').exists() else {}
    res['trajectory_sha256'] = hashlib.sha256((trial / 'trajectory.npz').read_bytes()).hexdigest()
    res['trial_xml_sha256'] = hashlib.sha256((trial / 'trial.xml').read_bytes()).hexdigest()
    N = len(t)
    dt = m.opt.timestep

    # ------------------------------------------------------------------------------------------------ G0
    # seeded variation (run.py --seed, variations.py): recomputed here from the seed, never taken from the trial
    var_ok, ref_key, var = True, None, None
    if (trial / 'variation.json').exists():
        import variations as V
        stored = json.loads((trial / 'variation.json').read_text())
        var = V.sample(stored['seed'])
        var_ok = all(abs(float(stored.get(k, np.nan)) - var[k]) < 1e-12 for k in V.KEYS)
        if ref is not None:
            ref_key = V.apply_key(ref.key('start').qpos, ref, var)
    mc, caps = model_checks(m, ref, ref_key)
    g0 = dict(mc)
    g0['variation'] = var
    g0['variation_matches_seed'] = var_ok
    motor_ids = [[a for a in range(m.nu) if m.actuator_trnid[a, 0] == m.joint(j).id][0] for j in JOINTS_29]
    ratio = np.abs(ctrl[:, motor_ids]) / caps
    g0['max_recorded_ctrl_over_cap'] = float(ratio.max()) if N else 0.0
    dadr = [m.jnt_dofadr[m.joint(j).id] for j in JOINTS_29]
    g0['max_recorded_qfrc_actuator_over_cap'] = float((np.abs(tr['qfrc_actuator'][:, dadr]) / caps).max()) if N else 0.0
    g0['time_continuous'] = bool(N > 1 and abs(t[0]) < 1e-12 and np.allclose(np.diff(t), dt, rtol=0, atol=1e-9))
    g0['finite'] = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)) and np.all(np.isfinite(ctrl)))
    key = m.key('start')
    g0['initial_state_is_keyframe_start'] = bool(np.allclose(qpos[0], key.qpos, rtol=0, atol=1e-12) and not np.any(qvel[0]))
    inv = summ.get('invariants_measured', {})
    g0['summary_present'] = bool(summ)
    g0['summary_closed_loop_full_episode'] = summ.get('controller_execution') == 'closed_loop_full_episode'
    g0['summary_not_diagnostic'] = summ.get('diagnostic_only') is False
    g0['summary_qpos_qvel_writes_after_init'] = inv.get('qpos_qvel_writes_after_init')
    g0['summary_max_abs_xfrc_applied'] = inv.get('max_abs_xfrc_applied')
    g0['summary_max_abs_qfrc_applied'] = inv.get('max_abs_qfrc_applied')
    g0['summary_model_changed_ticks'] = inv.get('model_changed_during_run_ticks')
    fp = model_fingerprint(m)
    g0['model_fingerprint_sha256'] = fp
    g0['model_fingerprint_matches_summary'] = (inv.get('model_fingerprint_sha256') == fp) if inv else False
    cond = [g0['neq'] == 0, g0['mocap_only_recipient'], g0['mocap_world_children_without_children'],
            g0['gravcomp_bodies'] == 0, g0['actuator_gravcomp_joints'] == 0, g0['options_ok'], g0['actuators_ok'],
            g0['physical_params_match_reference'], g0['max_recorded_ctrl_over_cap'] <= 1 + 1e-9,
            g0['max_recorded_qfrc_actuator_over_cap'] <= 1 + 1e-6, g0['time_continuous'], g0['finite'],
            g0['initial_state_is_keyframe_start'], g0['summary_closed_loop_full_episode'], g0['summary_not_diagnostic'],
            g0['summary_qpos_qvel_writes_after_init'] == 0, g0['summary_max_abs_xfrc_applied'] == 0,
            g0['summary_max_abs_qfrc_applied'] == 0, g0['summary_model_changed_ticks'] == 0,
            g0['model_fingerprint_matches_summary'], g0['variation_matches_seed']]
    g0['pass'] = bool(all(cond))
    res['gates']['G0_invariants'] = g0

    # ------------------------------------------------------------------------------------------------ replay
    C = Classes(m)
    d = mj.MjData(m)
    d.qpos[:] = qpos[0]
    d.qvel[:] = qvel[0]
    if m.na:
        d.act[:] = act[0]
    d.mocap_pos[:] = mpos[0]
    d.mocap_quat[:] = mquat[0]
    d.ctrl[:] = ctrl[0]
    d.time = t[0]
    mj.mj_forward(m, d)
    pelvis, torso, bag = m.body('pelvis').id, m.body('torso_link').id, C.bag
    fj = m.body('pelvis').jntadr[0]
    fq, fd = m.jnt_qposadr[fj], m.jnt_dofadr[fj]
    strap_geoms, handle_ref = handle_geoms(m, C.bag_bodies, C.bag)        # v9: bag body when there is no strap
    bag_set = C.bag_bodies
    umi_geoms = [g for g in range(m.ngeom) if C.body_cls[int(m.geom_bodyid[g])] in UMI_CLS
                 and (m.geom_contype[g] or m.geom_conaffinity[g])]
    feet = [m.body('left_ankle_roll_link').id, m.body('right_ankle_roll_link').id]
    f6 = np.zeros(6)
    fromto = np.zeros(6)
    S = {k: [] for k in ('k', 't', 'pelvis', 'pelvis_vxy', 'torso_tilt', 'pelvis_tilt', 'bag', 'robot_only', 'recipient_only',
                         'bag_cls', 'fz_rpad', 'fz_fingers', 'robot_bag_fn', 'umi_handle_dist', 'viol', 'foot_fn',
                         'nonumi_umi_bag_fn')}
    max_qerr = max_verr = 0.0
    first_err_step = None

    def analyse(k):
        sup = {}
        fn_cls = {}
        viol = []
        foot_fn = [0.0, 0.0]
        robot_bag_fn = 0.0
        umi_base_bag = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            mj.mj_contactForce(m, d, i, f6)
            fn = float(f6[0])
            if fn <= EPS_N:
                continue
            c1, c2 = C.geom_cls(c.geom1), C.geom_cls(c.geom2)
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            F2 = c.frame.reshape(3, 3).T @ f6[:3]          # force on geom2 (world)
            in1, in2 = b1 in bag_set, b2 in bag_set
            if in1 and in2:
                continue                                   # internal: bag body <-> strap links
            if in1 or in2:
                other, Fb = (c1, F2) if in2 else (c2, -F2)
                sup[other] = sup.get(other, np.zeros(3)) + Fb
                fn_cls[other] = fn_cls.get(other, 0.0) + fn
                if other in ROBOT_CLS:
                    robot_bag_fn += fn
                    if other == 'umi_base':
                        umi_base_bag += fn
                        if fn >= 5.0:
                            viol.append(('umi_base-bag>=5N', fn))
                    elif other not in ('finger_L', 'finger_R'):
                        viol.append((f'{other}-bag', fn))
                continue
            for (ca, cb, ba) in ((c1, c2, b1), (c2, c1, b2)):
                if ca not in ROBOT_CLS:
                    continue
                if cb in ROBOT_CLS:
                    break                      # self contact: reported only
                if cb == 'floor':
                    if ca == 'foot':
                        foot_fn[feet.index(ba)] += fn
                    else:
                        viol.append((f'{ca}-floor', fn))
                elif cb == 'table':
                    if ca in UMI_CLS and fn < 5.0:
                        pass
                    else:
                        viol.append((f'{ca}-table', fn))
                elif cb in ('world_other', 'other_body'):
                    viol.append((f'{ca}-{cb}', fn))
                elif cb == 'rpad':
                    if not (ca in UMI_CLS and fn < 5.0):
                        viol.append((f'{ca}-recipient_pad', fn))
                elif cb == 'recipient_other':
                    viol.append((f'{ca}-recipient', fn))
                elif cb == 'recipient_body':
                    viol.append((f'{ca}-recipient_body', fn))
                break
        cls = set(sup)
        robot_only = bool(cls) and cls <= {'finger_L', 'finger_R'}
        recipient_only = bool(cls) and cls <= {'rpad', 'recipient_body'} and 'rpad' in cls     # v7
        dmin = -1.0                                        # only needed (and computed) while the recipient alone holds the bag
        if recipient_only:
            dmin = 1.0
            for hg in strap_geoms:
                for g in umi_geoms:
                    dmin = min(dmin, mj.mj_geomDistance(m, d, hg, g, 0.2, fromto))
        S['k'].append(k)
        S['t'].append(t[k])
        S['pelvis'].append(d.xpos[pelvis].copy())
        S['pelvis_vxy'].append(float(np.linalg.norm(qvel[k, fd:fd + 2])))
        S['torso_tilt'].append(tilt_deg(d.xmat[torso].reshape(3, 3)))
        S['pelvis_tilt'].append(tilt_deg(d.xmat[pelvis].reshape(3, 3)))
        S['bag'].append(d.xpos[bag].copy())
        S['robot_only'].append(robot_only)
        S['recipient_only'].append(recipient_only)
        S['bag_cls'].append(sorted(cls))
        S['fz_rpad'].append(float(sup.get('rpad', np.zeros(3))[2]))
        S['fz_fingers'].append(float(sup.get('finger_L', np.zeros(3))[2] + sup.get('finger_R', np.zeros(3))[2]))
        S['robot_bag_fn'].append(robot_bag_fn)
        S['umi_handle_dist'].append(dmin)
        S['viol'].append(viol)
        S['foot_fn'].append(foot_fn)
        S['nonumi_umi_bag_fn'].append(umi_base_bag)

    coll = G5V.StepCollector(m, C, WEIGHT) if v5 else None
    for i in range(1, N):
        d.ctrl[:] = ctrl[i]
        d.mocap_pos[:] = mpos[i]
        d.mocap_quat[:] = mquat[i]
        mj.mj_step(m, d)
        if coll is not None:
            coll.collect(d, i - 1)
        if (i - 1) % stride == 0:
            analyse(i - 1)
        e = float(np.max(np.abs(d.qpos - qpos[i])))
        ev = float(np.max(np.abs(d.qvel - qvel[i])))
        if e > max_qerr:
            max_qerr = e
        max_verr = max(max_verr, ev)
        if first_err_step is None and e >= 1e-6:
            first_err_step = i
    g1 = {'max_qpos_error': max_qerr, 'max_qvel_error': max_verr, 'first_step_qpos_error_ge_1e-6': first_err_step,
          'physics_steps': N - 1, 'pass': bool(N > 1 and max_qerr < 1e-6)}
    res['gates']['G1_motor_replay'] = g1

    # arrays
    ts = np.array(S['t'])
    P = np.array(S['pelvis'])
    BAG = np.array(S['bag'])
    torso_tilt = np.array(S['torso_tilt'])
    robot_only = np.array(S['robot_only'], bool)
    recipient_only = np.array(S['recipient_only'], bool)
    fz_rpad = np.array(S['fz_rpad'])
    robot_bag_fn = np.array(S['robot_bag_fn'])
    umi_d = np.array(S['umi_handle_dist'])
    foot_fn = np.array(S['foot_fn'])
    vxy = np.array(S['pelvis_vxy'])
    sdt = stride * dt
    n = len(ts)

    # ------------------------------------------------------------------------------------------------ G2
    crouch = np.zeros(n, bool)
    try:
        lg = np.load(trial / 'controller-log.npz')
        if 'cmd' in lg.files:
            lt, bh = lg['time'], lg['cmd'][:, 3]
            low = np.isfinite(bh) & (bh < 0.65)
            for k in range(n):
                sel = (lt <= ts[k] + 1e-9) & (lt >= ts[k] - 1.0)
                crouch[k] = bool(np.any(low[sel]))
    except Exception:  # noqa
        pass
    thr = np.where(crouch, 0.30, 0.55)
    low_pelvis = P[:, 2] < thr
    tilt_bad = torso_tilt >= 25.0
    viol_idx = [k for k in range(n) if S['viol'][k]]
    worst = {}
    for k in viol_idx:
        for name, fn in S['viol'][k]:
            if name not in worst or fn > worst[name][0]:
                worst[name] = (fn, float(ts[k]))
    g2 = {'min_pelvis_z': float(P[:, 2].min()) if n else None, 'pelvis_floor_violations': int(low_pelvis.sum()),
          'first_low_pelvis_t': float(ts[np.argmax(low_pelvis)]) if low_pelvis.any() else None,
          'commanded_crouch_samples': int(crouch.sum()),
          'max_torso_tilt_deg': float(torso_tilt.max()) if n else None,
          'max_pelvis_tilt_deg': float(np.max(S['pelvis_tilt'])) if n else None,
          'contact_violation_samples': len(viol_idx),
          'first_contact_violation_t': float(ts[viol_idx[0]]) if viol_idx else None,
          'contact_violations_worst': {k: {'max_normal_N': v[0], 't': v[1]} for k, v in worst.items()},
          'max_umi_table_and_umi_pad_allowance_N': 5.0}
    g2['pass'] = bool(n > 0 and not low_pelvis.any() and not tilt_bad.any() and not viol_idx)
    res['gates']['G2_no_fall'] = g2

    # ------------------------------------------------------------------------------------------------ G3
    bag_z0 = float(qpos[0, m.jnt_qposadr[m.body('backpack').jntadr[0]] + 2])
    lifted = (BAG[:, 2] - bag_z0) >= 0.05 if n else np.zeros(0, bool)
    good3 = lifted & robot_only
    r3 = runs(good3)
    best3 = max(r3, key=lambda r: r[1] - r[0]) if r3 else None
    dur3 = (best3[1] - best3[0]) * sdt if best3 else 0.0
    g3 = {'bag_rest_z': bag_z0, 'max_bag_lift_m': float((BAG[:, 2] - bag_z0).max()) if n else None,
          'longest_lifted_robot_pads_only_s': dur3,
          'interval': [float(ts[best3[0]]), float(ts[best3[1] - 1])] if best3 else None,
          'robot_pads_only_samples': int(robot_only.sum())}
    g3['pass'] = bool(dur3 >= 0.5 - 1e-9)
    t_pickup = float(ts[best3[0]] + 0.5) if g3['pass'] else None
    res['gates']['G3_pickup'] = g3

    # ------------------------------------------------------------------------------------------------ G4
    cands = []
    for a, b in runs(robot_only):
        cands.append({'interval': [float(ts[a]), float(ts[b - 1])],
                      'bag_horizontal_travel_m': float(np.max(np.linalg.norm(BAG[a:b, :2] - BAG[a, :2], axis=1))),
                      'pelvis_horizontal_travel_m': float(np.max(np.linalg.norm(P[a:b, :2] - P[a, :2], axis=1))),
                      'max_lift_m': float((BAG[a:b, 2] - bag_z0).max())})
    passing = [c for c in cands if c['bag_horizontal_travel_m'] >= 0.4 and c['pelvis_horizontal_travel_m'] >= 0.3]
    best4 = passing[0] if passing else (max(cands, key=lambda c: min(c['bag_horizontal_travel_m'] / 0.4,
                                                                      c['pelvis_horizontal_travel_m'] / 0.3)) if cands else None)
    g4 = dict(best4) if best4 else {'interval': None}
    g4['pass'] = bool(best4 is not None and best4['bag_horizontal_travel_m'] >= 0.4 and best4['pelvis_horizontal_travel_m'] >= 0.3)
    res['gates']['G4_carry'] = g4

    # ------------------------------------------------------------------------------------------------ G5
    final_ok = recipient_only & (robot_bag_fn <= EPS_N) & (umi_d >= 0.05)
    t_hc = None
    if n and final_ok[-1]:
        k0 = n - 1
        while k0 > 0 and final_ok[k0 - 1]:
            k0 -= 1
        t_hc = float(ts[k0])
    T_end = float(t[-1])
    g5 = {'t_handover_complete_s': t_hc, 'episode_end_s': T_end,
          'final_recipient_only_duration_s': (T_end - t_hc) if t_hc is not None else 0.0,
          'final_state_bag_support_classes': S['bag_cls'][-1] if n else None,
          'final_umi_handle_distance_m': float(umi_d[-1]) if n else None,
          'umi_clearance_reference': f'{handle_ref}: {len(strap_geoms)} geoms; -1 = not computed (bag not recipient-only)'}
    ok5 = t_hc is not None and (T_end - t_hc) >= 1.0 - 1e-9
    if t_hc is not None:
        sel = ts >= t_hc
        g5['mean_vertical_support_by_recipient_pads_N'] = float(fz_rpad[sel].mean())
        sel1 = ts >= T_end - 1.0
        g5['mean_vertical_support_last_1s_N'] = float(fz_rpad[sel1].mean())
        ok5 &= g5['mean_vertical_support_by_recipient_pads_N'] >= 0.8 * WEIGHT and g5['mean_vertical_support_last_1s_N'] >= 0.8 * WEIGHT
        g5['pickup_before_handover'] = bool(t_pickup is not None and t_pickup <= t_hc)
        ok5 &= g5['pickup_before_handover']
    last = ts >= T_end - 1.0
    stand_ok = bool(n and np.all(P[last, 2] >= 0.55) and np.all(torso_tilt[last] < 25.0) and np.all(vxy[last] < 0.5)
                    and not any(S['viol'][k] for k in np.flatnonzero(last)))
    g5['robot_standing_stable_last_1s'] = stand_ok
    g5['max_pelvis_speed_last_1s'] = float(vxy[last].max()) if n else None
    g5['pass'] = bool(ok5 and stand_ok)
    res['gates']['G5_handover'] = g5

    # ------------------------------------------------------------------------------------------------ G6
    moved = np.linalg.norm(P[:, :2] - P[0, :2], axis=1) > 0.10 if n else np.zeros(0, bool)
    unloaded = (foot_fn.min(axis=1) < 1.0) if n else np.zeros(0, bool)
    ms = moved | unloaded
    t_ms = float(ts[np.argmax(ms)]) if ms.any() else None
    pre = min(t_ms, MAX_PRE_ROLL) if t_ms is not None else None
    g6 = {'t_motion_start_s': t_ms, 'pre_roll_s': pre, 't_handover_complete_s': t_hc,
          'human_duration_s': HUMAN_DURATION_S, 'limit_s': PACE_FACTOR * HUMAN_DURATION_S}
    if t_hc is not None and pre is not None:
        g6['robot_duration_s'] = t_hc - pre
        g6['ratio_to_human'] = g6['robot_duration_s'] / HUMAN_DURATION_S
        g6['pass'] = bool(g6['robot_duration_s'] <= PACE_FACTOR * HUMAN_DURATION_S + 1e-9)
    else:
        g6['robot_duration_s'] = None
        g6['pass'] = False
    res['gates']['G6_pace'] = g6
    if v5:
        t_pick0 = float(ts[best3[0]]) if g3['pass'] else None
        res['gates']['G7_visual_honesty'] = G5V.gate_g7(trial, m, tr, coll, C, WEIGHT, frame_stride=frame_stride)
        res['gates']['G8_controlled_handover'] = G5V.gate_g8(trial, tr, coll, WEIGHT, t_pick0)
        t_open = res['gates']['G8_controlled_handover'].get('e_release_provenance', {}).get('t_open_command_s')
        res['gates']['G9_smoothness'] = G5V.gate_g9(trial, tr, coll, t_hc, t_pick=t_pick0, t_open=t_open)
    res['all_pass'] = bool(all(g['pass'] for g in res['gates'].values()))
    res['summary'] = {k: v['pass'] for k, v in res['gates'].items()}
    if write:
        (trial / 'gates.json').write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (
            o.item() if isinstance(o, np.generic) else str(o))) + '\n')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('trial', type=Path)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--reference-xml', type=Path, default=REFERENCE_XML)
    ap.add_argument('--frame-stride', type=int, default=1)
    ap.add_argument('--v4-only', action='store_true')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    r = check(a.trial, stride=a.stride, reference_xml=a.reference_xml, v5=not a.v4_only, frame_stride=a.frame_stride,
              write=a.out is None)
    if a.out is not None:
        a.out.write_text(json.dumps(r, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (
            o.item() if isinstance(o, np.generic) else str(o))) + '\n')
    print(json.dumps(r['summary'], indent=1))
    for k, g in r['gates'].items():
        if not g['pass']:
            brief = {kk: vv for kk, vv in g.items() if not isinstance(vv, (list, dict)) or kk in ('contact_violations_worst', 'parts_pass')}
            print(k, 'FAIL', json.dumps(brief, default=str)[:600])
    sys.exit(0 if r['all_pass'] else 1)


if __name__ == '__main__':
    main()
