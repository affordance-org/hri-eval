#!/usr/bin/env python3
"""v9: the v7 harness runner with the strap made optional (the v9 bag is ONE rigid body: no handle, no strap):
  * DEFAULT_XML stays the v3 hook scene for old callers; v9 episodes pass xml=HARNESS / 'g1_umi_room_v9_body.xml'
    (= V9_XML).  bag_kind() is 'strap' (v3-v7 scenes: bag_strap_tab exists) or 'body' (v9);
  * telemetry rows carry strap_qpos / strap_loop_top_center only for strap bags; for body bags they carry
    bag_grasp_sites (world position of every bag_grasp_* site: the grasp frame of the scene, TRUE bag state);
  * scene-context.json: bag_start_pose describes the rigid body bag (kind, grasp sites in the bag frame) for body bags;
  * disclosures for body bags: rigid bag without handle or strap grasped directly, the bag / pad contact, the
    assumed wide-stroke UMI; name-free wording (no person names, no machine paths, no em/en dashes).
Everything else (control loop, invariant measurement, outputs) is the v7 runner.

v3 copy of ../../harness/episode.py (the original is untouched): shared closed-loop episode harness for the
G1 + 2 UMI fetch-and-handover variants.  v3: scene = g1_umi_room_v3_hook.xml (articulated bag strap, kp-5000 finger
servos, upright bag at the video mark, hook or pinch recipient hand); the bag is the whole backpack SUBTREE.

One continuous MuJoCo episode from ONE initial state (keyframe 'start', set once):
  every 10 physics steps (50 Hz)   controller.policy_step(ctx) -> ControlOutput (held targets, gains, commands, ...)
  every physics step (500 Hz)      tau = kp (q* - q) - kd dq + tau_ff + [robot-only gravity torque on selected joints]
                                   clipped to the URDF caps (= motor ctrlrange) -> d.ctrl; finger servo targets -> d.ctrl;
                                   recipient mocap <- recipient.mocap(t); mj_step.
The runner MEASURES the forbidden-mechanism invariants instead of trusting the controller:
  * qpos/qvel/act/time writes after init: the state is snapshotted after every mj_step and compared bit-for-bit
    right before the next mj_step (after the controller ran and after the runner wrote ctrl/mocap) -> counted;
  * xfrc_applied / qfrc_applied: max |.| checked before every mj_step;
  * model arrays (masses, inertias, limits, actuator params, friction, solref, options, ...) hashed at every policy
    tick -> any mid-run change is counted;
  * neq, nmocap + mocap body names (only recipient_*), gravcomp, actuator gravcomp, options at load.
Forbidden mechanisms make the runner raise unless allow_forbidden=True (used only by the gate-checker unit tests).

Outputs (TRIAL DIRECTORY CONTRACT): trial.xml, trajectory.npz (every physics step), controller-log.npz
(50 Hz), telemetry.json (50 Hz), events.json, summary.json, scene-context.json, recipient-motion.json, README.md and,
when check=True, gates.json from check_gates.py (independent re-simulation).  trial.xml is the exact source text
(asset dirs made absolute; asset sha256 in summary.json) and the episode model is compiled FROM trial.xml; the
compiled-model fingerprint (sha256 over masses, inertias, limits, actuators, contact params, options) is in summary.json.

Plug-ins: Gr00tLocomotion, FalconLocomotion, PathFollower, OnlineArmIK, RobotGravity, compose_gr00t, load_reference.
See README.md for the API.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import mujoco as mj
import numpy as np

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

from gr00t_wbc import (ARM_KD_14, ARM_KP_14, JOINT_NAMES_29, KD_15, KP_15, Gr00tWbcPolicy,   # noqa: E402
                       urdf_effort_limits)
import recipient as RCP  # noqa: E402

DEFAULT_XML = HARNESS / "scene" / "room_point_fetch.xml"     # (v7 episodes pass xml=HARNESS / 'g1_umi_room_v7_body.xml')
V9_XML = HARNESS / "scene" / "g1_umi_room_v9_body.xml"          # v9 episodes: xml=V9_XML (rigid bag, no handle, no strap)
PINCH_XML = HARNESS / 'g1_umi_room_v3_pinch.xml'
REFERENCE_NPZ = HARNESS / 'reference.npz'      # not shipped: load_reference() is not used by this evaluation
FALCON_ONNX = HARNESS / 'policy' / 'falcon_g1_29dof.onnx'   # not shipped: FalconLocomotion is optional
FINGER_JOINTS = ['left_umi_left_finger_joint', 'left_umi_right_finger_joint',
                 'right_umi_left_finger_joint', 'right_umi_right_finger_joint']
ARM_IDX = np.arange(15, 29)
HUMAN_EVENT_FRAMES = (120, 252)                       # the second person motion start -> handover complete (recording frames)
HUMAN_DURATION_S = (HUMAN_EVENT_FRAMES[1] - HUMAN_EVENT_FRAMES[0]) / 30.0     # 4.4 s
CAPS_29 = urdf_effort_limits(JOINT_NAMES_29)
BAG_MASS = 1.2
GROUPS = {'legs': list(range(0, 4)) + list(range(6, 10)), 'ankles': [4, 5, 10, 11], 'waist': [12, 13, 14],
          'arms': [15, 16, 17, 18, 19, 22, 23, 24, 25, 26], 'wrist_pitch_yaw': [20, 21, 27, 28]}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jdefault(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def bag_subtree(m, root='backpack'):
    """Ids of all bodies in the bag subtree (bag body + strap links)."""
    r0 = m.body(root).id
    out = set()
    for b in range(m.nbody):
        r = b
        while r > 0 and r != r0:
            r = m.body_parentid[r]
        if r == r0:
            out.add(b)
    return out


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def portable_xml_text(xml_path):
    """XML text with meshdir/texturedir made absolute so trial.xml compiles from any directory."""
    xml_path = Path(xml_path).resolve()
    txt = xml_path.read_text()

    def fix(mo):
        attr, val = mo.group(1), mo.group(2)
        if val.startswith('/'):
            return mo.group(0)
        return f'{attr}="{(xml_path.parent / val).resolve()}/"'
    return re.sub(r'\b(meshdir|texturedir)="([^"]*)"', fix, txt)


def asset_hashes(xml_text):
    out = {}
    md = re.search(r'meshdir="([^"]*)"', xml_text)
    base = Path(md.group(1)) if md else Path('.')
    for f in sorted(set(re.findall(r'\bfile="([^"]+)"', xml_text))):
        p = Path(f) if f.startswith('/') else base / f
        out[f] = sha256_file(p) if p.exists() else None
    return out


def model_fingerprint(m):
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


def measure_model_invariants(m):
    mocap = [m.body(b).name for b in range(m.nbody) if m.body_mocapid[b] >= 0]
    return {'neq': int(m.neq), 'nmocap': int(m.nmocap), 'mocap_bodies': mocap,
            'mocap_only_recipient': all(n.startswith('recipient_') for n in mocap),
            'mocap_bodies_childless_world_children': all(
                m.body_parentid[m.body(n).id] == 0 and not any(m.body_parentid[c] == m.body(n).id for c in range(m.nbody))
                for n in mocap),
            'gravcomp_bodies': int(np.count_nonzero(m.body_gravcomp)),
            'actuator_gravcomp_joints': int(np.count_nonzero(m.jnt_actgravcomp)),
            'noslip_iterations': int(m.opt.noslip_iterations), 'timestep': float(m.opt.timestep),
            'integrator': ['euler', 'rk4', 'implicit', 'implicitfast'][int(m.opt.integrator)],
            'cone': ['pyramidal', 'elliptic'][int(m.opt.cone)], 'impratio': float(m.opt.impratio),
            'iterations': int(m.opt.iterations), 'ls_iterations': int(m.opt.ls_iterations), 'na': int(m.na),
            'gravity': m.opt.gravity.tolist()}


# ============================================================================================ controller interface
@dataclass
class ControlOutput:
    """What a controller returns every 50 Hz tick (held for the next 10 physics steps).

    q_target, kp, kd: (29,) in JOINT_NAMES_29 order.  tau = kp (q_target - q) - kd dq + tau_ff (+ gravity)
    tau_ff:     optional (29,) held feed-forward torque [Nm]
    gravcomp:   optional (29,) bool: add the ROBOT-ONLY generalised gravity torque (controller's model copy, qvel = 0,
                RNE on a scratch MjData; payload not included) on these joints, recomputed every physics step
    fingers:    (4,) finger servo targets [m] (L-UMI left, L-UMI right, R-UMI left, R-UMI right); 0 = open
    cmd:        dict(vx, vy, yaw_rate, base_height, torso_rpy=(r, p, y)) as sent to the locomotion policy (logged)
    policy_obs / policy_action: the locomotion policy's observation and raw action at this tick (logged)
    arm_ik_targets: optional (14,) left tool pos(3)+quat(4), right pos+quat (logged; NaN = unused)
    phase, phase_reason: phase label; a change is written to events.json with the reason
    log:        dict of extra fixed-shape numeric arrays to put in controller-log.npz
    done:       end the episode (after this tick, before stepping)
    """
    q_target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray | None = None
    gravcomp: np.ndarray | None = None
    fingers: np.ndarray = field(default_factory=lambda: np.zeros(4))
    cmd: dict | None = None
    policy_obs: np.ndarray | None = None
    policy_action: np.ndarray | None = None
    arm_ik_targets: np.ndarray | None = None
    phase: str = ''
    phase_reason: str = ''
    log: dict | None = None
    done: bool = False


class Ctx:
    """Read-only view of the running episode passed to controller.policy_step.  `d` is the live MjData: reading is
    fine, any write to qpos/qvel/act/time is detected and counted (and fails G0); use `scratch()` for kinematics."""

    def __init__(self, runner):
        self.r = runner
        self.m = runner.m
        self.d = runner.d
        self.recipient = runner.recipient

    def refresh(self, step, tick):
        r, d = self.r, self.d
        self.step, self.tick, self.t = step, tick, float(d.time)
        self.q29 = d.qpos[r.qadr].copy()
        self.dq29 = d.qvel[r.dadr].copy()
        self.base_pos = d.qpos[r.free_q:r.free_q + 3].copy()
        self.base_quat = d.qpos[r.free_q + 3:r.free_q + 7].copy()
        self.base_yaw = yaw_of(self.base_quat)
        self.base_linvel_world = d.qvel[r.free_d:r.free_d + 3].copy()
        self.base_angvel_body = d.qvel[r.free_d + 3:r.free_d + 6].copy()
        self.fingers_q = d.qpos[r.finger_qadr].copy()
        self.qpos = d.qpos.copy()
        self.qvel = d.qvel.copy()
        self.source_frame = r.recipient.source_frame(self.t) if r.recipient is not None else None

    def scratch(self):
        """A scratch MjData at the current state (positions/velocities/mocap copied, mj_forward run).  Safe to use
        for site/body poses, contacts, distances; writing to it does not affect the simulation."""
        ds = self.r.scratch
        ds.qpos[:] = self.d.qpos
        ds.qvel[:] = self.d.qvel
        ds.mocap_pos[:] = self.d.mocap_pos
        ds.mocap_quat[:] = self.d.mocap_quat
        ds.ctrl[:] = self.d.ctrl
        ds.time = self.d.time
        mj.mj_forward(self.m, ds)
        return ds

    def site(self, name):
        ds = self.scratch()
        s = self.m.site(name).id
        return ds.site_xpos[s].copy(), ds.site_xmat[s].reshape(3, 3).copy()

    def body(self, name):
        ds = self.scratch()
        b = self.m.body(name).id
        return ds.xpos[b].copy(), ds.xmat[b].reshape(3, 3).copy()

    def event(self, name, reason, **data):
        self.r.add_event(name, reason, **data)


# ============================================================================================ gravity compensation
class RobotGravity:
    """Generalised gravity torque of the robot's OWN links on the given joints (RNE with qvel = 0 on a scratch
    MjData of the episode model; the bag is a separate free body, so it is never included)."""

    def __init__(self, m, joint_names=JOINT_NAMES_29):
        self.m = m
        self.d = mj.MjData(m)
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in joint_names])
        self.out = np.zeros(m.nv)

    def __call__(self, qpos):
        self.d.qpos[:] = qpos
        self.d.qvel[:] = 0
        mj.mj_kinematics(self.m, self.d)
        mj.mj_comPos(self.m, self.d)
        mj.mj_rne(self.m, self.d, 0, self.out)
        return self.out[self.dadr].copy()


# ============================================================================================ runner
class EpisodeRunner:
    def __init__(self, controller, out_dir, xml=DEFAULT_XML, recipient=None, max_time=9.0, keyframe='start',
                 description='', allow_forbidden=False, telemetry=True, contacts_in_telemetry=True, check=True,
                 verbose=True):
        self.controller = controller
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.src_xml = Path(xml).resolve()
        self.max_time = float(max_time)
        self.keyframe = keyframe
        self.description = description
        self.allow_forbidden = allow_forbidden
        self.telemetry_on = telemetry
        self.contacts_in_telemetry = contacts_in_telemetry
        self.check = check
        self.verbose = verbose
        # ---- the exact model: trial.xml written first, then loaded from it (the checker loads the same file)
        txt = portable_xml_text(self.src_xml)
        (self.out / 'trial.xml').write_text(txt)
        self.m = mj.MjModel.from_xml_path(str(self.out / 'trial.xml'))
        stale = self.out / 'trial.mjb'
        if stale.exists():
            stale.unlink()
        self.asset_sha = asset_hashes(txt)
        m = self.m
        self.inv_load = measure_model_invariants(m)
        problems = []
        if m.neq:
            problems.append(f'neq={m.neq}')
        if not self.inv_load['mocap_only_recipient']:
            problems.append(f'mocap bodies {self.inv_load["mocap_bodies"]}')
        if self.inv_load['gravcomp_bodies'] or self.inv_load['actuator_gravcomp_joints']:
            problems.append('gravcomp')
        if m.opt.noslip_iterations:
            problems.append('noslip')
        if abs(m.opt.timestep - 0.002) > 1e-12 or self.inv_load['integrator'] != 'implicitfast' or \
                self.inv_load['cone'] != 'elliptic' or abs(m.opt.impratio - 10) > 1e-12:
            problems.append('physics options differ from dt 0.002 / implicitfast / elliptic / impratio 10')
        self.forbidden_at_load = problems
        if problems and not allow_forbidden:
            raise RuntimeError('forbidden model mechanisms: ' + '; '.join(problems))
        # ---- indices
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES_29])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINT_NAMES_29])
        act_of = {}
        for a in range(m.nu):
            act_of.setdefault(int(m.actuator_trnid[a, 0]), []).append(a)
        self.aid = np.array([act_of[m.joint(n).id][0] for n in JOINT_NAMES_29])
        for a in self.aid:
            assert m.actuator_gaintype[a] == mj.mjtGain.mjGAIN_FIXED and abs(m.actuator_gainprm[a, 0] - 1) < 1e-12
            assert m.actuator_biastype[a] == mj.mjtBias.mjBIAS_NONE and abs(m.actuator_gear[a, 0] - 1) < 1e-12
        self.caps = CAPS_29.copy()
        assert np.allclose(m.actuator_ctrlrange[self.aid, 1], self.caps) and np.allclose(m.actuator_forcerange[self.aid, 1], self.caps), \
            'motor ctrlrange/forcerange must equal the URDF effort caps'
        self.faid = np.array([act_of[m.joint(n).id][0] for n in FINGER_JOINTS])
        self.finger_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in FINGER_JOINTS])
        self.fing_lo = m.actuator_ctrlrange[self.faid, 0].copy()
        self.fing_hi = m.actuator_ctrlrange[self.faid, 1].copy()
        fj = m.body('pelvis').jntadr[0]
        self.free_q, self.free_d = int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])
        self.bag = m.body('backpack').id
        self.bag_bodies = bag_subtree(m)             # bag body + strap links = ONE object
        bj = m.body('backpack').jntadr[0]
        self.bag_q = int(m.jnt_qposadr[bj])
        self.pelvis = m.body('pelvis').id
        self.torso = m.body('torso_link').id
        self.robot_bodies = set()
        for b in range(m.nbody):
            r = b
            while r > 0:
                if r == self.pelvis:
                    self.robot_bodies.add(b)
                    break
                r = m.body_parentid[r]
        self.mocap_names = self.inv_load['mocap_bodies']
        self.mocap_id = {n: int(m.body_mocapid[m.body(n).id]) for n in self.mocap_names}
        # ---- recipient
        scene_hand = ('curl' if any(n.startswith('recipient_hand_l_') for n in self.mocap_names) else
                      'skin' if 'recipient_hand_l' in self.mocap_names else
                      'hook' if 'recipient_hook_hand' in self.mocap_names else 'pinch')
        if recipient is None and self.mocap_names:
            if scene_hand == 'curl':                # v6: curlable visible left hand (recipient6 + the scene sidecar)
                import recipient6 as RCP6
                recipient = RCP6.Recipient6.for_scene(xml)
            elif scene_hand == 'skin':
                import recipient5 as RCP5
                recipient = RCP5.Recipient5(mocap_bodies=self.mocap_names)
            else:
                recipient = RCP.Recipient(hand=scene_hand)
        self.recipient = recipient
        if recipient is not None and self.mocap_names:
            assert getattr(recipient, 'hand', 'pinch') == scene_hand, \
                f'Recipient(hand={getattr(recipient, "hand", "pinch")!r}) but the scene has a {scene_hand} hand'
            assert set(recipient.mocap_bodies) == set(self.mocap_names), (recipient.mocap_bodies, self.mocap_names)
        if recipient is not None and scene_hand == 'hook' and 'recipient_hook_hand' in self.mocap_names:
            for gname, gtype, prm in recipient.hp.geoms():
                g = m.geom(gname)
                if gtype == 'capsule':
                    assert abs(g.size[0] - prm['r']) < 1e-9, f'{gname}: HookParams differ from the scene XML'
                else:
                    assert np.allclose(g.size, prm['half']) and np.allclose(g.pos, prm['pos']), \
                        f'{gname}: HookParams differ from the scene XML'
        if recipient is not None and 'recipient_palm_pad' in self.mocap_names:
            g_palm = m.geom('recipient_palm_pad_collider')
            g_thumb = m.geom('recipient_thumb_pad_collider')
            assert np.allclose(g_palm.size, recipient.pp.palm_half) and np.allclose(g_thumb.size, recipient.pp.thumb_half), \
                'recipient PinchParams pad sizes differ from the scene XML (rebuild the scene or pass matching PinchParams)'
        self.gravity = RobotGravity(m)
        self.scratch = mj.MjData(m)
        self.tel_data = mj.MjData(m)
        self.d = mj.MjData(m)
        self.events = []
        self.fingerprint0 = model_fingerprint(m)
        # bag collision vertices (for clearance telemetry)
        self.bag_verts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in self.bag_bodies and (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                self.bag_verts.append((g, m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()))
        self.table_top_z = float(m.geom('table_top').pos[2] + m.geom('table_top').size[2])
        names_j = {m.joint(j).name for j in range(m.njnt)}
        if {'bag_strap_anchor', 'bag_strap_bend'} <= names_j:          # v3-v7 strap bag
            self.strap_qadr = np.concatenate([np.arange(m.jnt_qposadr[m.joint(j).id], m.jnt_qposadr[m.joint(j).id] +
                                                        (4 if m.jnt_type[m.joint(j).id] == mj.mjtJoint.mjJNT_BALL else 1))
                                              for j in ('bag_strap_anchor', 'bag_strap_bend')])
            self.strap_site = m.site('strap_loop_top_center').id
        else:                                                          # v9: rigid bag body, no strap
            self.strap_qadr = np.zeros(0, int)
            self.strap_site = None
        self.bag_grasp_sites = {m.site(s).name: s for s in range(m.nsite)
                                if int(m.site_bodyid[s]) == self.bag and m.site(s).name.startswith('bag_grasp_')}
        self.finger_bodies = {s: {m.body(f'{s}_umi_{f}_finger_holder').id for f in ('left', 'right')} for s in ('left', 'right')}
        self.pad_bodies = {n: m.body(n).id for n in ('recipient_palm_pad', 'recipient_thumb_pad', 'recipient_hook_hand')
                           if n in self.mocap_names}
        if scene_hand in ('skin', 'curl'):  # v5/v6: every skin-matched recipient body can carry the bag
            body = ('pelvis', 'spine_01', 'spine_02', 'spine_03', 'thigh_l', 'calf_l', 'foot_l', 'thigh_r', 'calf_r', 'foot_r')
            self.pad_bodies = {n: m.body(n).id for n in self.mocap_names     # v7: torso/leg colliders are not pads
                               if n.replace('recipient_', '') not in body}
        self.scene_hand = scene_hand

    # ---------------------------------------------------------------------------------------------- helpers
    @staticmethod
    def _event_value(v):
        """v7 (B4b): numeric arrays/lists are rounded floats; lists/tuples holding any string (e.g. signals=[...])
        are kept verbatim as lists of str; numpy scalars become Python scalars; bools stay bools."""
        if isinstance(v, np.ndarray):
            return np.round(v.astype(float), 6).tolist() if v.dtype.kind in 'biuf' else [str(x) for x in v.ravel()]
        if isinstance(v, (list, tuple)):
            if any(isinstance(x, (str, bytes)) for x in v):
                return [x.decode() if isinstance(x, bytes) else str(x) for x in v]
            try:
                return np.round(np.asarray(v, float), 6).tolist()
            except (TypeError, ValueError):
                return [str(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v

    def add_event(self, name, reason, **data):
        self.events.append({'time': round(float(self.d.time), 6), 'event': name, 'reason': reason,
                            **{k: self._event_value(v) for k, v in data.items()}})

    def bag_kind(self):
        """v9: 'strap' (v3-v7 scenes with the modelled strap) or 'body' (v9: rigid bag, no handle, no strap)."""
        return 'strap' if self.strap_site is not None else 'body'

    def _is_v7_scene(self):
        try:
            return abs(float(self.m.geom('bag_strap_tab_collision').size[2]) - 0.010) > 1e-6   # v3 tab: 20 mm (half 10)
        except Exception:
            return False

    def _tab_half(self):
        try:
            return float(self.m.geom('bag_strap_tab_collision').size[2])
        except Exception:
            return 0.010

    def _has_body_colliders(self):
        names = {self.m.body(b).name for b in range(self.m.nbody)}
        return any(f'recipient_{b}' in names for b in ('pelvis', 'spine_02', 'thigh_l'))

    def gname(self, g):
        n = self.m.geom(g).name
        return n if n else f'{self.m.body(self.m.geom_bodyid[g]).name}#g{g}'

    def set_mocap(self, t):
        if self.recipient is None:
            return
        poses = self.recipient.mocap(t)
        for n, mid in self.mocap_id.items():
            p, q = poses[n]
            self.d.mocap_pos[mid] = p
            self.d.mocap_quat[mid] = q

    def telemetry_row(self, phase, tau):
        m, d, ds = self.m, self.d, self.tel_data
        ds.qpos[:] = d.qpos
        ds.qvel[:] = d.qvel
        ds.act[:] = d.act
        ds.mocap_pos[:] = d.mocap_pos
        ds.mocap_quat[:] = d.mocap_quat
        ds.ctrl[:] = d.ctrl
        ds.time = d.time
        mj.mj_forward(m, ds)
        f6 = np.zeros(6)
        contacts = []
        pad = {'left': 0.0, 'right': 0.0}
        rpad = ({'palm': 0.0, 'thumb': 0.0} if 'recipient_palm_pad' in self.pad_bodies else
                {n.replace('recipient_', ''): 0.0 for n in self.pad_bodies} if self.scene_hand in ('skin', 'curl') else {'hook': 0.0})
        for i in range(ds.ncon):
            c = ds.contact[i]
            mj.mj_contactForce(m, ds, i, f6)
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if self.contacts_in_telemetry:
                contacts.append({'geom1': self.gname(c.geom1), 'geom2': self.gname(c.geom2), 'body1': m.body(b1).name,
                                 'body2': m.body(b2).name, 'pos': np.round(c.pos, 5).tolist(),
                                 'normal': np.round(c.frame[:3], 5).tolist(), 'force6': np.round(f6, 4).tolist(),
                                 'dist': round(float(c.dist), 6)})
            in1, in2 = b1 in self.bag_bodies, b2 in self.bag_bodies
            if in1 != in2:                           # contacts between bag bodies (bag <-> strap) are internal
                other = b2 if in1 else b1
                for s in ('left', 'right'):
                    if other in self.finger_bodies[s]:
                        pad[s] += float(f6[0])
                for n, bid in self.pad_bodies.items():
                    if other == bid:
                        key = (n.replace('recipient_', '') if self.scene_hand in ('skin', 'curl') else
                               'hook' if 'hook' in n else ('palm' if 'palm' in n else 'thumb'))
                        rpad[key] += float(f6[0])
        bp = d.qpos[self.bag_q:self.bag_q + 3]
        minz = min(float(((v @ ds.geom_xmat[g].reshape(3, 3).T) + ds.geom_xpos[g])[:, 2].min()) for g, v in self.bag_verts)
        Rp = ds.xmat[self.pelvis].reshape(3, 3)
        Rt = ds.xmat[self.torso].reshape(3, 3)
        ratio = np.abs(tau) / self.caps
        row = {'time': round(float(d.time), 6), 'phase': phase,
               'base_pos': np.round(d.qpos[self.free_q:self.free_q + 3], 5).tolist(),
               'base_yaw': round(yaw_of(d.qpos[self.free_q + 3:self.free_q + 7]), 5),
               'base_height': round(float(d.qpos[self.free_q + 2]), 5),
               'tilt_deg': round(math.degrees(math.acos(max(-1, min(1, Rp[2, 2])))), 3),
               'torso_tilt_deg': round(math.degrees(math.acos(max(-1, min(1, Rt[2, 2])))), 3),
               'left_pad_normal_N': round(pad['left'], 4), 'right_pad_normal_N': round(pad['right'], 4),
               'bag_pos': np.round(bp, 5).tolist(), 'bag_quat': np.round(d.qpos[self.bag_q + 3:self.bag_q + 7], 6).tolist(),
               'bag_min_z': round(minz, 5), 'bag_clearance_above_table': round(minz - self.table_top_z, 5),
               **({'strap_qpos': np.round(d.qpos[self.strap_qadr], 6).tolist(),
                   'strap_loop_top_center': np.round(ds.site_xpos[self.strap_site], 5).tolist()}
                  if self.strap_site is not None else
                  {'bag_grasp_sites': {n: np.round(ds.site_xpos[s], 5).tolist() for n, s in self.bag_grasp_sites.items()}}),
               'recipient_pad_normal_N': {k: round(v, 4) for k, v in rpad.items()},
               'torque_ratio_max': {k: round(float(ratio[v].max()), 4) for k, v in GROUPS.items()},
               'contacts': contacts}
        if self.recipient is not None:
            info = getattr(self.recipient, 'last_info', {}) or {}
            row['reference_frame'] = round(float(self.recipient.source_frame(float(d.time))), 4)
            row['recipient_reach_weight'] = round(float(info.get('w', 0.0)), 4)
            row['recipient_gap_m'] = round(float(info.get('gap', float('nan'))), 5)
            if self.scene_hand in ('skin', 'curl'):
                row['recipient_source_frame_rate'] = None
            if hasattr(self.recipient, 'curl_value'):
                row['recipient_curl_l'] = round(float(self.recipient.curl_value(float(d.time))), 6)
            if 'pinch_centre' in info:
                row['recipient_pinch_centre'] = np.round(info['pinch_centre'], 5).tolist()
        return row

    # ---------------------------------------------------------------------------------------------- run
    def run(self):
        m, d = self.m, self.d
        dt = m.opt.timestep
        dec = int(round(0.02 / dt))
        n_max = int(round(self.max_time / dt))
        kid = m.key(self.keyframe).id
        mj.mj_resetDataKeyframe(m, d, kid)        # THE ONLY STATE WRITE (initialisation)
        d.ctrl[:] = 0
        self.set_mocap(0.0)
        mj.mj_forward(m, d)
        ctx = Ctx(self)
        # buffers
        N = n_max + 1
        T = {'time': np.zeros(N), 'qpos': np.zeros((N, m.nq)), 'qvel': np.zeros((N, m.nv)), 'ctrl': np.zeros((N, m.nu)),
             'qfrc_actuator': np.zeros((N, m.nv)), 'act': np.zeros((N, m.na)), 'mocap_pos': np.zeros((N, m.nmocap, 3)),
             'mocap_quat': np.zeros((N, m.nmocap, 4))}

        def record(i):
            T['time'][i] = d.time
            T['qpos'][i] = d.qpos
            T['qvel'][i] = d.qvel
            T['ctrl'][i] = d.ctrl
            T['qfrc_actuator'][i] = d.qfrc_actuator
            T['act'][i] = d.act
            T['mocap_pos'][i] = d.mocap_pos
            T['mocap_quat'][i] = d.mocap_quat
        record(0)
        log_rows = []
        tel = []
        phase_names = []
        writes, write_steps = 0, []
        model_changes = 0
        max_xfrc = max_qfrc = 0.0
        tau_demand_max = np.zeros(29)
        tau_applied_max = np.zeros(29)
        sat_steps = np.zeros(29, int)
        failure = None
        end_reason = 'max_time'
        held = None
        phase = None
        step = 0
        snap = (d.qpos.copy(), d.qvel.copy(), d.act.copy(), float(d.time))
        if hasattr(self.controller, 'reset'):
            ctx.refresh(0, 0)
            self.controller.reset(ctx)
        wall0 = time.time()
        try:
            while step < n_max:
                tick = step % dec == 0
                if tick:
                    ctx.refresh(step, step // dec)
                    out = self.controller.policy_step(ctx)
                    out.q_target = np.asarray(out.q_target, float).reshape(29)
                    out.kp = np.broadcast_to(np.asarray(out.kp, float), (29,)).copy()
                    out.kd = np.broadcast_to(np.asarray(out.kd, float), (29,)).copy()
                    out.fingers = np.asarray(out.fingers, float).reshape(4)
                    if out.gravcomp is not None:
                        out.gravcomp = np.asarray(out.gravcomp, bool).reshape(29)
                    if out.phase != phase:
                        self.add_event('phase', out.phase_reason or '', phase=out.phase, previous=phase)
                        phase = out.phase
                    if phase not in phase_names:
                        phase_names.append(phase)
                    held = out
                    if model_fingerprint(m) != self.fingerprint0:
                        model_changes += 1
                    if out.done:
                        end_reason = 'controller_done'
                        log_rows.append(self._log_row(out, phase_names.index(phase), ctx))
                        break
                q = d.qpos[self.qadr]
                dq = d.qvel[self.dadr]
                tau = held.kp * (held.q_target - q) - held.kd * dq
                if held.tau_ff is not None:
                    tau = tau + held.tau_ff
                if held.gravcomp is not None and held.gravcomp.any():
                    tau = tau + np.where(held.gravcomp, self.gravity(d.qpos), 0.0)
                np.maximum(tau_demand_max, np.abs(tau), out=tau_demand_max)
                sat_steps += np.abs(tau) > self.caps
                tau = np.clip(tau, -self.caps, self.caps)
                np.maximum(tau_applied_max, np.abs(tau), out=tau_applied_max)
                d.ctrl[self.aid] = tau
                d.ctrl[self.faid] = np.clip(held.fingers, self.fing_lo, self.fing_hi)
                self.set_mocap(float(d.time))
                if tick:
                    log_rows.append(self._log_row(held, phase_names.index(phase), ctx))
                    if self.telemetry_on:
                        tel.append(self.telemetry_row(phase, tau))
                xf = float(np.abs(d.xfrc_applied).max()) if d.xfrc_applied.size else 0.0
                qf = float(np.abs(d.qfrc_applied).max()) if d.qfrc_applied.size else 0.0
                max_xfrc, max_qfrc = max(max_xfrc, xf), max(max_qfrc, qf)
                # state-write detection right before stepping: the state must be bit-identical to the one mj_step
                # produced (anything the controller or this loop did in between is caught)
                if not (np.array_equal(d.qpos, snap[0]) and np.array_equal(d.qvel, snap[1]) and
                        np.array_equal(d.act, snap[2]) and float(d.time) == snap[3]):
                    writes += 1
                    if len(write_steps) < 50:
                        write_steps.append(step)
                mj.mj_step(m, d)
                step += 1
                record(step)
                snap = (d.qpos.copy(), d.qvel.copy(), d.act.copy(), float(d.time))
                if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
                    failure = f'simulation diverged at t={d.time:.3f}'
                    end_reason = 'diverged'
                    break
                if self.verbose and step % 500 == 0:
                    print(f'  t={d.time:5.2f}s phase={phase} pelvis={np.round(d.qpos[self.free_q:self.free_q+3], 3)} '
                          f'bag={np.round(d.qpos[self.bag_q:self.bag_q+3], 3)} wall={time.time()-wall0:.1f}s', flush=True)
        except Exception as e:  # controller crash: keep everything recorded so far
            failure = f'controller exception at t={d.time:.3f}: {e!r}'
            end_reason = 'exception'
            self.add_event('exception', traceback.format_exc()[-2000:])
        wall = time.time() - wall0
        n = step + 1
        for k in T:
            T[k] = T[k][:n]
        np.savez_compressed(self.out / 'trajectory.npz', **T)
        self._write_log(log_rows, phase_names)
        if self.telemetry_on:
            (self.out / 'telemetry.json').write_text(json.dumps(tel, separators=(',', ':'), default=jdefault) + '\n')
        (self.out / 'events.json').write_text(json.dumps(self.events, indent=1, default=jdefault) + '\n')
        if self.recipient is not None:
            self.recipient.write_motion_json(self.out / 'recipient-motion.json', t_end=float(d.time))
        inv = dict(self.inv_load)
        inv.update({'qpos_qvel_writes_after_init': int(writes), 'write_steps_first50': write_steps,
                    'max_abs_xfrc_applied': max_xfrc, 'max_abs_qfrc_applied': max_qfrc,
                    'model_changed_during_run_ticks': int(model_changes),
                    'model_fingerprint_sha256': self.fingerprint0,
                    'torque': {'joint_names': JOINT_NAMES_29, 'cap_Nm': self.caps.tolist(),
                               'max_applied_Nm': tau_applied_max.tolist(),
                               'max_applied_over_cap': float((tau_applied_max / self.caps).max()),
                               'max_demand_over_cap': (tau_demand_max / self.caps).tolist(),
                               'saturated_steps': sat_steps.tolist(),
                               'note': 'demanded PD torque is clipped to the URDF cap before it is written to ctrl'},
                    'forbidden_at_load': self.forbidden_at_load, 'initialisation': f'mj_resetDataKeyframe({self.keyframe!r}) once'})
        self.summary = {
            'controller_execution': 'closed_loop_full_episode', 'diagnostic_only': False,
            'description': self.description,
            'episode': {'physics_steps': step, 'sim_time_s': float(d.time), 'wall_time_s': wall,
                        'realtime_factor': float(d.time / wall) if wall > 0 else None, 'end_reason': end_reason,
                        'policy_ticks': len(log_rows), 'initial_state': f'keyframe {self.keyframe!r}'},
            'policy': self.controller.policy_info() if hasattr(self.controller, 'policy_info') else {},
            'physics': {k: inv[k] for k in ('timestep', 'integrator', 'cone', 'impratio', 'noslip_iterations',
                                            'iterations', 'ls_iterations', 'gravity')} | {'mujoco': mj.__version__,
                                                                                         'policy_hz': 50, 'pd_hz': 500},
            'model': {'trial_xml_sha256': sha256_file(self.out / 'trial.xml'), 'source_xml': str(self.src_xml),
                      'source_sha256': sha256_file(self.src_xml), 'asset_sha256': self.asset_sha},
            'invariants_measured': inv,
            'outcome': {'pickup_verified': None, 'carry_verified': None, 'handover_verified': None,
                        'source': 'check_gates.py (independent re-simulation) - filled when check=True'},
            'timings': {'human_duration_s': HUMAN_DURATION_S, 'human_frames': list(HUMAN_EVENT_FRAMES),
                        'robot_duration_s': None, 'pre_roll_s': None},
            'failure': failure,
            'disclosures': self.disclosures()}
        self._write_scene_context(T)
        self._write_readme()
        self._dump_summary()
        if self.check:
            import check_gates
            g = check_gates.check(self.out, write=True)
            o = self.summary['outcome']
            o.update({'pickup_verified': g['gates']['G3_pickup']['pass'], 'carry_verified': g['gates']['G4_carry']['pass'],
                      'handover_verified': g['gates']['G5_handover']['pass'], 'all_gates_pass': g['all_pass'],
                      'gates': {k: v['pass'] for k, v in g['gates'].items()}})
            self.summary['timings']['robot_duration_s'] = g['gates']['G6_pace'].get('robot_duration_s')
            self.summary['timings']['pre_roll_s'] = g['gates']['G6_pace'].get('pre_roll_s')
            self.summary['timings']['handover_complete_s'] = g['gates']['G6_pace'].get('t_handover_complete_s')
            self._dump_summary()
        if self.verbose:
            print(f'episode done: {end_reason}, sim {d.time:.2f}s wall {wall:.1f}s '
                  f'({d.time / max(wall, 1e-9):.2f}x realtime), writes={writes}, failure={failure}')
        return self.summary

    # ---------------------------------------------------------------------------------------------- outputs
    def _log_row(self, out, phase_id, ctx):
        cmd = out.cmd or {}
        rpy = cmd.get('torso_rpy', (np.nan, np.nan, np.nan))
        row = {'time': ctx.t, 'phase_id': phase_id, 'q_target': out.q_target, 'kp': out.kp, 'kd': out.kd,
               'tau_ff': out.tau_ff if out.tau_ff is not None else np.zeros(29),
               'gravcomp_mask': out.gravcomp if out.gravcomp is not None else np.zeros(29, bool),
               'finger_target': out.fingers,
               'cmd': np.array([cmd.get('vx', np.nan), cmd.get('vy', np.nan), cmd.get('yaw_rate', np.nan),
                                cmd.get('base_height', np.nan), *rpy], float),
               'arm_joint_target': out.q_target[15:],
               'arm_ik_targets': out.arm_ik_targets if out.arm_ik_targets is not None else np.full(14, np.nan),
               'base_pos': ctx.base_pos, 'base_yaw': ctx.base_yaw}
        if out.policy_obs is not None:
            row['policy_obs'] = np.asarray(out.policy_obs, float).ravel()
        if out.policy_action is not None:
            row['policy_action'] = np.asarray(out.policy_action, float).ravel()
        for k, v in (out.log or {}).items():
            row['x_' + k] = np.asarray(v, float)
        return row

    def _write_log(self, rows, phase_names):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        out = {'phase_names': np.array(phase_names, dtype=str)}
        for k in keys:
            vals = [r.get(k) for r in rows]
            shape = next(np.shape(v) for v in vals if v is not None)
            arr = np.full((len(rows),) + tuple(shape), np.nan)
            for i, v in enumerate(vals):
                if v is not None and np.shape(v) == shape:
                    arr[i] = v
            out[k] = arr
        out['cmd_columns'] = np.array(['vx', 'vy', 'yaw_rate', 'base_height', 'torso_roll', 'torso_pitch', 'torso_yaw'])
        out['joint_names'] = np.array(JOINT_NAMES_29)
        np.savez_compressed(self.out / 'controller-log.npz', **out)

    def disclosures(self):
        if self.bag_kind() == 'body':
            return self._disclosures_v9()
        base = [
            'Backpack stands UPRIGHT on the low table at the raw-video layout mark (1.663, 5.399), yaw 0 '
            '(' + ('v7/scene/g1_umi_room_v7.xml' if self._is_v7_scene() else 'v3/scene/g1_umi_room_v3.xml') +
            '); in the recording it lies on its back there (top pointing west).',
            'The scanned rigid carry handle (bag_1_15) is replaced by a modelled articulated webbing strap: ball joint '
            '(weak spring to upright, damping, cone limit) -> ' + f"{2000 * self._tab_half():.0f}" + ' mm tab -> hinge -> flat 110 x 58 x 4 mm haul loop whose '
            'loop is rigid; an open chain of child bodies of the bag (no '
            'equality constraints), 25 g of the 1.2 kg total.' + (
                ' v7 scene: the tab is 40 mm long (v3: 20 mm, masses unchanged) and the UMI soft-finger pads and the '
                'strap geoms use a stiffer, less dissipative contact (solref 0.004 1, solimp 0.99 0.999 0.0005, '
                'condim 4, friction 1.0) to limit MuJoCo soft-contact creep of the strap in the pinch (measured with a '
                'test-only gantry probe, v7/README.md); constant for the whole episode, noslip 0.'
                if self._is_v7_scene() else ''),
            'UMI finger servos: assumed spec 35 N per finger (position servo kp 5000 N/m, kv 30, force-limited to '
            '+/-35 N, command range 0-0.05 m); not manufacturer data.',
            'Recipient (the person) is a REPLAYED RECORDING (kinematic, infinite mass), retimed by holding poses (never '
            'faster than real time); his LEFT ARM ONLY is adapted by an analytic 2-bone IK during the receive so his '
            'palm meets the presented strap; his receiving hand is ' + (
                'his own avatar LEFT hand with a SCRIPTED finger curl (two hinges per finger: knuckle 70 deg and '
                'middle joint 80 deg at full curl, thumb rigid; the avatar has no finger bones and the recording has no '
                'finger motion, so the curl is a modelling choice shown on the website as a glTF morph), physically one '
                'rigid mocap collider body per finger segment fitted to the curled skin (recipient_hand_l_*), plus v5 '
                'skin-matched convex pieces for his forearms and right hand'
                if getattr(self, 'scene_hand', None) == 'curl' or (self.recipient is not None and
                                                                   hasattr(self.recipient, 'curl_value')) else
                'a kinematic palm + curled-finger hook (recipient_hook_hand, one mocap body: 2 finger capsules + curled '
                'tips + palm box), which carries the strap loop by normal force'
                if self.recipient is not None and getattr(self.recipient, 'hand', 'pinch') == 'hook' else
                'his own avatar hand/forearm surface (v5 skin-matched convex pieces in the hand/lowerarm bone frames, '
                'mocap bodies recipient_{hand,lowerarm}_{l,r}); no finger articulation (the avatar has no finger bones)'
                if self.recipient is not None and getattr(self.recipient, 'hand', 'pinch') == 'skin' else
                'a kinematic two-pad pinch (palm pad + thumb pad mocap boxes)') +
            '. The same adapted bones drive the physics colliders and recipient-motion.json.' + (
                ' His pelvis, torso (spine_01-03), thighs, calves and feet are also physical: rigid convex mocap pieces '
                'fitted to his visible surface (skin + clothes) and posed at his bones every step; they collide with the '
                'robot and the bag (not with the room); blend regions (hips, waist) deform with his joints, the pieces '
                'do not.' if self._has_body_colliders() else ''),
            'Robot start pose = the second person\'s frame-120 pelvis pose (keyframe start); backpack mass 1.2 kg and UMI 0.8 kg are '
            'assumed values.']
        extra = self.controller.disclosures() if hasattr(self.controller, 'disclosures') else []
        return base + list(extra)

    def _disclosures_v9(self):
        """v9 (rigid bag body, no handle, no strap).  Name-free, path-free, no em/en dashes."""
        m = self.m
        bag_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.bag and (m.geom_contype[g] or m.geom_conaffinity[g])]
        sr, si = m.geom_solref[bag_g[0]], m.geom_solimp[bag_g[0]]
        fing = self.fing_lo.min(), self.fing_hi.max()
        base = [
            'The backpack stands UPRIGHT on the low table at the raw-video layout mark (1.663, 5.399), yaw 0; in the '
            'recording it lies on its back there (top pointing west). No direct grasp of the lying bag fits the gripper, '
            'so the upright pose is a disclosed deviation.',
            f'The backpack is ONE rigid body: its {len(bag_g)} scanned body parts as convex collision hulls, '
            f'{float(sum(m.body_mass[b] for b in self.bag_bodies)):.1f} kg (assumed). The scanned carry handle is not in '
            'the model and no strap or handle was added: the robot grasps the bag body directly.',
            f'The bag hulls and the gripper finger pads use a stiff, low-dissipation contact (solref {sr[0]:g} {sr[1]:g}, '
            f'solimp {si[0]:g} {si[1]:g} {si[2]:g}, condim {int(m.geom_condim[bag_g[0]])}, friction '
            f'{m.geom_friction[bag_g[0]][0]:g}), chosen with a test-only gantry probe to limit MuJoCo soft-contact creep '
            'in the pinch; constant for the whole episode, noslip 0.',
            'Gripper finger servos: assumed spec 35 N per finger (position servo kp 5000 N/m, kv 30, force-limited to '
            f'+/-35 N); assumed WIDE-STROKE gripper: command range {fing[0]:g} to {fing[1]:g} m per finger, open gap '
            '153 mm (35 mm per finger wider than the reference model); not manufacturer data.',
            'The recipient is a REPLAYED RECORDING (kinematic, infinite mass). His left hand is his own avatar hand with a '
            'SCRIPTED finger curl (the avatar has no finger bones and the recording has no finger motion), physically one '
            'rigid collider per finger segment fitted to the curled skin, plus skin-matched convex pieces for his '
            'forearms and right hand. The same bones drive the physics colliders and the motion shown on the website.' + (
                ' His pelvis, torso, thighs, calves and feet are also physical: rigid convex pieces fitted to his visible '
                'surface and posed at his bones every step; they collide with the robot and the bag, not with the room.'
                if self._has_body_colliders() else ''),
            'Robot start pose = the fetcher\'s recorded frame-120 pelvis pose; backpack mass 1.2 kg and gripper mass '
            '0.8 kg are assumed values.']
        extra = self.controller.disclosures() if hasattr(self.controller, 'disclosures') else []
        return base + list(extra)

    def _write_scene_context(self, T):
        m = self.m
        q0 = T['qpos'][0]
        ctx = {'room_from_physics_translation': [0.0, 0.0, 0.0], 'room_offset': 0,
               'frame': 'room frame of the recording (x/y floor plan, z up, metres)',
               'table_top_z': self.table_top_z,
               'bag_start_pose': ({'pos': q0[self.bag_q:self.bag_q + 3].tolist(), 'quat_wxyz': q0[self.bag_q + 3:self.bag_q + 7].tolist(),
                                   'variant': 'v3: upright on the low table at the video mark (disclosed deviation; '
                                              'recording: lying on its back)',
                                   'strap_qpos': q0[self.strap_qadr].tolist(),
                                   'strap_joints': ['bag_strap_anchor (ball, qpos 4)', 'bag_strap_bend (hinge, qpos 1)'],
                                   'strap_bodies': ['bag_strap_tab', 'bag_strap_loop'],
                                   'handle_center_site': 'strap_loop_top_center (v2: bag_handle_center)',
                                   'mass_kg': float(sum(m.body_mass[b] for b in self.bag_bodies))}
                                  if self.bag_kind() == 'strap' else
                                  {'pos': q0[self.bag_q:self.bag_q + 3].tolist(), 'quat_wxyz': q0[self.bag_q + 3:self.bag_q + 7].tolist(),
                                   'variant': 'v9: upright on the low table at the video mark (disclosed deviation; '
                                              'recording: lying on its back)',
                                   'kind': 'one rigid body (scanned body parts), no handle, no strap',
                                   'grasp_sites_bag_frame': {n: {'pos': m.site_pos[s].tolist(), 'quat_wxyz': m.site_quat[s].tolist(),
                                                                 'axes': 'x closing, y pad width, z approach (UMI base axes)'}
                                                             for n, s in self.bag_grasp_sites.items()},
                                   'mass_kg': float(sum(m.body_mass[b] for b in self.bag_bodies))}),
               'robot_start': {'keyframe': self.keyframe, 'pelvis_pos': q0[self.free_q:self.free_q + 3].tolist(),
                               'pelvis_yaw_rad': yaw_of(q0[self.free_q + 3:self.free_q + 7]),
                               'source': "the second person's frame-120 pelvis pose"},
               'recipient': {'person': 'person_01', 'kind': 'replayed recording, retimed (holds only), adapted left arm '
                             '(2-bone IK) during the receive, kinematic ' + ('palm + curled-finger hook hand'
                             if getattr(self.recipient, 'hand', 'pinch') == 'hook' else 'two-pad pinch hand'),
                             'mocap_bodies': self.mocap_names,
                             'schedules': self.recipient.schedules_json() if self.recipient is not None else None},
               'recording': str(RCP.MOTION.parent.parent)}
        (self.out / 'scene-context.json').write_text(json.dumps(ctx, indent=1, default=jdefault) + '\n')

    def _write_readme(self):
        s = self.summary
        drv = self.controller.describe() if hasattr(self.controller, 'describe') else '(controller provided no description)'
        txt = f"""# Trial: {self.out.name}

{self.description}

One continuous closed-loop MuJoCo episode (`mj_step` only) from keyframe `{self.keyframe}` of `{self.src_xml.name}`;
{s['episode']['physics_steps']} physics steps (dt {self.m.opt.timestep}), {s['episode']['sim_time_s']:.2f} s simulated.

## What drives what
* Policy ({s['policy'].get('name', '?')}) at 50 Hz -> joint position targets; joint PD torques every physics step
  (500 Hz), clipped to the official URDF effort limits, written to plain torque motors.
* Fingers: force-limited position servos (+/-35 N), targets from the controller.
* Recipient: mocap colliders `recipient_*` driven by `recipient.py` (replayed recording, see disclosures).
* Controller: {drv}

## Measured invariants (runner instrumentation)
neq {s['invariants_measured']['neq']}, mocap {s['invariants_measured']['mocap_bodies']},
qpos/qvel writes after init {s['invariants_measured']['qpos_qvel_writes_after_init']},
max |xfrc_applied| {s['invariants_measured']['max_abs_xfrc_applied']}, max |qfrc_applied| {s['invariants_measured']['max_abs_qfrc_applied']},
gravcomp bodies {s['invariants_measured']['gravcomp_bodies']}, max torque/cap {s['invariants_measured']['torque']['max_applied_over_cap']:.3f}.
Gates: see `gates.json` (independent re-simulation by `check_gates.py`).

## Disclosures
""" + '\n'.join(f'* {x}' for x in s['disclosures']) + '\n'
        (self.out / 'README.md').write_text(txt)

    def _dump_summary(self):
        (self.out / 'summary.json').write_text(json.dumps(self.summary, indent=1, default=jdefault) + '\n')


# ============================================================================================ plug-ins
class Gr00tLocomotion:
    """GR00T decoupled WBC (Balance/Walk ONNX, bit-exact port in ../gr00t/gr00t_wbc.py).  Observations are exactly
    the trained ones (pelvis quat, pelvis body angular velocity, 29 joint q/dq, commands, last action, 6-frame
    history).  step() returns the 15 lower-body (legs + waist) position targets for KP_15/KD_15 PD."""
    name = 'NVIDIA GR00T-WholeBodyControl decoupled WBC (Balance + Walk)'

    def __init__(self):
        self.pol = Gr00tWbcPolicy()
        import gr00t_wbc
        self.paths = [gr00t_wbc.DEFAULT_BALANCE_ONNX, gr00t_wbc.DEFAULT_WALK_ONNX]
        self.kp15, self.kd15 = KP_15.copy(), KD_15.copy()

    def reset(self):
        self.pol.reset()

    def step(self, ctx, vx=0.0, vy=0.0, yaw_rate=0.0, base_height=0.74, torso_rpy=(0.0, 0.0, 0.0)):
        self.pol.set_command(vx=vx, vy=vy, yaw_rate=yaw_rate, base_height=base_height, torso_rpy=torso_rpy)
        q15 = self.pol.step(ctx.base_quat, ctx.base_angvel_body, ctx.q29, ctx.dq29)
        self.cmd = {'vx': float(vx), 'vy': float(vy), 'yaw_rate': float(yaw_rate), 'base_height': float(base_height),
                    'torso_rpy': tuple(float(x) for x in torso_rpy)}
        self.obs = self.pol.obs_buffer.copy()
        self.action = self.pol.action.astype(float).copy()
        return q15

    def info(self):
        return {'name': self.name, 'model_used_last': self.pol.last_model,
                'weights_sha256': {str(p): sha256_file(p) for p in self.paths}}


def compose_gr00t(q15, arm14, arm_kp=ARM_KP_14, arm_kd=ARM_KD_14, gravcomp_arms=True):
    """29-joint (q_target, kp, kd, gravcomp mask) = GR00T legs/waist + arm joint PD (+ robot-only gravity comp)."""
    q = np.concatenate([np.asarray(q15, float), np.asarray(arm14, float)])
    kp = np.concatenate([KP_15, np.broadcast_to(arm_kp, (14,))])
    kd = np.concatenate([KD_15, np.broadcast_to(arm_kd, (14,))])
    g = np.zeros(29, bool)
    if gravcomp_arms:
        g[15:] = True
    return q, kp, kd, g


class FalconLocomotion:
    """FALCON g1_29dof policy (port in ../falcon/falcon_policy.py): outputs all 29 targets (residual arms around
    ref_upper) with its trained MOTOR_KP/MOTOR_KD; the runner applies the PD with URDF caps (FALCON's bridge law)."""
    name = 'FALCON G1-29DoF loco-manipulation (LeCAR-Lab, a967a6d)'

    def __init__(self, onnx=FALCON_ONNX, prefill=True):
        import falcon_policy as FP
        self.FP = FP
        self.pol = FP.FalconPolicy(onnx)
        self.onnx = Path(onnx)
        self.prefill = prefill
        self.k = 0

    def reset(self):
        self.pol.reset()
        self.k = 0

    def step(self, ctx, lin_vel=(0.0, 0.0), ang_vel=0.0, stand=0, base_height=0.75, waist=(0.0, 0.0, 0.0), ref_upper=None):
        FP = self.FP
        cmd = FP.FalconCommand(lin_vel=tuple(lin_vel), ang_vel=float(ang_vel), stand=int(stand),
                               base_height=float(base_height), waist=tuple(waist), ref_upper=ref_upper)
        args = (ctx.base_quat, ctx.base_angvel_body, ctx.q29, ctx.dq29, cmd)
        if self.k == 0 and self.prefill:
            self.pol.prefill_history(*args)
        o = self.pol.step(*args)
        self.k += 1
        self.cmd = {'vx': float(lin_vel[0]), 'vy': float(lin_vel[1]), 'yaw_rate': float(ang_vel),
                    'base_height': float(base_height), 'torso_rpy': (float(waist[1]), float(waist[2]), float(waist[0]))}
        self.obs = self.pol.obs_buf.ravel().copy()
        self.action = o['raw_action'].copy()
        return o['q_target'], o['kp'], o['kd']

    def info(self):
        return {'name': self.name, 'weights_sha256': {str(self.onnx): sha256_file(self.onnx)}}


class PathFollower:
    """PI path follower: reference pelvis path (t, x, y, yaw) -> body-frame (vx, vy, yaw_rate) from the MEASURED pelvis
    pose (ideal localisation from the simulator; disclose).  Feed-forward = reference velocity lead_s ahead (the
    policies respond with ~0.3 s lag) in the body frame, feedback = kp * position error (+ ki * integral), heading P;
    saturation and acceleration limits; zero command (-> GR00T Balance model) once a stationary reference is reached
    within stop_tol."""

    def __init__(self, t_ref, xy_ref, yaw_ref, kp_pos=1.0, ki_pos=0.2, kp_yaw=2.0, v_max=(1.0, 0.35), vx_min=-0.3,
                 wz_max=1.2, accel=(1.5, 1.0), yaw_accel=4.0, i_limit=0.15, stop_tol=(0.04, math.radians(4)),
                 min_walk_speed=0.0, lead_s=0.3):
        self.t = np.asarray(t_ref, float)
        self.xy = np.asarray(xy_ref, float)
        self.yaw = np.unwrap(np.asarray(yaw_ref, float))
        self.kp, self.ki, self.kpy = kp_pos, ki_pos, kp_yaw
        self.v_max, self.vx_min, self.wz_max = v_max, vx_min, wz_max
        self.accel, self.yaw_accel, self.i_limit = accel, yaw_accel, i_limit
        self.stop_tol = stop_tol
        self.min_walk_speed = min_walk_speed
        self.lead_s = lead_s          # feed-forward uses the reference velocity lead_s ahead (policy response lag)
        self.integ = np.zeros(2)
        self.prev = np.zeros(3)

    @classmethod
    def from_reference(cls, ref=None, t0=0.0, t1=None, **kw):
        ref = load_reference() if ref is None else ref
        t = ref['time']
        sel = (t >= t0 - 1e-9) & (t <= (t1 if t1 is not None else t[-1]) + 1e-9)
        return cls(t[sel], ref['pelvis_xy'][sel], ref['pelvis_yaw'][sel], **kw)

    def ref(self, t):
        x = np.interp(t, self.t, self.xy[:, 0])
        y = np.interp(t, self.t, self.xy[:, 1])
        yaw = np.interp(t, self.t, self.yaw)
        h = 0.02
        t_a, t_b = max(self.t[0], t - h), min(self.t[-1], t + h)
        if t_b - t_a < 1e-9 or t < self.t[0] or t > self.t[-1]:
            v = np.zeros(2)
            wz = 0.0
        else:
            v = (np.array([np.interp(t_b, self.t, self.xy[:, 0]), np.interp(t_b, self.t, self.xy[:, 1])]) -
                 np.array([np.interp(t_a, self.t, self.xy[:, 0]), np.interp(t_a, self.t, self.xy[:, 1])])) / (t_b - t_a)
            wz = (np.interp(t_b, self.t, self.yaw) - np.interp(t_a, self.t, self.yaw)) / (t_b - t_a)
        return np.array([x, y]), float(yaw), v, float(wz)

    def reset(self):
        self.integ[:] = 0
        self.prev[:] = 0

    def command(self, t, xy, yaw, dt=0.02):
        p_ref, yaw_ref, _, _ = self.ref(t)
        _, _, v_ref, wz_ref = self.ref(t + self.lead_s)
        c, s = math.cos(yaw), math.sin(yaw)
        Rt = np.array([[c, s], [-s, c]])
        e_b = Rt @ (p_ref - np.asarray(xy)[:2])
        e_yaw = wrap(yaw_ref - yaw)
        v_ff = Rt @ v_ref
        stationary = np.linalg.norm(v_ref) < 1e-3 and abs(wz_ref) < 1e-3
        if stationary and np.linalg.norm(e_b) < self.stop_tol[0] and abs(e_yaw) < self.stop_tol[1]:
            self.integ[:] = 0
            des = np.zeros(3)
        else:
            self.integ = np.clip(self.integ + e_b * dt, -self.i_limit, self.i_limit)
            v = v_ff + self.kp * e_b + self.ki * self.integ
            wz = wz_ref + self.kpy * e_yaw
            des = np.array([np.clip(v[0], self.vx_min, self.v_max[0]), np.clip(v[1], -self.v_max[1], self.v_max[1]),
                            np.clip(wz, -self.wz_max, self.wz_max)])
            if self.min_walk_speed > 0 and 0 < np.linalg.norm(des[:2]) < self.min_walk_speed and not stationary:
                des[:2] *= self.min_walk_speed / np.linalg.norm(des[:2])
        lim = np.array([self.accel[0] * dt, self.accel[1] * dt, self.yaw_accel * dt])
        cmd = self.prev + np.clip(des - self.prev, -lim, lim)
        if not des.any() and np.all(np.abs(cmd) <= lim + 1e-12):
            cmd = np.zeros(3)
        self.prev = cmd
        return float(cmd[0]), float(cmd[1]), float(cmd[2]), {'pos_err_body': e_b, 'yaw_err': e_yaw, 'p_ref': p_ref,
                                                             'yaw_ref': yaw_ref}


def load_reference(path=REFERENCE_NPZ):
    r = np.load(path)
    return {k: r[k] for k in r.files}


# ---------------------------------------------------------------------------------------------- online arm IK
class SceneKin:
    """Adapter with the interface of retarget/ik_model.Kin on a SCRATCH copy of the episode model (never the
    simulated MjData).  Tool = `<side>_grasp_center` site (+ optional offset along the approach axis z)."""

    def __init__(self, model, tool_offset=0.0):
        from ik_solver import ARM_JOINTS
        m = self.m = model
        self.d = mj.MjData(model)
        jid = lambda n: m.joint(n).id  # noqa: E731
        self.arm_names = ARM_JOINTS
        assert list(ARM_JOINTS) == JOINT_NAMES_29[15:]
        self.arm_qadr = np.array([m.jnt_qposadr[jid(n)] for n in ARM_JOINTS])
        self.arm_dadr = np.array([m.jnt_dofadr[jid(n)] for n in ARM_JOINTS])
        self.arm_lo = np.array([m.jnt_range[jid(n)][0] for n in ARM_JOINTS])
        self.arm_hi = np.array([m.jnt_range[jid(n)][1] for n in ARM_JOINTS])
        self.site = {s: m.site(f'{s}_grasp_center').id for s in ('left', 'right')}
        self.site_body = {s: int(m.site_bodyid[self.site[s]]) for s in self.site}
        self.tool_offset = {'left': float(tool_offset), 'right': float(tool_offset)} if np.isscalar(tool_offset) else dict(tool_offset)
        self.body = lambda n: m.body(n).id  # noqa: E731
        self.geom_name = lambda g: m.geom(g).name or f'geom{g}'  # noqa: E731

    def set_state(self, qpos_full):
        self.d.qpos[:] = qpos_full
        self.d.qvel[:] = 0

    def set_arm(self, q):
        self.d.qpos[self.arm_qadr] = q

    def fk(self):
        mj.mj_kinematics(self.m, self.d)
        mj.mj_comPos(self.m, self.d)

    def tool(self, side):
        s = self.site[side]
        R = self.d.site_xmat[s].reshape(3, 3).copy()
        return self.d.site_xpos[s] + R[:, 2] * self.tool_offset[side], R

    def jac_tool(self, side):
        p, _ = self.tool(side)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        mj.mj_jac(self.m, self.d, jp, jr, p, self.site_body[side])
        return jp[:, self.arm_dadr], jr[:, self.arm_dadr]

    def jac_point(self, body, point):
        jp = np.zeros((3, self.m.nv))
        mj.mj_jac(self.m, self.d, jp, None, point, body)
        return jp[:, self.arm_dadr]


class SceneCollisionSet:
    """Geom groups of the episode scene for retarget/ik_solver.ArmIK collision rows."""

    def __init__(self, kin):
        m = kin.m
        bname = lambda g: m.body(m.geom_bodyid[g]).name  # noqa: E731
        coll = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])]

        def sel(pred):
            return [g for g in coll if pred(bname(g), m.geom(g).name)]
        self.groups = {}
        for s in ('left', 'right'):
            self.groups[f'{s}_hand'] = sel(lambda b, n: b.startswith(f'{s}_umi') or b in (f'{s}_wrist_pitch_link', f'{s}_wrist_yaw_link', f'{s}_wrist_roll_link'))
            self.groups[f'{s}_forearm'] = sel(lambda b, n: b == f'{s}_elbow_link')
            self.groups[f'fingers_{s}'] = sel(lambda b, n: b.startswith(f'{s}_umi_') and 'finger' in b)
        self.groups['torso'] = sel(lambda b, n: b in ('torso_link', 'head_link', 'logo_link', 'pelvis', 'pelvis_contour_link'))
        self.groups['legs'] = sel(lambda b, n: ('hip_' in b) or ('knee' in b) or ('ankle' in b))
        self.groups['table'] = sel(lambda b, n: n.startswith('table_'))
        self.groups['bag'] = sel(lambda b, n: b == 'backpack' or b.startswith('bag_strap_'))
        self.groups['bag_body'] = sel(lambda b, n: b == 'backpack')            # bag without the strap
        self.groups['strap'] = sel(lambda b, n: b.startswith('bag_strap_'))
        self.groups['recipient'] = sel(lambda b, n: b.startswith('recipient_'))
        self.groups['bag_carried'] = []

    def pairs(self, spec):
        return [(ga, gb, ds) for a, b, ds in spec for ga in self.groups[a] for gb in self.groups[b]]


NATURAL_IDX = [2, 4, 6, 9, 11, 13]      # shoulder yaw, wrist roll, wrist yaw (left, right) within the 14 arm joints
DEFAULT_PAIRS = [('left_hand', 'torso', 0.03), ('right_hand', 'torso', 0.03), ('left_hand', 'legs', 0.05),
                 ('right_hand', 'legs', 0.05), ('right_hand', 'left_hand', 0.04), ('left_forearm', 'torso', 0.02),
                 ('right_forearm', 'torso', 0.02)]


class OnlineArmIK:
    """Online damped-least-squares arm IK (retarget/ik_solver.ArmIK, OSQP) on a scratch copy of the episode model,
    with natural-range bounds (|shoulder yaw|, |wrist roll|, |wrist yaw| <= natural_limit) by default.

        ik = OnlineArmIK(runner_or_model)
        q14, info = ik.solve(ctx.qpos, {'left': dict(pos=p, axis=[0, 0, -1], wp=1.0, wr=0.1)}, q_nom=None, iters=8)
    The solve starts from the measured arm joints (or q0), keeps base/legs/waist at the measured state."""

    def __init__(self, model, natural=True, natural_limit=0.6, tool_offset=0.0, pair_spec=DEFAULT_PAIRS, margin=0.02):
        from ik_solver import ArmIK
        if hasattr(model, 'm'):
            model = model.m
        self.kin = SceneKin(model, tool_offset)
        self.col = SceneCollisionSet(self.kin)
        self.ik = ArmIK(self.kin, self.col)
        self.lo = self.kin.arm_lo + margin
        self.hi = self.kin.arm_hi - margin
        if natural:
            self.lo[NATURAL_IDX] = np.maximum(self.lo[NATURAL_IDX], -natural_limit)
            self.hi[NATURAL_IDX] = np.minimum(self.hi[NATURAL_IDX], natural_limit)
        self.pair_list = self.col.pairs(pair_spec) if pair_spec else []

    def solve(self, qpos_full, targets, q_nom=None, q0=None, iters=8, pair_list=None, **kw):
        self.kin.set_state(qpos_full)
        q_now = np.asarray(qpos_full)[self.kin.arm_qadr].copy()
        q0 = q_now if q0 is None else np.asarray(q0, float)
        q_nom = q0 if q_nom is None else np.asarray(q_nom, float)
        return self.ik.solve(np.clip(q0, self.lo, self.hi), targets, q_nom,
                             self.pair_list if pair_list is None else pair_list, iters=iters, lo=self.lo, hi=self.hi, **kw)

    def fk_tool(self, qpos_full, q14=None, side='left'):
        self.kin.set_state(qpos_full)
        if q14 is not None:
            self.kin.set_arm(q14)
        self.kin.fk()
        return self.kin.tool(side)
