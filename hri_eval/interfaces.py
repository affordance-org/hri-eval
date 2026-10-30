"""Command-level policy interface.

Subclass CommandPolicy and implement act(obs) -> Command. run.py wraps such a class in CommandAdapter, which runs the
NVIDIA GR00T WBC walking policy for the legs and waist and joint PD (with robot-only gravity compensation) for the
arms, at 50 Hz. Run it with python run.py --policy MODULE:CLASS.

    import interfaces as I

    class MyPolicy(I.CommandPolicy):
        def act(self, obs):
            return I.Command(vx=0.2, grippers=(0.0, 0.0))

obs (a dict, every 50 Hz tick; numpy arrays in world coordinates, metres, radians, quaternions wxyz):
    t           simulated time [s]
    q29, dq29   joint positions [rad] and velocities [rad/s] in episode.JOINT_NAMES_29 order (legs 0-11, waist 12-14,
                left arm 15-21, right arm 22-28)
    base_pos    pelvis position;  base_quat  pelvis orientation;  base_yaw  pelvis heading
    grippers    measured closing of the (left, right) gripper, 0 open, 1 at the closed end of the servo range (empty
                and fully closed reads about 0.90: the pads meet first)
    person      {keypoint: xyz} of the recorded person: PERSON_KEYPOINTS (joint centres; hand_l, hand_r = palm centres)
    objects     {name: {'pos', 'quat', 'center'}} for 'backpack' (free, moves) and 'water_bottle', 'cup', 'candy_box'
                (fixed to the table): pos, quat = the object frame (backpack: its body frame, origin at the bottom
                centre; the others: their collision mesh frame), center = centre of the collision mesh bounding box

Command fields:
    vx, vy, yaw_rate  base velocity command to GR00T [m/s, m/s, rad/s] in the pelvis heading frame, sent as given;
                      a command with |(vx, vy, yaw_rate)| < 0.05 runs the GR00T Balance model (stand in place)
    base_height       GR00T base height command [m], default 0.74
    arms              14 arm joint targets [rad] (left arm 7 then right arm 7, JOINT_NAMES_29[15:] order), clipped to the
                      joint limits; None holds the last arm target (at the start: the start pose)
    grippers          (left, right) in [0, 1]: 0 open, 1 closed (force-limited servos, 35 N per finger); None holds
                      the last gripper command
    phase             optional label; a change is written to events.json

act() may return a Command, a dict of these fields, or None (= Command()). Command() and None open both grippers
(grippers defaults to (0, 0)); pass grippers=None to keep the current grip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

import episode as E
from fetch_present import KEYPOINTS
from recipient import q2R

PERSON_KEYPOINTS = [k for k, _ in KEYPOINTS] + ['hand_l', 'hand_r']
TABLE_OBJECTS = ('water_bottle', 'cup', 'candy_box')


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    arms: object = None
    grippers: object = (0.0, 0.0)
    base_height: float = 0.74
    phase: str = ''


class CommandPolicy:
    """Base class of command-level policies. Override act(obs); reset(obs) is called once before the first tick."""

    def reset(self, obs):
        pass

    def act(self, obs):
        raise NotImplementedError

    def describe(self):
        return ''

    def disclosures(self):
        return []


def is_command_policy(cls):
    return any(c.__name__ == 'CommandPolicy' for c in getattr(cls, '__mro__', ())) or \
        (hasattr(cls, 'act') and not hasattr(cls, 'policy_step'))


def person_keypoints(rcp, t):
    P, Q, _, _ = rcp.full_pose(t)
    rec = rcp.rec
    out = {}
    for name, bone in KEYPOINTS:
        if bone:
            out[name] = P[rec.idx[bone]].copy()
        else:
            hi = rec.idx['head']
            out[name] = P[hi] + q2R(Q[hi]) @ (rec.rest_tail[hi] - rec.rest_head[hi])
    for s in ('l', 'r'):
        out[f'hand_{s}'] = np.asarray(rcp.hand_frame(t, s)[0], float).copy()
    return out


class CommandAdapter:
    """Joint-level controller (reset / policy_step -> episode.ControlOutput) around a CommandPolicy."""

    def __init__(self, policy):
        self.policy = policy
        self.loco = E.Gr00tLocomotion()

    def reset(self, ctx):
        m = ctx.m
        self.loco.reset()
        self.arm = ctx.q29[15:].copy()
        jr = np.array([m.jnt_range[m.joint(n).id] for n in E.JOINT_NAMES_29[15:]])
        self.arm_lo, self.arm_hi = jr[:, 0], jr[:, 1]
        fa = [next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == m.joint(n).id) for n in E.FINGER_JOINTS]
        self.f_lo, self.f_hi = m.actuator_ctrlrange[fa, 0].copy(), m.actuator_ctrlrange[fa, 1].copy()
        self.f_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in E.FINGER_JOINTS])
        self.bag_q = int(m.jnt_qposadr[m.body('backpack').jntadr[0]])
        bag = m.body('backpack').id
        V = np.vstack([mesh_verts(m, g) @ quat_R(m.geom_quat[g]).T + m.geom_pos[g] for g in range(m.ngeom)
                       if m.geom_bodyid[g] == bag and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH
                       and (m.geom_contype[g] or m.geom_conaffinity[g])])
        self.bag_c = 0.5 * (V.min(0) + V.max(0))                 # bounding-box centre in the bag frame
        self.static = {}
        for o in TABLE_OBJECTS:
            g = m.geom(f'object_{o}')
            W = mesh_verts(m, g.id) @ quat_R(g.quat).T + g.pos
            self.static[o] = {'pos': g.pos.copy(), 'quat': g.quat.copy(), 'center': 0.5 * (W.min(0) + W.max(0))}
        self.grip = np.zeros(2)
        if hasattr(self.policy, 'reset'):
            self.policy.reset(self.obs(ctx))

    def obs(self, ctx):
        g = (ctx.qpos[self.f_qadr] - self.f_lo) / (self.f_hi - self.f_lo)
        q = ctx.qpos
        bp, bq = q[self.bag_q:self.bag_q + 3].copy(), q[self.bag_q + 3:self.bag_q + 7].copy()
        objects = {'backpack': {'pos': bp, 'quat': bq, 'center': bp + quat_R(bq) @ self.bag_c}}
        objects.update({o: {k: x.copy() for k, x in v.items()} for o, v in self.static.items()})
        return {'t': ctx.t, 'q29': ctx.q29.copy(), 'dq29': ctx.dq29.copy(), 'base_pos': ctx.base_pos.copy(),
                'base_quat': ctx.base_quat.copy(), 'base_yaw': ctx.base_yaw,
                'grippers': np.clip([g[:2].mean(), g[2:].mean()], 0.0, 1.0),
                'person': person_keypoints(ctx.recipient, ctx.t) if ctx.recipient is not None else {},
                'objects': objects}

    def policy_step(self, ctx):
        c = self.policy.act(self.obs(ctx))
        c = Command() if c is None else Command(**c) if isinstance(c, dict) else c
        v = np.array([c.vx, c.vy, c.yaw_rate, c.base_height], float)
        if not np.all(np.isfinite(v)):
            raise ValueError(f'non-finite command {v}')
        q15 = self.loco.step(ctx, v[0], v[1], v[2], base_height=v[3])
        if c.arms is not None:
            a = np.asarray(c.arms, float).reshape(14)
            if not np.all(np.isfinite(a)):
                raise ValueError('non-finite arm targets')
            self.arm = np.clip(a, self.arm_lo, self.arm_hi)
        g = np.clip(np.asarray(c.grippers if c.grippers is not None else self.grip, float).reshape(2), 0.0, 1.0)
        self.grip = g
        fingers = self.f_lo + np.repeat(g, 2) * (self.f_hi - self.f_lo)
        q, kp, kd, gc = E.compose_gr00t(q15, self.arm)
        return E.ControlOutput(q_target=q, kp=kp, kd=kd, gravcomp=gc, fingers=fingers, cmd=self.loco.cmd,
                               policy_obs=self.loco.obs, policy_action=self.loco.action, phase=c.phase or 'command',
                               log={'cmd_grippers': g})

    def policy_info(self):
        return self.loco.info() | {'interface': 'command', 'policy_class': type(self.policy).__name__}

    def describe(self):
        d = self.policy.describe() if hasattr(self.policy, 'describe') else ''
        return (f'Command-level policy {type(self.policy).__name__} (interfaces.CommandPolicy): base velocity commands to '
                'GR00T WBC (legs, waist), arm joint targets by PD with robot-only gravity compensation, gripper '
                'commands to the force-limited finger servos.' + (f' {d}' if d else ''))

    def disclosures(self):
        return list(self.policy.disclosures()) if hasattr(self.policy, 'disclosures') else []


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_R(q):
    return q2R(np.asarray(q, float) / np.linalg.norm(q))


def mesh_verts(m, g):
    i = m.geom_dataid[g]
    return m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]].astype(float)
