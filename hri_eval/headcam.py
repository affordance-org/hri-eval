#!/usr/bin/env python3
"""v11 head camera: an ASSUMED forward-looking RGB camera in the G1 head (the head is part of torso_link).

The camera is NOT part of the physics model used for gates: the mount (body, position, quaternion, fov) lives here and
in headcam_mount.json; poses are computed from the body pose (d.xpos / d.xmat of torso_link), and renders use a
render-only copy of the model with the camera added (render_model), which leaves every kinematic/dynamic array of the
physics model unchanged (asserted).

Camera convention = MuJoCo's: the camera looks along its -z axis, +y is image up, +x is image right.
Mount frame: torso_link (+x forward, +y left, +z up). The quaternion is wxyz, camera frame -> torso_link frame.

Perception helpers (visibility gating): a point is USABLE only if it is inside the frustum (with a margin) and
unoccluded along the ray from the camera (mujoco.mj_ray on the live scene: robot, room, table, tabletop objects, bag,
recipient colliders; read-only).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------------------ the mount
BODY = 'torso_link'
POS = (0.0710, 0.0, 0.4350)       # torso_link frame: centre line, middle of the visor band, 1-2 mm proud of the face shell
ASPECT = 16.0 / 9.0               # the website's egocam panel (.hri-ego canvas aspect-ratio 16/9)
HFOV_DEG = 100.0                  # horizontal field of view (16:9 -> vertical 67.68 deg), chosen by the scan
PITCH_DEG = 15.0                  # optical axis pitched down from the torso's forward (+x) axis (chosen by the scan)


def fovy_of(hfov_deg, aspect=ASPECT):
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) / aspect))


def cam_R_body(pitch_deg):
    """camera axes (columns x_c, y_c, z_c) in the torso_link frame for a camera on the centre line looking forward,
    pitched down by pitch_deg (no roll, no yaw)."""
    p = math.radians(pitch_deg)
    xc = np.array([0.0, -1.0, 0.0])
    yc = np.array([math.sin(p), 0.0, math.cos(p)])
    zc = np.array([-math.cos(p), 0.0, math.sin(p)])
    return np.stack([xc, yc, zc], 1)


def quat_of_R(R):
    q = np.zeros(4)
    mj.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    if q[0] < 0:
        q = -q
    return q


def mount(pitch_deg=PITCH_DEG, hfov_deg=HFOV_DEG, pos=POS, aspect=ASPECT):
    R = cam_R_body(pitch_deg)
    return {'name': 'head_cam', 'body': BODY, 'pos': [float(x) for x in pos],
            'quat_wxyz': [float(x) for x in quat_of_R(R)], 'pitch_down_deg': float(pitch_deg),
            'hfov_deg': float(hfov_deg), 'fovy_deg': float(fovy_of(hfov_deg, aspect)), 'aspect': float(aspect),
            'convention': 'MuJoCo camera: looks along -z, +y up, +x right; quat wxyz camera->body frame'}


def load_mount(path=HERE / 'headcam_mount.json'):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------------------------------------ poses
def body_pose(m, d, body=BODY):
    b = m.body(body).id
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def cam_pose(m, d, mt):
    """world camera position C and rotation R (columns = camera x, y, z axes in world) from the live body pose"""
    p, Rb = body_pose(m, d, mt['body'])
    Rc = np.zeros(9)
    mj.mju_quat2Mat(Rc, np.asarray(mt['quat_wxyz'], float))
    return p + Rb @ np.asarray(mt['pos'], float), Rb @ Rc.reshape(3, 3)


def cam_pose_from_body(p, Rb, mt):
    Rc = np.zeros(9)
    mj.mju_quat2Mat(Rc, np.asarray(mt['quat_wxyz'], float))
    return p + Rb @ np.asarray(mt['pos'], float), Rb @ Rc.reshape(3, 3)


def optical_axis_pitch_deg(R):
    """degrees below the horizon of the optical axis (-z_cam)"""
    f = -R[:, 2]
    return math.degrees(math.asin(-f[2]))


# ------------------------------------------------------------------------------------------------ projection
def project(C, R, X, fovy_deg, aspect=ASPECT):
    """normalised image coords (u right, v up; |u|, |v| <= 1 inside the frame) and depth along the optical axis"""
    X = np.atleast_2d(X)
    pc = (X - C) @ R
    depth = -pc[..., 2]
    ty = math.tan(math.radians(fovy_deg) / 2)
    tx = ty * aspect
    with np.errstate(divide='ignore', invalid='ignore'):
        u = pc[..., 0] / depth / tx
        v = pc[..., 1] / depth / ty
    return u, v, depth


def in_frustum(u, v, depth, margin=0.05, near=0.05):
    """inside the frame with a margin (fraction of the half width / half height)"""
    return (depth > near) & (np.abs(u) <= 1 - margin) & (np.abs(v) <= 1 - margin)


# ------------------------------------------------------------------------------------------------ occlusion
# geom groups the ray sees: 0 (room, table, objects, recipient colliders, robot decorations), 1, 2 (robot and bag
# visual meshes).  Group 3 (collision duplicates of the same meshes) is skipped.
RAY_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)


class Occlusion:
    """mj_ray visibility on a (scratch) MjData.  The torso_link body (the head shell behind the lens) is excluded."""

    def __init__(self, m, groups=RAY_GROUPS, exclude_body=BODY):
        self.m = m
        self.groups = np.asarray(groups, np.uint8)
        self.excl = m.body(exclude_body).id
        self.gid = np.zeros(1, np.int32)
        self.gbody = m.geom_bodyid
        self.recip = np.array([m.body(b).name.startswith('recipient_') for b in range(m.nbody)])

    def first_hit(self, d, C, X):
        v = np.asarray(X, float) - C
        dist = mj.mj_ray(self.m, d, C, v, self.groups, 1, self.excl, self.gid)
        return float(dist), int(self.gid[0])          # dist in units of |v|: 1.0 = at the point; -1 = no hit

    def point_visible(self, d, C, X, surface_r=0.0, tol=0.01, own=(), own_r=0.15):
        """A keypoint (a joint centre, inside the limb) is visible if the first thing the camera ray hits is
          (a) nothing before the point (within tol m), or
          (b) the surface of the keypoint's OWN limb segments (body names in `own`, e.g. the forearm and hand
              colliders for the wrist) within own_r of the joint: the camera sees that joint's own limb there, or
          (c) (surface_r > 0) any recipient collider within surface_r of the point.
        Anything else in front (the other arm, the torso, an object, the robot's own arm, room furniture) occludes."""
        v = np.asarray(X, float) - C
        L = float(np.linalg.norm(v))
        dist, g = self.first_hit(d, C, X)
        if dist < 0 or dist * L >= L - tol:
            return True, dist, g
        if g >= 0:
            Hp = C + dist * v
            e = float(np.linalg.norm(Hp - X))
            b = int(self.gbody[g])
            if own and self.m.body(b).name in own and e <= own_r:
                return True, dist, g
            if surface_r > 0 and self.recip[b] and e <= surface_r:
                return True, dist, g
        return False, dist, g

    def object_visible_fraction(self, d, C, pts, owner_bodies):
        """fraction of sample points whose first hit along the camera ray is the object itself (or nothing before the
        point): occlusion by anything else (robot arm, person, other objects, room) counts as hidden."""
        n_ok = 0
        for X in pts:
            v = X - C
            dist = mj.mj_ray(self.m, d, C, v, self.groups, 1, self.excl, self.gid)
            g = int(self.gid[0])
            if dist < 0 or dist >= 1 - 1e-3 or (g >= 0 and int(self.gbody[g]) in owner_bodies):
                n_ok += 1
        return n_ok / max(1, len(pts))


# the keypoint's own limb segments (recipient colliders; the avatar's collider set has no head, neck or upper-arm piece)
OWN = {'shoulder_r': ('recipient_spine_03',), 'shoulder_l': ('recipient_spine_03',),
       'elbow_r': ('recipient_lowerarm_r',), 'elbow_l': ('recipient_lowerarm_l',),
       'wrist_r': ('recipient_lowerarm_r', 'recipient_hand_r'), 'wrist_l': ('recipient_lowerarm_l', 'recipient_hand_l'),
       'hand_r': ('recipient_hand_r',), 'neck': ('recipient_spine_03',),
       'chest': ('recipient_spine_03', 'recipient_spine_02'),
       'spine': ('recipient_spine_02', 'recipient_spine_01', 'recipient_spine_03'),
       'pelvis': ('recipient_pelvis', 'recipient_spine_01', 'recipient_thigh_l', 'recipient_thigh_r'),
       'hip_l': ('recipient_pelvis', 'recipient_thigh_l'), 'hip_r': ('recipient_pelvis', 'recipient_thigh_r'),
       'knee_l': ('recipient_thigh_l', 'recipient_calf_l'), 'knee_r': ('recipient_thigh_r', 'recipient_calf_r'),
       'ankle_l': ('recipient_calf_l', 'recipient_foot_l'), 'ankle_r': ('recipient_calf_r', 'recipient_foot_r')}

# how far the seen surface of the keypoint's own segment may be from the joint centre (the recording's spine joints
# lie near the back, about 0.2 m behind the chest surface; limb joints sit inside 0.03-0.1 m thick limbs)
OWN_R = {'chest': 0.25, 'spine': 0.25, 'pelvis': 0.25, 'neck': 0.18, 'shoulder_r': 0.15, 'shoulder_l': 0.15}
OWN_R_DEFAULT = 0.15

# keypoint "surface radius": distance from the joint centre to the limb surface that a detector would see
SURFACE_R = {'head': 0.13, 'head_top': 0.06, 'neck': 0.10, 'shoulder_r': 0.10, 'shoulder_l': 0.10, 'elbow_r': 0.07,
             'elbow_l': 0.07, 'wrist_r': 0.06, 'wrist_l': 0.06, 'hand_r': 0.03, 'chest': 0.15, 'spine': 0.15,
             'pelvis': 0.15}


def object_samples(m, d, body_or_geom, kind, n=48, seed=0):
    """world sample points on an object's collision meshes (vertices, subsampled deterministically)"""
    rng = np.random.default_rng(seed)
    if kind == 'body':
        b = m.body(body_or_geom).id
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH
                 and (m.geom_contype[g] or m.geom_conaffinity[g])]
    else:
        geoms = [m.geom(body_or_geom).id]
    V = []
    for g in geoms:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        V.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    V = np.vstack(V)
    if len(V) > n:
        V = V[rng.choice(len(V), n, replace=False)]
    return V, {int(m.geom_bodyid[g]) for g in geoms}


# ------------------------------------------------------------------------------------------------ render model
def render_model(xml_path, mt, offw=1280, offh=720):
    """compile a RENDER-ONLY copy of the scene with the head camera added to its body.  Asserts that nq, nv, nbody,
    ngeom, body poses, masses and joint layout are identical to the physics model (a camera adds no mass, contact or
    actuation), so trajectory qpos rows replay unchanged."""
    spec = mj.MjSpec.from_file(str(xml_path))
    b = spec.body(mt['body'])
    c = b.add_camera()
    c.name = mt.get('name', 'head_cam')
    c.pos = list(mt['pos'])
    c.quat = list(mt['quat_wxyz'])
    c.fovy = float(mt['fovy_deg'])
    mr = spec.compile()
    m0 = mj.MjModel.from_xml_path(str(xml_path))
    for f in ('nq', 'nv', 'nu', 'nbody', 'ngeom', 'njnt', 'nmocap'):
        assert getattr(mr, f) == getattr(m0, f), f
    for f in ('body_pos', 'body_quat', 'body_mass', 'body_inertia', 'jnt_qposadr', 'geom_size', 'geom_pos'):
        assert np.array_equal(getattr(mr, f), getattr(m0, f)), f
    mr.vis.global_.offwidth = max(mr.vis.global_.offwidth, offw)
    mr.vis.global_.offheight = max(mr.vis.global_.offheight, offh)
    return mr


if __name__ == '__main__':
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    mt = mount(float(args[0]) if len(args) > 0 else PITCH_DEG, float(args[1]) if len(args) > 1 else HFOV_DEG)
    Rb = cam_R_body(mt['pitch_down_deg'])
    mt['quat_xyzw'] = [mt['quat_wxyz'][1], mt['quat_wxyz'][2], mt['quat_wxyz'][3], mt['quat_wxyz'][0]]
    mt['forward_axis_in_body'] = [float(x) for x in -Rb[:, 2]]
    mt['up_axis_in_body'] = [float(x) for x in Rb[:, 1]]
    mt['assumed_sensor'] = ('ASSUMED head camera: not on the stock G1 (whose only head camera is the D435 pitched 47.6 deg '
                            'down, fovy 58). A forward-looking wide-angle RGB camera behind the visor band of the face, '
                            'on the head centre line, 13.4 mm in front of, 17.5 mm right of and 5.1 mm above the stock '
                            'D435 optical centre (1-2 mm proud of the visor surface)')
    mt['not_in_physics_model'] = ('a camera adds no mass, contact or actuation; it is kept OUT of the physics/gate model '
                                  '(trial.xml unchanged); poses come from torso_link and this mount; renders use '
                                  'headcam.render_model (a render-only copy with the camera added)')
    print(json.dumps(mt, indent=1))
    if '--write' in sys.argv:
        (HERE / 'headcam_mount.json').write_text(json.dumps(mt, indent=1) + '\n')
    # self check: the stock D435 quaternion is the 47.6 deg version of this construction (+ its y offset)
    q = quat_of_R(cam_R_body(47.6))
    print('47.6 deg construction', np.round(q, 6), 'stock ego_d435 quat (0.659252 0.255707 -0.255707 -0.659252)')
