#!/usr/bin/env python3
"""Recipient (the person) = REPLAYED RECORDING, retimed to robot time, with an adapted LEFT-arm reach and a
kinematic two-pad pinch hand.  Pure kinematics; nothing here touches the robot or the bag.

What it does
  (a) Retiming     robot time t [s] -> source frame s(t) of the 30 fps recording.  Piecewise linear, append-only,
                   playback rate r in [0, 1] (r = 0 holds a pose, r = 1 is real time; never faster).
  (b) Reach        analytic 2-bone IK on upperarm_l / lowerarm_l (+ hand_l) with the shoulder (upperarm_l head)
                   fixed, so the PINCH CENTRE of his left hand reaches a world target; the elbow keeps the
                   recorded swivel (closest point of the elbow circle to the recorded elbow) + optional offset;
                   optional world orientation target for the hand.  Blended in/out with a smooth weight
                   w(t) in [0, 1] (per-bone slerp between the recorded and the adapted rotation).
  (c) Pads         palm pad (fixed on the palm) and thumb pad (moves along the palm normal) poses from the adapted
                   hand bone; gap(t) from gap_open to gap_closed (closure distance parameterised).
  (d) Output       recipient-motion.json in h2r-avatar-motion/1 (same 21 bones, adapted rotations AND recomputed
                   child positions, 30 fps in ROBOT time, plus the robot time -> source frame mapping).

All schedules (retiming, weight, target, hand orientation, gap) are APPEND-ONLY knot tracks: a controller may
extend them online (knots only at t >= now), so evaluating the tracks after the episode reproduces exactly what was
applied during it (the exported motion == the colliders that were simulated).

Skinning convention of the recording: world = R_i (v - rest_head_i) + p_i, R_i/p_i in the room frame.  The avatar's
hand_l rotation is a copy of lowerarm_l in every recorded frame (no hand orientation was captured).

CLI:  python recipient.py --validate   (writes recipient_check/{validation.json, *.png})
"""
from __future__ import annotations

import bisect
import copy
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MOTION = Path(__file__).resolve().parent / 'person' / 'point_fetch_01.json'
GEOMETRY_JSON = HERE / 'recipient_geometry.json'      # written by build_variant_scene.py (v3 copy)
FPS = 30.0
START_FRAME = 120                                     # robot t = 0 <-> recording frame 120 (scene keyframe 'start')
ARM_L = ('upperarm_l', 'lowerarm_l', 'hand_l')


# ----------------------------------------------------------------------------------------- quaternion helpers (wxyz)
def qnorm(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q2R(q):
    w, x, y, z = qnorm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R2q(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if w >= 0 else -q


def slerp(a, b, u):
    a, b = qnorm(a), qnorm(b)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return qnorm(a + u * (b - a))
    th = math.acos(min(1.0, d))
    s = math.sin(th)
    return (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / s


def rot_between(u, v):
    """Minimal rotation matrix taking direction u to direction v."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    c = float(np.dot(u, v))
    ax = np.cross(u, v)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(u, p)
        ax /= np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def axis_angle_R(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def twist_about(R, axis):
    """Twist angle of rotation R about `axis` (swing-twist decomposition)."""
    q = R2q(R)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    p = np.dot(q[1:], a)
    return 2.0 * math.atan2(p, q[0])


def smoothstep(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


# ----------------------------------------------------------------------------------------- append-only schedules
class KnotTrack:
    """Append-only piecewise schedule of vectors.  value(t) holds the first/last knot outside the knot span and
    interpolates inside ('linear' or 'smooth' = smoothstep within each segment, C1 at knots).
    append(now, t_knot, v): t_knot >= now; first freezes the past by inserting (now, value(now)) so values at
    times <= now never change."""

    def __init__(self, v0, interp='linear', name=''):
        self.t = [0.0]
        self.v = [np.atleast_1d(np.asarray(v0, float)).copy()]
        self.interp = interp
        self.name = name

    def value(self, t):
        ts = self.t
        if t <= ts[0]:
            return self.v[0].copy()
        if t >= ts[-1]:
            return self.v[-1].copy()
        i = bisect.bisect_right(ts, t) - 1
        u = (t - ts[i]) / (ts[i + 1] - ts[i])
        if self.interp == 'smooth':
            u = smoothstep(u)
        return self.v[i] + u * (self.v[i + 1] - self.v[i])

    def append(self, now, t_knot, v):
        v = np.atleast_1d(np.asarray(v, float)).copy()
        if t_knot < now - 1e-12:
            raise ValueError(f'{self.name}: knot at {t_knot} is in the past (now {now})')
        if now > self.t[-1] + 1e-12:
            self.t.append(float(now))
            self.v.append(self.value(now))
        if abs(t_knot - self.t[-1]) < 1e-12:
            if t_knot <= now + 1e-12 and len(self.t) > 1:
                # a step change "now" would rewrite the value applied at `now`; shift it by one physics step
                t_knot = self.t[-1] + 0.002
            else:
                self.v[-1] = v
                return
        self.t.append(float(t_knot))
        self.v.append(v)

    def to_json(self):
        return {'interp': self.interp, 'knots': [[round(t, 6), [round(float(x), 7) for x in v]] for t, v in zip(self.t, self.v)]}


class Retiming:
    """Robot time -> recording source frame.  Segments of constant playback rate r in [0, 1]
    (frames advance at r * 30 per second; r = 0 holds the pose).  Append-only (set_rate(now, r))."""

    def __init__(self, start_frame=START_FRAME, rate=0.0, fps=FPS, last_frame=None):
        self.fps = fps
        self.seg = [(0.0, float(start_frame), float(rate))]   # (t0, s0, rate)
        self.last_frame = last_frame

    def __call__(self, t):
        i = bisect.bisect_right([s[0] for s in self.seg], t) - 1
        t0, s0, r = self.seg[max(i, 0)]
        s = s0 + self.fps * r * (t - t0)
        if self.last_frame is not None:
            s = min(s, float(self.last_frame))
        return s

    def set_rate(self, now, rate):
        if not (-1e-12 <= rate <= 1.0 + 1e-12):
            raise ValueError('playback rate must be in [0, 1] (hold or real time; never faster than the recording)')
        if now < self.seg[-1][0] - 1e-12:
            raise ValueError('retiming is append-only')
        s_now = self(now)
        if abs(now - self.seg[-1][0]) < 1e-12:
            self.seg[-1] = (self.seg[-1][0], self.seg[-1][1], float(rate))
        else:
            self.seg.append((float(now), s_now, float(rate)))

    @classmethod
    def from_plan(cls, start_frame, plan, fps=FPS):
        """plan = [(duration_s, rate), ...] executed from t = 0; the last rate continues."""
        r = cls(start_frame, plan[0][1] if plan else 1.0, fps)
        t = 0.0
        for k, (dur, rate) in enumerate(plan):
            if k:
                r.set_rate(t, rate)
            t += dur
        return r

    def time_of_frame(self, s):
        """Earliest robot time at which the CURRENT schedule reaches source frame s (None if it never does)."""
        for k, (t0, s0, r) in enumerate(self.seg):
            t1 = self.seg[k + 1][0] if k + 1 < len(self.seg) else math.inf
            if s <= s0 + 1e-9:
                return t0
            if r > 0:
                t = t0 + (s - s0) / (self.fps * r)
                if t <= t1:
                    return t
        return None

    def to_json(self):
        return {'fps': self.fps, 'segments': [{'t0': round(t0, 6), 'source_frame0': round(s0, 6), 'rate': r}
                                              for t0, s0, r in self.seg],
                'rule': 'source_frame(t) = source_frame0 + fps * rate * (t - t0); 0 <= rate <= 1 (hold or real time)'}


# ----------------------------------------------------------------------------------------- recording
class Recording:
    def __init__(self, path=MOTION):
        self.path = Path(path)
        self.raw = json.loads(self.path.read_text())
        assert self.raw['format'] == 'h2r-avatar-motion/1'
        self.bones = self.raw['bones']
        self.names = [b['name'] for b in self.bones]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.rest_head = np.array([b['rest_head'] for b in self.bones], float)
        self.rest_tail = np.array([b['rest_tail'] for b in self.bones], float)
        fr = self.raw['frames']
        assert [f['frame'] for f in fr] == list(range(len(fr))) and all(f['valid'] for f in fr)
        self.P = np.array([f['position'] for f in fr], float)        # (F, 21, 3)
        self.Q = np.array([f['rotation_wxyz'] for f in fr], float)   # (F, 21, 4)
        self.wall_ns = [f['wall_ns'] for f in fr]
        self.nframes = len(fr)

    def at(self, s):
        """Bone positions/rotations at a (fractional) source frame: exact copy at integer frames, otherwise
        linear position / slerp rotation between the two neighbours."""
        s = min(max(float(s), 0.0), self.nframes - 1.0)
        i = int(math.floor(s + 1e-9))
        u = s - i
        if u < 1e-9 or i >= self.nframes - 1:
            return self.P[i].copy(), self.Q[i].copy()
        P = (1 - u) * self.P[i] + u * self.P[i + 1]
        Q = np.array([slerp(self.Q[i, b], self.Q[i + 1, b], u) for b in range(len(self.names))])
        return P, Q

    def at_bones(self, s, names):
        s = min(max(float(s), 0.0), self.nframes - 1.0)
        i = int(math.floor(s + 1e-9))
        u = s - i
        ids = [self.idx[n] for n in names]
        if u < 1e-9 or i >= self.nframes - 1:
            return self.P[i, ids].copy(), self.Q[i, ids].copy()
        P = (1 - u) * self.P[i, ids] + u * self.P[i + 1, ids]
        Q = np.array([slerp(self.Q[i, b], self.Q[i + 1, b], u) for b in ids])
        return P, Q


# ----------------------------------------------------------------------------------------- hand / pinch frame
@dataclass
class PinchParams:
    gap_open: float = 0.070            # m, distance palm-pad face -> thumb-pad face when open
    # m, closure distance.  The handle is 0.02462 m thick; contacts with these kinematic pads are VERY stiff
    # (tests/test_pad_grip.py --air: 0.0244 -> ~290 N total squeeze, holds the hanging bag with 0.1 mm slip;
    # 0.0246 -> 34 N, 6.6 mm slip; 0.0230 -> 9 kN).  Align pinch x with the handle thickness axis.
    gap_closed: float = 0.0244
    palm_half: tuple = (0.008, 0.035, 0.040)     # palm pad box half sizes (x = palm normal, y = across palm, z = fingers)
    thumb_half: tuple = (0.008, 0.011, 0.020)    # thumb pad box half sizes
    distal_offset: float = 0.0         # m, pinch centre shifted from the hand-box centre toward the fingers
    # palm side = -(thinnest PCA axis of the hand-skin box as stored in recipient_geometry.json): verified on the
    # avatar skin (recipient_check/hand_local_pinch_frame.png): the thumb and the finger curl lie on that side and
    # the knuckles bulge on the other.
    palm_sign: float = -1.0
    thumb_offset_yz: tuple = (0.0, 0.0)
    pad_shift_z: float = 0.0           # m, both pads shifted along pinch z (+ = toward the fingertips) relative to the
                                       # pinch centre, e.g. -0.03 so the pads overlap only the top of a handle that
                                       # stands ~15-20 mm proud of the bag body


@dataclass
class HookParams:
    """v3 'hook' receiving hand (kinematic): palm box + curled-finger hook (capsules), all in ONE mocap body
    `recipient_hook_hand` whose frame is the HOOK FRAME (same axes as the pinch frame: x = palm normal, y = across the
    palm, z = toward the fingertips).  Its origin is the HOOK SEAT: the point on the palm-side surface of the finger
    bars where the strap's loop top bar rests (target of set_target).  Use palm-up (x = world up) with z through the
    strap loop (along the loop normal) to carry the bag by normal force.

    geoms (hook frame):  fingers k: capsule radius finger_r along z, centre line x = -finger_r, y = (k - (n-1)/2) *
    finger_spacing, z in [-finger_len/2, +finger_len/2];  tips k: capsule from the finger end (x=-r, z=+len/2) to
    (x = -r + curl_up, z = len/2 + curl_fwd) (curled fingertips retain the loop);  palm: box half palm_half centred at
    (x = -palm_half[0], 0, z = -finger_len/2 - palm_half[2]) (palm-side face flush with the seat, wider than the loop
    opening so the loop cannot slide off toward the wrist)."""
    n_fingers: int = 2
    finger_r: float = 0.0065
    finger_spacing: float = 0.013
    finger_len: float = 0.040
    curl_up: float = 0.014
    curl_fwd: float = 0.008
    palm_half: tuple = (0.010, 0.018, 0.040)     # narrow (index+middle) palm block: clears a UMI pinching the strap tab
    seat_z: float = 0.040              # seat along hand z from the skin-fit hand box centre (finger region)
    seat_x: float | None = None        # seat along the palm normal from the box centre (default: the palm face)

    def geoms(self):
        """[(name, type, params)] in the hook frame; type 'capsule' (fromto, r) or 'box' (pos, half)."""
        out = []
        r, L = self.finger_r, self.finger_len
        for k in range(self.n_fingers):
            y = (k - (self.n_fingers - 1) / 2) * self.finger_spacing
            out.append((f'recipient_hook_finger{k}', 'capsule', {'fromto': [-r, y, -L / 2, -r, y, L / 2], 'r': r}))
            out.append((f'recipient_hook_tip{k}', 'capsule',
                        {'fromto': [-r, y, L / 2, -r + self.curl_up, y, L / 2 + self.curl_fwd], 'r': r}))
        ph = self.palm_half
        out.append(('recipient_hook_palm', 'box', {'pos': [-ph[0], 0.0, -L / 2 - ph[2]], 'half': list(ph)}))
        return out


def hook_frame_in_hand(geom, rec, hp: HookParams, pp):
    """Hook frame (axes = pinch frame) and seat origin in the hand_l bone-local frame."""
    R_ph, _, info = pinch_frame_in_hand(geom, rec, pp)
    hb = geom['hand_l']
    half = np.asarray(hb['half_size'], float)
    centre = np.asarray(hb['pos'], float)
    k_thin = int(np.argmin(half))
    sx = half[k_thin] if hp.seat_x is None else hp.seat_x
    c = centre + R_ph[:, 0] * sx + R_ph[:, 2] * hp.seat_z
    return R_ph, c, dict(info, seat_rule='hand box centre + palm normal * seat_x + finger axis * seat_z')


def load_geometry():
    if GEOMETRY_JSON.exists():
        return json.loads(GEOMETRY_JSON.read_text())
    import sys
    sys.path.insert(0, str(HERE.parent / 'scene'))
    import build_scene as B      # read-only import
    g = B.recipient_geometry()
    return json.loads(json.dumps(g, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o))


def pinch_frame_in_hand(geom, rec: Recording, pp: PinchParams):
    """Pinch frame (R_ph columns x = palm normal toward the object, y = across the palm, z = toward the fingers;
    origin c_ph = pinch centre at closure) in the hand_l bone-local frame (world = R_hand x + p_hand)."""
    hb = geom['hand_l']
    Rb = q2R(hb['quat'])
    half = np.asarray(hb['half_size'], float)
    centre = np.asarray(hb['pos'], float)
    i_hand = rec.idx['hand_l']
    fwd = rec.rest_tail[rec.idx['lowerarm_l']] - rec.rest_head[rec.idx['lowerarm_l']]
    fwd = fwd + (rec.rest_tail[i_hand] - rec.rest_head[i_hand])
    k_thin = int(np.argmin(half))
    rest = [k for k in range(3) if k != k_thin]
    k_len = max(rest, key=lambda k: abs(np.dot(Rb[:, k], fwd)))
    z = Rb[:, k_len] * np.sign(np.dot(Rb[:, k_len], fwd))
    x = Rb[:, k_thin] * pp.palm_sign
    x = x - np.dot(x, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_ph = np.stack([x, y, z], 1)
    c = centre + x * (half[k_thin] + pp.gap_closed / 2) + z * pp.distal_offset
    return R_ph, c, {'hand_box_half': half.tolist(), 'thin_axis': k_thin, 'length_axis': k_len,
                     'hand_box_centre_local': centre.tolist()}


# ----------------------------------------------------------------------------------------- the recipient
class Recipient:
    """Replayed the person with retiming, adapted left-arm reach and a kinematic pinch hand."""

    MOCAP_BODIES = ('recipient_lowerarm_l', 'recipient_palm_pad', 'recipient_thumb_pad',
                    'recipient_lowerarm_r', 'recipient_hand_r')
    MOCAP_BODIES_HOOK = ('recipient_lowerarm_l', 'recipient_hook_hand', 'recipient_lowerarm_r', 'recipient_hand_r')

    def __init__(self, retiming: Retiming | None = None, pinch: PinchParams | None = None, motion_path=MOTION,
                 geometry=None, hand='pinch', hook: HookParams | None = None):
        """hand = 'pinch' (v2 two-pad pinch: recipient_palm_pad + recipient_thumb_pad) or 'hook' (v3:
        recipient_hook_hand, a palm + curled-finger hook; set_target/orient_hand then act on the HOOK FRAME)."""
        assert hand in ('pinch', 'hook')
        self.hand = hand
        self.rec = Recording(motion_path)
        self.retiming = retiming or Retiming(START_FRAME, rate=1.0, last_frame=self.rec.nframes - 1)
        if self.retiming.last_frame is None:
            self.retiming.last_frame = self.rec.nframes - 1
        self.pp = pinch or PinchParams()
        self.hp = hook or HookParams()
        self.geom = geometry or load_geometry()
        if hand == 'hook':
            self.R_ph, self.c_ph, self.pinch_info = hook_frame_in_hand(self.geom, self.rec, self.hp, self.pp)
        else:
            self.R_ph, self.c_ph, self.pinch_info = pinch_frame_in_hand(self.geom, self.rec, self.pp)
        self.mocap_bodies = self.MOCAP_BODIES_HOOK if hand == 'hook' else self.MOCAP_BODIES
        self.weight = KnotTrack(0.0, 'smooth', 'reach_weight')
        self.target = KnotTrack(np.zeros(3), 'linear', 'reach_target')
        self.has_target = False
        self.hand_q = KnotTrack(np.array([1.0, 0, 0, 0]), 'linear', 'hand_orientation')   # world pinch-frame quat
        self.hand_w = KnotTrack(0.0, 'smooth', 'hand_orientation_weight')
        self.swivel = KnotTrack(0.0, 'linear', 'swivel_deg')
        self.gap = KnotTrack(self.pp.gap_open, 'smooth', 'pinch_gap')
        self.ids_arm = [self.rec.idx[n] for n in ARM_L]
        self.rh = {n: self.rec.rest_head[self.rec.idx[n]] for n in self.rec.names}
        self._cache_t = None
        self.notes = []

    # ---------------------------------------------------------------- online / planned schedule edits
    def set_target(self, now, p, at=None):
        """Reach target (world point for the pinch centre) reached at time `at` (default now + 0.02 s, i.e. a
        one-policy-tick linear ramp from the current target)."""
        p = np.asarray(p, float)
        if not self.has_target:
            self.target.v[0] = p.copy()
            self.has_target = True
        self.target.append(now, now + 0.02 if at is None else at, p)

    def blend(self, now, w, duration):
        """Ramp the reach weight to w (0 = recorded arm, 1 = fully adapted) over `duration` seconds (smoothstep)."""
        self.weight.append(now, now + max(duration, 0.002), [w])

    def orient_hand(self, now, R_world_pinch, duration=0.3, w=1.0):
        """Optional world orientation target of the pinch frame (x = palm normal toward the object)."""
        q = R2q(np.asarray(R_world_pinch, float))
        if self.hand_w.value(now)[0] <= 0 and len(self.hand_q.t) == 1:
            self.hand_q.v[0] = q
        self.hand_q.append(now, now + 0.002, q)
        self.hand_w.append(now, now + max(duration, 0.002), [w])

    def close(self, now, duration=0.25, gap=None):
        self.gap.append(now, now + max(duration, 0.002), [self.pp.gap_closed if gap is None else gap])

    def open(self, now, duration=0.25, gap=None):
        self.gap.append(now, now + max(duration, 0.002), [self.pp.gap_open if gap is None else gap])

    def set_swivel(self, now, deg, duration=0.3):
        self.swivel.append(now, now + max(duration, 0.002), [deg])

    # ---------------------------------------------------------------- evaluation
    def source_frame(self, t):
        return self.retiming(t)

    def _ik(self, P_arm, Q_arm, w, target, hand_q, hand_w, swivel_deg):
        """Adapted (P, Q) for upperarm_l, lowerarm_l, hand_l given the recorded ones."""
        rh = self.rh
        S, E0, W0 = P_arm[0], P_arm[1], P_arm[2]
        Ru0, Rl0, Rh0 = q2R(Q_arm[0]), q2R(Q_arm[1]), q2R(Q_arm[2])
        C0 = W0 + Rh0 @ self.c_ph
        info = {}
        if hand_w > 0:
            # hand orientation target: wrist target from the desired pinch frame, 2-bone IK on the wrist
            Rh_t_full = q2R(hand_q) @ self.R_ph.T
            Rh_t = Rh_t_full if hand_w >= 1 else q2R(slerp(R2q(Rh0), R2q(Rh_t_full), hand_w))
            Wt = target - Rh_t @ self.c_ph
            end0, endt = W0, Wt
        else:
            Rh_t = None
            end0, endt = C0, target
        L1 = np.linalg.norm(E0 - S)
        f0 = end0 - E0
        L2 = np.linalg.norm(f0)
        dvec = endt - S
        dist = np.linalg.norm(dvec)
        dh = dvec / dist
        dmin, dmax = abs(L1 - L2) + 1e-4, L1 + L2 - 1e-4
        dc = min(max(dist, dmin), dmax)
        info['reach_clamped_m'] = float(dist - dc)
        cosa = (L1 * L1 + dc * dc - L2 * L2) / (2 * L1 * dc)
        sina = math.sqrt(max(0.0, 1 - cosa * cosa))
        O = S + dh * L1 * cosa
        pref = (E0 - S) - np.dot(E0 - S, dh) * dh
        if np.linalg.norm(pref) < 1e-6:
            pref = np.array([0, 0, -1.0]) - np.dot([0, 0, -1.0], dh) * dh
        n = pref / np.linalg.norm(pref)
        if swivel_deg:
            n = axis_angle_R(dh, math.radians(swivel_deg)) @ n
        E = O + n * L1 * sina
        T_eff = S + dh * dc
        D_up = rot_between(E0 - S, E - S)
        Ru = D_up @ Ru0
        D_low = rot_between(f0, T_eff - E)
        Rl = D_low @ Rl0
        if Rh_t is not None:
            # forearm twist that best matches the requested hand orientation (pronation/supination)
            axis = (T_eff - E) / np.linalg.norm(T_eff - E)
            Rrel0 = Rl0.T @ Rh0
            Rl_want = Rh_t @ Rrel0.T
            psi = twist_about(Rl_want @ Rl.T, axis)
            Rl = axis_angle_R(axis, psi) @ Rl
            Rh = Rh_t
        else:
            Rh = Rl @ Rl0.T @ Rh0
        # blend each bone rotation, FK the child positions from the fixed shoulder
        qs = []
        for Rr, Ra in ((Ru0, Ru), (Rl0, Rl), (Rh0, Rh)):
            qa = R2q(Ra)
            qr = qnorm(Q_arm[len(qs)])
            if np.dot(qa, qr) < 0:
                qa = -qa
            qs.append(slerp(qr, qa, w) if w < 1 else qa)
        Rub, Rlb, Rhb = q2R(qs[0]), q2R(qs[1]), q2R(qs[2])
        Pl = S + Rub @ (rh['lowerarm_l'] - rh['upperarm_l'])
        Ph = Pl + Rlb @ (rh['hand_l'] - rh['lowerarm_l'])
        Pa = np.stack([S, Pl, Ph])
        Qa = np.stack(qs)
        c = Ph + Rhb @ self.c_ph
        info['pinch_centre'] = c
        info['target_error_m'] = float(np.linalg.norm(c - target)) if w >= 1 else None
        return Pa, Qa, info

    def arm(self, t, names=None):
        """Adapted (P, Q) of the requested bones at robot time t (default: all 21)."""
        names = list(self.rec.names) if names is None else list(names)
        need = list(dict.fromkeys(names + list(ARM_L)))
        s = self.retiming(t)
        P, Q = self.rec.at_bones(s, need)
        w = float(self.weight.value(t)[0])
        info = {'source_frame': s, 'w': w}
        if w > 0 and self.has_target:
            ia = [need.index(n) for n in ARM_L]
            Pa, Qa, inf = self._ik(P[ia], Q[ia], w, self.target.value(t), qnorm(self.hand_q.value(t)),
                                   float(self.hand_w.value(t)[0]), float(self.swivel.value(t)[0]))
            P[ia], Q[ia] = Pa, Qa
            info.update(inf)
        out = [need.index(n) for n in names]
        return P[out], Q[out], info

    def full_pose(self, t):
        """All 21 bones at robot time t; exactly the recording when w = 0 and s(t) is an integer frame."""
        s = self.retiming(t)
        w = float(self.weight.value(t)[0])
        P, Q = self.rec.at(s)
        if w > 0 and self.has_target:
            P2, Q2, _ = self.arm(t, list(ARM_L))
            for k, n in enumerate(ARM_L):
                P[self.rec.idx[n]], Q[self.rec.idx[n]] = P2[k], Q2[k]
        return P, Q, s, w

    def pad_poses(self, t, P_hand=None, Q_hand=None):
        if P_hand is None:
            P, Q, _ = self.arm(t, ['hand_l'])
            P_hand, Q_hand = P[0], Q[0]
        Rh = q2R(Q_hand)
        Rp = Rh @ self.R_ph
        c = P_hand + Rh @ self.c_ph
        x = Rp[:, 0]
        g = float(self.gap.value(t)[0])
        gc = self.pp.gap_closed
        zs = Rp[:, 2] * self.pp.pad_shift_z
        palm = c - x * (gc / 2 + self.pp.palm_half[0]) + zs
        thumb = c - x * (gc / 2) + x * (g + self.pp.thumb_half[0]) + Rp[:, 1] * self.pp.thumb_offset_yz[0] \
            + Rp[:, 2] * self.pp.thumb_offset_yz[1] + zs
        q = R2q(Rp)
        return {'recipient_palm_pad': (palm, q), 'recipient_thumb_pad': (thumb, q.copy()), 'pinch_centre': c,
                'gap': g, 'R_pinch': Rp}

    def mocap(self, t):
        """{mocap body name: (pos, quat_wxyz)} for every recipient body of the harness scene at robot time t."""
        if self._cache_t == t:
            return self._cache
        names = ['lowerarm_l', 'hand_l', 'lowerarm_r', 'hand_r']
        P, Q, info = self.arm(t, names)
        pads = self.pad_poses(t, P[1], Q[1])
        out = {'recipient_lowerarm_l': (P[0], Q[0]),
               'recipient_lowerarm_r': (P[2], Q[2]), 'recipient_hand_r': (P[3], Q[3])}
        if self.hand == 'hook':
            out['recipient_hook_hand'] = (pads['pinch_centre'].copy(), R2q(pads['R_pinch']))
        else:
            out['recipient_palm_pad'] = pads['recipient_palm_pad']
            out['recipient_thumb_pad'] = pads['recipient_thumb_pad']
        self._cache_t, self._cache = t, out
        self.last_info = dict(info, gap=pads['gap'], pinch_centre=pads['pinch_centre'])
        return out

    # ---------------------------------------------------------------- export
    def schedules_json(self):
        return {'retiming': self.retiming.to_json(), 'reach_weight': self.weight.to_json(),
                'reach_target': self.target.to_json() if self.has_target else None,
                'hand_orientation': self.hand_q.to_json(), 'hand_orientation_weight': self.hand_w.to_json(),
                'swivel_deg': self.swivel.to_json(), 'pinch_gap': self.gap.to_json(),
                'hand': self.hand, 'hook_params': asdict(self.hp) if self.hand == 'hook' else None,
                'pinch_params': asdict(self.pp), 'pinch_frame_in_hand': {'R': self.R_ph.tolist(), 'c': self.c_ph.tolist(),
                                                                          **self.pinch_info}}

    def write_motion_json(self, path, t_end, fps=FPS, extra=None):
        """h2r-avatar-motion/1 for the person in ROBOT time (frame k <-> t = k / fps), adapted left-arm bones."""
        raw = self.rec.raw
        out = {k: copy.deepcopy(v) for k, v in raw.items() if k != 'frames'}
        frames = []
        n = int(math.floor(t_end * fps + 1e-9)) + 1
        mapping = []
        for k in range(n):
            t = k / fps
            P, Q, s, w = self.full_pose(t)
            i = int(math.floor(s + 1e-9))
            exact = (abs(s - round(s)) < 1e-9) and w <= 0
            if exact:
                src = raw['frames'][int(round(s))]
                fr = {'frame': k, 'wall_ns': src['wall_ns'], 'valid': True,
                      'rotation_wxyz': copy.deepcopy(src['rotation_wxyz']), 'position': copy.deepcopy(src['position'])}
            else:
                fr = {'frame': k, 'wall_ns': raw['frames'][min(i, self.rec.nframes - 1)]['wall_ns'], 'valid': True,
                      'rotation_wxyz': [[round(float(x), 7) for x in q] for q in Q],
                      'position': [[round(float(x), 7) for x in p] for p in P]}
            fr['source_frame'] = round(float(s), 6)
            fr['robot_time_s'] = round(t, 6)
            fr['left_arm_adaptation_weight'] = round(w, 6)
            frames.append(fr)
            mapping.append([round(t, 6), round(float(s), 6)])
        out['frames'] = frames
        out['robot_time'] = {'fps': fps, 'frame_k_time_s': 'k / fps', 'source_recording': str(self.rec.path),
                             'source_fps': FPS, 'robot_time_to_source_frame': mapping}
        out['adaptation'] = {
            'kind': 'replayed recording; LEFT ARM ONLY (upperarm_l, lowerarm_l, hand_l rotations) adapted by an analytic '
                    '2-bone IK (shoulder fixed, recorded elbow swivel) during the receive, blended in/out; child '
                    'positions recomputed by FK from the fixed shoulder; every other bone is the recording',
            'schedules': self.schedules_json(),
            'hand_model': 'kinematic palm + curled-finger hook (recipient_hook_hand, one mocap body; carries the strap loop '
                          'by normal force)' if self.hand == 'hook' else
                          'kinematic two-pad pinch (recipient_palm_pad fixed on the palm, recipient_thumb_pad moving '
                          'along the palm normal); the same adapted bones drive the physics colliders and this file'}
        if extra:
            out['adaptation'].update(extra)
        Path(path).write_text(json.dumps(out, separators=(',', ':')) + '\n')
        return out


# ----------------------------------------------------------------------------------------- validation
def validate(outdir=HERE / 'recipient_check'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    res = {}
    # (1) unadapted, identity retiming (t = 0 <-> frame 0, rate 1): output == recording exactly
    r0 = Recipient(Retiming(0, rate=1.0))
    tmp = outdir / '_identity.json'
    out = r0.write_motion_json(tmp, t_end=(r0.rec.nframes - 1) / FPS)
    same = all(f['rotation_wxyz'] == g['rotation_wxyz'] and f['position'] == g['position']
               for f, g in zip(out['frames'], r0.rec.raw['frames'])) and len(out['frames']) == r0.rec.nframes
    res['identity_retiming_unadapted_equals_recording'] = bool(same)
    tmp.unlink()
    # (1b) start frame 120, hold 1.5 s, then play: unadapted frames equal recorded frames exactly
    r1 = Recipient(Retiming.from_plan(START_FRAME, [(1.5, 0.0), (10.0, 1.0)]))
    o1 = r1.write_motion_json(outdir / '_hold.json', t_end=4.0)
    ok = True
    for f in o1['frames']:
        src = r1.rec.raw['frames'][int(round(f['source_frame']))]
        ok &= f['rotation_wxyz'] == src['rotation_wxyz'] and f['position'] == src['position']
    res['hold_then_play_unadapted_equals_recorded_frames'] = bool(ok)
    res['hold_then_play_source_frames_first_last'] = [o1['frames'][0]['source_frame'], o1['frames'][-1]['source_frame']]
    (outdir / '_hold.json').unlink()
    # (2) adapted reach: targets at G1-presentable heights in front of the person
    rec = r1.rec
    errs = []
    cases = []
    S245 = rec.P[245, rec.idx['upperarm_l']]
    probe = Recipient(Retiming(245, rate=0.0))
    Pp, Qp, _ = probe.arm(0.0, list(ARM_L))
    L1 = float(np.linalg.norm(Pp[1] - Pp[0]))
    L2 = float(np.linalg.norm(Pp[2] + q2R(Qp[2]) @ probe.c_ph - Pp[1]))
    res['left_shoulder_f245'] = S245.tolist()
    res['reach_radius_shoulder_to_pinch_centre_m'] = L1 + L2
    res['upper_len_m'], res['elbow_to_pinch_centre_m'] = L1, L2
    for tgt in ([1.40, 4.10, 1.00], [1.45, 4.05, 0.95], [1.35, 4.15, 1.10], [1.43, 4.00, 0.90], [1.43, 4.25, 0.85]):
        rr = Recipient(Retiming(236, rate=1.0))
        rr.set_target(0.0, tgt)
        rr.blend(0.0, 1.0, 0.0)
        for t in np.linspace(0.01, 0.8, 9):
            P, Q, info = rr.arm(float(t), list(ARM_L))
            errs.append(info['target_error_m'])
        cases.append({'target': tgt, 'max_err_m': float(max(errs[-9:])), 'reach_clamped_m': info['reach_clamped_m'],
                      'dist_from_shoulder_m': float(np.linalg.norm(np.asarray(tgt) - P[0]))})
    res['adapted_target_cases'] = cases
    reach_ok = [c['max_err_m'] for c in cases if c['reach_clamped_m'] <= 0]
    res['adapted_max_target_error_m_reachable_targets'] = float(max(reach_ok))
    res['adapted_palm_within_1cm'] = bool(max(reach_ok) < 0.01)
    res['note_unreachable'] = 'targets farther than the reach radius from the shoulder are clamped to the arm length (reported)'
    # bone lengths preserved
    rr = Recipient(Retiming(236, rate=1.0))
    rr.set_target(0.0, [1.43, 4.05, 0.95])
    rr.blend(0.0, 1.0, 0.0)
    P, Q, _ = rr.arm(0.3, list(ARM_L))
    P0, _ = rec.at_bones(236 + 9, list(ARM_L))
    res['bone_length_change_m'] = [float(abs(np.linalg.norm(P[1] - P[0]) - np.linalg.norm(P0[1] - P0[0]))),
                                   float(abs(np.linalg.norm(P[2] - P[1]) - np.linalg.norm(P0[2] - P0[1])))]
    # (3) blend in/out continuity (planned schedule): f236..f262 window
    rb = Recipient(Retiming.from_plan(START_FRAME, [(0.0, 1.0)]))
    t_in0 = (236 - START_FRAME) / FPS
    rb.set_target(0.0, [1.43, 4.05, 0.95])
    rb.weight.append(0.0, t_in0, [0.0])
    rb.weight.append(0.0, t_in0 + 0.25, [1.0])
    rb.weight.append(0.0, t_in0 + 0.65, [1.0])
    rb.weight.append(0.0, t_in0 + 0.95, [0.0])
    rb.gap.append(0.0, t_in0 + 0.30, [rb.pp.gap_open])
    rb.gap.append(0.0, t_in0 + 0.45, [rb.pp.gap_closed])
    ts = np.arange(t_in0 - 0.1, t_in0 + 1.1, 0.002)
    cs = np.array([rb.pad_poses(float(t))['pinch_centre'] for t in ts])
    v = np.linalg.norm(np.diff(cs, axis=0), axis=1) / 0.002
    res['blend_max_pinch_speed_mps'] = float(v.max())
    res['blend_max_pinch_accel_mps2'] = float((np.abs(np.diff(v)) / 0.002).max())
    res['pinch_frame'] = rb.schedules_json()['pinch_frame_in_hand']
    (outdir / 'validation.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    # (4) renders: skinned avatar points (LBS) recorded vs adapted
    skin = None
    try:
        sk = np.load(HERE / 'person' / 'avatar' / 'playback_mesh.npz')
        desc = json.loads((HERE / 'person' / 'avatar' / 'playback_mesh.json').read_text())
        body = next(m for m in desc['meshes'] if m['role'] == 'body')
        skin = (sk[body['key'] + '_vertices'], sk[body['key'] + '_weights'])
    except Exception as e:  # noqa
        res['skin_error'] = str(e)

    def lbs(P, Q):
        V, W = skin
        out = np.zeros_like(V)
        for b in range(len(rec.names)):
            wb = W[:, b]
            sel = wb > 1e-4
            if not np.any(sel):
                continue
            out[sel] += wb[sel, None] * ((V[sel] - rec.rest_head[b]) @ q2R(Q[b]).T + P[b])
        return out

    def box_edges(c, R, h):
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
        pts = corners @ R.T + c
        E = []
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(np.abs(corners[i] - corners[j]) > 1e-12) == 1:
                    E.append((pts[i], pts[j]))
        return E

    t_full = t_in0 + 0.5
    views = [('front', (0, 1)), ('side', (1, 2)), ('top', (0, 2))]
    for name, t in (('blend_in', t_in0 + 0.12), ('full_reach_closed', t_full), ('blend_out', t_in0 + 0.82)):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5.5))
        Pa, Qa, s, w = rb.full_pose(float(t))
        Pr, Qr = rec.at(s)
        pads = rb.pad_poses(float(t))
        for ax, (vn, _) in zip(axs, views):
            ij = {'front': (0, 2), 'side': (1, 2), 'top': (0, 1)}[vn]
            if skin is not None:
                Vr, Va = lbs(Pr, Qr), lbs(Pa, Qa)
                sub = slice(None, None, 7)
                ax.scatter(Vr[sub, ij[0]], Vr[sub, ij[1]], s=0.3, c='0.75', label='recorded skin')
                ax.scatter(Va[sub, ij[0]], Va[sub, ij[1]], s=0.3, c='tab:blue', alpha=0.5, label='adapted skin')
            for PP, col in ((Pr, 'k'), (Pa, 'tab:red')):
                ch = [rec.idx[n] for n in ('clavicle_l', 'upperarm_l', 'lowerarm_l', 'hand_l')]
                ax.plot(PP[ch, ij[0]], PP[ch, ij[1]], '-o', c=col, ms=3, lw=2)
            for key, h, col in (('recipient_palm_pad', rb.pp.palm_half, 'tab:green'),
                                ('recipient_thumb_pad', rb.pp.thumb_half, 'tab:orange')):
                pos, q = pads[key]
                for a, b in box_edges(pos, q2R(q), np.asarray(h)):
                    ax.plot([a[ij[0]], b[ij[0]]], [a[ij[1]], b[ij[1]]], c=col, lw=1)
            tg = rb.target.value(float(t))
            ax.plot(tg[ij[0]], tg[ij[1]], 'x', c='m', ms=12, mew=2)
            c = pads['pinch_centre']
            x = pads['R_pinch'][:, 0]
            ax.arrow(c[ij[0]], c[ij[1]], 0.08 * x[ij[0]], 0.08 * x[ij[1]], color='m', width=0.002)
            ax.set_aspect('equal')
            ax.set_title(f'{vn} ({"xyz"[ij[0]]}-{"xyz"[ij[1]]})')
            ax.grid(alpha=0.3)
        fig.suptitle(f'{name}: t={t:.2f}s src frame {s:.1f} weight {w:.2f} gap {pads["gap"]*1000:.0f} mm  '
                     f'(black = recorded arm, red = adapted, magenta x = target, arrow = palm normal)')
        fig.tight_layout()
        fig.savefig(outdir / f'adapted_arm_{name}.png', dpi=110)
        plt.close(fig)
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    a = ap.parse_args()
    if a.validate:
        validate()
