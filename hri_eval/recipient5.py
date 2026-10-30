#!/usr/bin/env python3
"""v5 recipient (the person): replayed recording with
  (a) anatomically limited arm adaptation: analytic 2-bone IK (shoulder = recorded upperarm head, recorded bone lengths)
      for a HAND FRAME goal (position + orientation).  The elbow swivel is chosen at PLANNING time by a search that
      keeps shoulder elevation, elbow flexion (0-145), forearm pronation/supination, wrist flexion/extension (70/70) and
      radial/ulnar deviation (20/30) inside the limits; every evaluated pose is additionally clamped to the limits
      (forearm twist first, then wrist deviation/flexion; the hand orientation gives way, reported as
      orientation_error_deg).  All quaternions are kept in one hemisphere (no slerp flips).
  (b) smooth time-parameterised motion: the hand goal is a chain of QUINTIC segments (position, rotation vector,
      swivel) with continuous velocity/acceleration at re-plans; durations come from the speed limits (hand <= 0.8 m/s,
      <= 240 deg/s, with a 20 % margin for the body's own motion); engage/release blends are min-jerk; everything is
      evaluated at the physics time t (no 50 Hz staircase).
  (c) idle retiming: instead of freezing, his own recorded quiet frames are replayed back and forth (cosine ping-pong
      inside a window of his clip, peak rate <= 1); explicit holds are limited to 0.5 s.
  (d) causality: every goal is decided at `now` from the current state (the recorded pose at s(now) for the planning
      search); knots are append-only (t >= now) and logged with the time they were decided.

Mocap bodies (skin-matched colliders, build_variant_scene.py --hand skin): recipient_{upperarm,lowerarm,hand}_{l,r};
each is posed EXACTLY at its bone (p_B, R_B), so the physics colliders are the visible skin.

API (same names as v4 where possible):
    rcp = Recipient5(start_frame=120)                     # t = 0 <-> f120, playing at rate 1
    rcp.hold(now, 0.4)                                    # pause <= 0.5 s, then continue playing (rate 1)
    rcp.play(now, rate=1.0)                               # continue forward from the current frame (rate ramps 0.3 s)
    rcp.idle(now, window=None, peak_rate=0.45)            # ping-pong in a quiet window (default: quietest >=1.2 s
                                                          #   window of f160-f222); if s(now) < window start it
                                                          #   plays forward into the window first
    rcp.reach(now, pos, R=None, duration=None, swivel_deg=None, side='l')   # hand-frame goal (world); engages the arm
    rcp.move(now, pos, R=None, duration=None)             # alias of reach once engaged
    rcp.release(now, duration=None, side='l')             # blend the arm back to the recording (min-jerk)
    rcp.hand_frame(t, side='l') -> (pos, R)               # current hand frame (world) of the evaluated pose
    rcp.full_pose(t) -> (P21, Q21, s, w)                  # bones at robot time t
    rcp.mocap(t) -> {body: (pos, quat)}                   # called by the runner every physics step
    rcp.angles(t, side='l') -> dict                       # anatomical angles of the evaluated pose
    rcp.write_motion_json(path, t_end)                    # h2r-avatar-motion/1 at 30 fps of robot time
    validate()                                            # CLI --validate: writes recipient5_check.json
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

import recipient as RCP
import skin as SKN
from recipient import q2R, qnorm, rot_between, axis_angle_R

HERE = Path(__file__).resolve().parent
FPS = 30.0
START_FRAME = 120


def R2q(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def exp_R(v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(v).as_matrix()


def minjerk(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (10 - 15 * u + 6 * u * u)


@dataclass
class Limits:
    elbow_flex: tuple = (0.0, 145.0)
    wrist_flex: tuple = (-70.0, 70.0)          # - extension / + flexion
    wrist_dev: tuple = (-30.0, 20.0)           # - ulnar / + radial
    pronation: tuple = (-80.0, 80.0)           # forearm twist vs the upper arm, relative to the avatar rest relation
    wrist_twist: tuple = (-10.0, 10.0)         # hand vs forearm about the forearm axis (anatomically ~0)
    shoulder_elevation: tuple = (0.0, 120.0)   # upper arm angle from straight down
    v_hand: float = 0.8                        # m/s
    w_hand: float = 240.0                      # deg/s
    margin: float = 0.8                        # planned peak = margin * limit (headroom for the body's own motion)
    max_hold_s: float = 0.5


# ------------------------------------------------------------------------------------------------ time map
class TimeMap:
    """robot time -> source frame.  Segments (append-only):
       ('play', t0, s0, r0, r1, T)  rate ramps linearly r0 -> r1 over T then stays r1 (rates in [-1, 1])
       ('pingpong', t0, a, b, theta0, omega)  s = c - A cos(theta0 + omega (t - t0)), c=(a+b)/2, A=(b-a)/2
    """

    def __init__(self, start_frame=START_FRAME, rate=1.0, last_frame=303):
        self.seg = [('play', 0.0, float(start_frame), float(rate), float(rate), 0.0)]
        self.last = float(last_frame)
        self.log = []

    def _eval(self, seg, t):
        kind = seg[0]
        if kind == 'play':
            _, t0, s0, r0, r1, T = seg
            dt = t - t0
            if T > 0 and dt < T:
                s = s0 + FPS * (r0 * dt + 0.5 * (r1 - r0) / T * dt * dt)
                r = r0 + (r1 - r0) * dt / T
            else:
                sT = s0 + FPS * (0.5 * (r0 + r1) * T)
                s = sT + FPS * r1 * (dt - T)
                r = r1
            return s, r
        _, t0, a, b, th0, om = seg
        c, A = 0.5 * (a + b), 0.5 * (b - a)
        th = th0 + om * (t - t0)
        return c - A * math.cos(th), A * om * math.sin(th) / FPS

    def at(self, t):
        i = len(self.seg) - 1
        while i > 0 and self.seg[i][1] > t:
            i -= 1
        s, r = self._eval(self.seg[i], t)
        return min(max(s, 0.0), self.last), r

    def __call__(self, t):
        return self.at(t)[0]

    def _append(self, now, seg):
        if now < self.seg[-1][1] - 1e-12:
            raise ValueError('time map is append-only')
        if abs(now - self.seg[-1][1]) < 1e-12 and len(self.seg) > 1:
            self.seg[-1] = seg
        else:
            self.seg.append(seg)

    def play(self, now, rate=1.0, ramp=0.3):
        s, r = self.at(now)
        assert -1.0 <= rate <= 1.0
        self._append(now, ('play', float(now), s, r, float(rate), float(ramp)))
        self.log.append({'t': now, 'op': 'play', 'rate': rate, 's': s})

    def hold(self, now, duration, max_hold=0.5):
        if duration > max_hold + 1e-9:
            raise ValueError(f'holds are limited to {max_hold} s (use idle())')
        s, r = self.at(now)
        self._append(now, ('play', float(now), s, r, 0.0, 0.1))
        self.log.append({'t': now, 'op': 'hold', 'duration': duration, 's': s})
        self.play(now + duration, 1.0)

    def pingpong(self, now, a, b, peak_rate=0.45):
        s, r = self.at(now)
        A = 0.5 * (b - a)
        om = peak_rate * FPS / A
        x = (0.5 * (a + b) - s) / A
        x = min(1.0, max(-1.0, x))
        th0 = math.acos(x)                    # in [0, pi]: moving forward (ds/dt >= 0)
        if r < 0:
            th0 = 2 * math.pi - th0           # keep moving backward if we were
        self._append(now, ('pingpong', float(now), float(a), float(b), th0, om))
        self.log.append({'t': now, 'op': 'pingpong', 'window': [a, b], 'peak_rate': peak_rate, 's': s})

    def to_json(self):
        return {'fps': FPS, 'segments': [list(x) for x in self.seg], 'ops': self.log,
                'rule': "play: s = s0 + fps*(r0*dt + (r1-r0)/(2T) dt^2) during the ramp, then rate r1; "
                        "pingpong: s = (a+b)/2 - (b-a)/2 cos(theta0 + omega dt)"}


def quiet_windows(rec: RCP.Recording, f0=160, f1=222, min_len=36, sk: SKN.Skin | None = None):
    """Rank windows [a, a+L] of the recording by the max skin-vertex speed inside (quietest first)."""
    sk = sk or SKN.Skin()
    sub = np.arange(0, len(sk.V0), 7)
    mask = np.zeros(len(sk.V0), bool)
    mask[sub] = True
    Vs = [sk.pose(rec.P[f], rec.Q[f], vmask=mask)[sub] for f in range(f0, f1 + 1)]
    sp = np.array([np.linalg.norm(Vs[i + 1] - Vs[i], axis=1).max() * FPS for i in range(len(Vs) - 1)])
    out = []
    for a in range(f0, f1 - min_len + 1):
        seg = sp[a - f0:a - f0 + min_len]
        out.append((float(np.quantile(seg, 0.9)), float(seg.mean()), a, a + min_len))
    out.sort()                         # robust: 90th percentile of the per-frame max skin speed (tracking spikes ignored)
    return out, sp


# ------------------------------------------------------------------------------------------------ quintic goal track
class PoseTrack:
    """Hand-frame goal (world): chain of quintic segments in position, rotation vector (relative to the segment start)
    and swivel; each new segment starts from the current (p, v, a) so re-plans are C2."""

    def __init__(self):
        self.segs = []        # (t0, T, p0, v0, a0, p1, R0, w0, rv1, sw0, swv0, sw1, decided_at, note)

    @staticmethod
    def _q(x0, v0, a0, x1, T, tau):
        """quintic from (x0, v0, a0) to (x1, 0, 0) over T, evaluated at tau (value, vel, acc)."""
        x0, v0, a0, x1 = (np.asarray(z, float) for z in (x0, v0, a0, x1))
        T2, T3, T4, T5 = T * T, T ** 3, T ** 4, T ** 5
        c0, c1, c2 = x0, v0, a0 / 2
        d = x1 - x0 - v0 * T - a0 * T2 / 2
        # solve the 3x3 system for c3, c4, c5 with end vel/acc = 0
        M = np.array([[T3, T4, T5], [3 * T2, 4 * T3, 5 * T4], [6 * T, 12 * T2, 20 * T3]])
        rhs = np.stack([d, -v0 - a0 * T, -a0], 0)
        c345 = np.linalg.solve(M, rhs.reshape(3, -1)).reshape((3,) + np.shape(x0))
        c3, c4, c5 = c345
        tau = min(max(tau, 0.0), T)
        x = c0 + c1 * tau + c2 * tau ** 2 + c3 * tau ** 3 + c4 * tau ** 4 + c5 * tau ** 5
        v = c1 + 2 * c2 * tau + 3 * c3 * tau ** 2 + 4 * c4 * tau ** 3 + 5 * c5 * tau ** 4
        a = 2 * c2 + 6 * c3 * tau + 12 * c4 * tau ** 2 + 20 * c5 * tau ** 3
        return x, v, a

    def state(self, t):
        """(p, v, a, R, w_vec(world), swivel, swivel_rate)"""
        if not self.segs:
            return None
        i = len(self.segs) - 1
        while i > 0 and self.segs[i][0] > t:
            i -= 1
        t0, T, p0, v0, a0, p1, R0, w0, rv1, sw0, swv0, sw1, _, _ = self.segs[i]
        tau = t - t0
        p, v, a = self._q(p0, v0, a0, p1, T, tau)
        rv, rw, _ = self._q(np.zeros(3), w0, np.zeros(3), rv1, T, tau)
        sw, swv, _ = self._q(sw0, swv0, 0.0, sw1, T, tau)
        R = R0 @ exp_R(rv)
        return p, v, a, R, R0 @ rw, float(sw), float(swv)

    def append(self, now, T, p1, R1, sw1, note=''):
        st = self.state(now)
        if st is None:
            raise RuntimeError('first segment needs a start pose')
        p, v, a, R, w, sw, swv = st
        self._push(now, T, p, v, a, p1, R, R.T @ w, R1, sw, swv, sw1, note)

    def _push(self, now, T, p0, v0, a0, p1, R0, w0_local, R1, sw0, swv0, sw1, note):
        if self.segs and now < self.segs[-1][0] - 1e-12:
            raise ValueError('goal track is append-only')
        rv1 = rotvec(R0.T @ R1)
        self.segs.append((float(now), float(T), np.array(p0, float), np.array(v0, float), np.array(a0, float),
                          np.array(p1, float), np.array(R0, float), np.array(w0_local, float), rv1, float(sw0),
                          float(swv0), float(sw1), float(now), note))

    def start(self, now, p, R, sw):
        self._push(now, 1e-3, p, np.zeros(3), np.zeros(3), p, R, np.zeros(3), R, sw, 0.0, sw, 'start')

    def to_json(self):
        return [{'t0': s[0], 'T': s[1], 'p0': s[2].tolist(), 'v0': s[3].tolist(), 'a0': s[4].tolist(),
                 'p1': s[5].tolist(), 'R0': s[6].tolist(), 'w0_local': s[7].tolist(), 'rotvec1': s[8].tolist(),
                 'swivel0_deg': s[9], 'swivel_rate0': s[10], 'swivel1_deg': s[11], 'decided_at': s[12], 'note': s[13]}
                for s in self.segs]


class WeightTrack:
    """min-jerk weight 0..1 (append-only)."""

    def __init__(self):
        self.k = [(0.0, 0.0, 0.0, 0.0)]       # (t0, w0, T, w1)

    def __call__(self, t):
        i = len(self.k) - 1
        while i > 0 and self.k[i][0] > t:
            i -= 1
        t0, w0, T, w1 = self.k[i]
        if T <= 0:
            return w1
        return w0 + (w1 - w0) * minjerk((t - t0) / T)

    def set(self, now, w1, T):
        w0 = self(now)
        if now < self.k[-1][0] - 1e-12:
            raise ValueError('append-only')
        self.k.append((float(now), w0, float(T), float(w1)))

    def to_json(self):
        return [{'t0': a, 'w0': b, 'T': c, 'w1': d} for a, b, c, d in self.k]


# ------------------------------------------------------------------------------------------------ the recipient
class Recipient5:
    hand = 'skin'
    MOCAP_BODIES = ('recipient_upperarm_l', 'recipient_lowerarm_l', 'recipient_hand_l',
                    'recipient_upperarm_r', 'recipient_lowerarm_r', 'recipient_hand_r')

    def __init__(self, start_frame=START_FRAME, motion_path=RCP.MOTION, limits: Limits | None = None,
                 mocap_bodies=None, idle_window=None, skin: SKN.Skin | None = None):
        self.rec = RCP.Recording(motion_path)
        self.lim = limits or Limits()
        self.tm = TimeMap(start_frame, 1.0, self.rec.nframes - 1)
        self.retiming = self.tm
        self.sk = skin or SKN.Skin(str(motion_path))
        self.anat = {s: SKN.HandAnatomy(self.sk, s) for s in ('l', 'r')}
        self.mocap_bodies = tuple(mocap_bodies or self.MOCAP_BODIES)
        self.hf = {s: self._hand_frame_local(s) for s in ('l', 'r')}
        self.goal = {s: PoseTrack() for s in ('l', 'r')}
        self.w = {s: WeightTrack() for s in ('l', 'r')}
        self.idle_window = idle_window
        self.decisions = []
        self._cache_t = None
        self.last_info = {}
        self.rh = {n: self.rec.rest_head[self.rec.idx[n]] for n in self.rec.names}

    # ---------------------------------------------------------------- hand frame (palm centre, from the skin)
    def _hand_frame_local(self, side):
        an = self.anat[side]
        z = an.hand_axis_rest
        x = an.palm_n_rest - (an.palm_n_rest @ z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], 1)              # rest coords == hand-bone local coords (rest rotations are I)
        hv = self.sk.V0[self.sk.bone_vertices(f'hand_{side}', 0.5)]
        wr = an.wrist_rest
        L = (hv - wr) @ z
        palm = hv[(L > 0.03) & (L < 0.09)]
        c = palm.mean(0)
        top = palm[(palm - c) @ x > np.quantile((palm - c) @ x, 0.9)]
        c = c + x * float(((top - c) @ x).mean())
        return R, c - wr                       # in hand-bone local coordinates (origin = bone head)

    # ---------------------------------------------------------------- retiming
    def source_frame(self, t):
        return self.tm(t)

    def play(self, now, rate=1.0):
        self._cache_t = None
        self.tm.play(now, rate)

    def hold(self, now, duration):
        self._cache_t = None
        self.tm.hold(now, duration, self.lim.max_hold_s)

    def idle(self, now, window=None, peak_rate=0.45):
        self._cache_t = None
        a, b = window or self.idle_window or self.default_idle_window()
        s = self.tm(now)
        if s < a - 1e-6:
            # play forward into the window first, then ping-pong from its start
            t_in = now + (a - s) / FPS
            self.tm.play(now, 1.0, 0.0)
            self.tm.pingpong(t_in, a, b, peak_rate)
        else:
            if s > b + 1e-6:
                raise ValueError(f'idle window {a}-{b} is behind the current frame {s:.1f}')
            self.tm.pingpong(now, a, b, peak_rate)
        self.decisions.append({'t': now, 'op': 'idle', 'window': [a, b]})

    def default_idle_window(self):
        f = HERE / 'idle_window.json'
        if f.exists():
            return tuple(json.loads(f.read_text())['window'])
        wins, _ = quiet_windows(self.rec, sk=self.sk)
        w = (wins[0][2], wins[0][3])
        f.write_text(json.dumps({'window': w, 'p90_of_max_skin_speed_mps': wins[0][0], 'mean_of_max': wins[0][1],
                                 'search': 'f160-f222, length 36 frames, quietest by the 90th percentile of the '
                                           'per-frame max skin-vertex speed'}) + '\n')
        return w

    # ---------------------------------------------------------------- arm IK with limits
    def _rec_arm(self, P, Q, side):
        n = [f'upperarm_{side}', f'lowerarm_{side}', f'hand_{side}']
        ii = [self.rec.idx[x] for x in n]
        return ii, P[ii].copy(), Q[ii].copy()

    def _swivel_ref(self, P, Q, side, dh):
        # reference = down + 0.3 * outward (lateral) direction of the recorded chest, made orthogonal to the axis
        cl = self.rec.idx[f'clavicle_{side}']
        sp = self.rec.idx['spine_03']
        out = P[cl] - P[sp]
        out[2] = 0
        out /= max(1e-9, np.linalg.norm(out))
        ref = np.array([0, 0, -1.0]) + 0.3 * out
        ref = ref - (ref @ dh) * dh
        return ref / np.linalg.norm(ref)

    def solve_arm(self, P, Q, side, Wpos, Rh_des, swivel_deg, clamp=True):
        """Adapted (Ru, Rl, Rh, positions) for the hand BONE rotation Rh_des and wrist position Wpos."""
        ii, Pa, Qa = self._rec_arm(P, Q, side)
        S, E0, W0 = Pa
        Ru0, Rl0 = q2R(Qa[0]), q2R(Qa[1])
        L1, L2 = np.linalg.norm(E0 - S), np.linalg.norm(W0 - E0)
        lim = self.lim
        dvec = Wpos - S
        dist = np.linalg.norm(dvec)
        dh = dvec / dist
        # elbow flexion limits -> distance range (flex = pi - interior angle)
        def d_of(flex):
            th = math.pi - math.radians(flex)
            return math.sqrt(L1 * L1 + L2 * L2 - 2 * L1 * L2 * math.cos(th))
        dmax, dmin = d_of(max(lim.elbow_flex[0], 0.5)), d_of(lim.elbow_flex[1])
        dc = min(max(dist, dmin), dmax)
        info = {'reach_clamped_m': float(dist - dc)}
        cosa = (L1 * L1 + dc * dc - L2 * L2) / (2 * L1 * dc)
        sina = math.sqrt(max(0.0, 1 - cosa * cosa))
        O = S + dh * L1 * cosa
        ref = self._swivel_ref(P, Q, side, dh)
        nvec = axis_angle_R(dh, math.radians(swivel_deg)) @ ref
        E = O + nvec * L1 * sina
        Wc = S + dh * dc
        Ru = rot_between(E0 - S, E - S) @ Ru0
        Rl = rot_between(W0 - E0, Wc - E) @ Rl0
        an = self.anat[side]
        f_w = (Wc - E) / np.linalg.norm(Wc - E)
        # forearm twist: remove the wrist twist, then clamp pronation
        def tw(Rl_, Rh_):
            from scipy.spatial.transform import Rotation
            x, y, z, w = Rotation.from_matrix(Rl_.T @ Rh_).as_quat()
            a = math.degrees(2 * math.atan2(np.dot([x, y, z], an.fore_rest), w))
            return (a + 180) % 360 - 180
        psi = tw(Rl, Rh_des)
        Rl = axis_angle_R(f_w, math.radians(psi)) @ Rl
        ang = an.angles_R(Ru, Rl, Rh_des)
        if clamp:
            pr = ang['pronation_deg']
            pc = min(max(pr, lim.pronation[0]), lim.pronation[1])
            if pc != pr:
                sgn = 1.0 if side == 'l' else -1.0
                Rl = axis_angle_R(f_w, math.radians(sgn * (pc - pr))) @ Rl
        Rh = Rh_des
        if clamp:
            # wrist twist within limits (hand rotates about the forearm axis if needed)
            wt = tw(Rl, Rh)
            wc = min(max(wt, lim.wrist_twist[0]), lim.wrist_twist[1])
            if wc != wt:
                Rh = axis_angle_R(f_w, math.radians(wc - wt)) @ Rh
            # wrist deviation / flexion: direction of the forearm axis in the hand frame
            fh = Rh.T @ f_w
            h, n, r = an.hand_axis_rest, an.palm_n_rest, an.radial_rest
            dev = math.degrees(math.atan2(-(fh @ r), fh @ h))
            flex = math.degrees(math.atan2(-(fh @ n), fh @ h))
            mg = 0.2                     # deg inside the limit (the exported json is rounded to 1e-7)
            dc_ = min(max(dev, lim.wrist_dev[0] + mg), lim.wrist_dev[1] - mg)
            fc_ = min(max(flex, lim.wrist_flex[0] + mg), lim.wrist_flex[1] - mg)
            if dc_ != dev or fc_ != flex:
                # keep the component along the third axis (n x h ... ) and rebuild the clamped direction
                fhc = h - math.tan(math.radians(dc_)) * r - math.tan(math.radians(fc_)) * n
                fhc = fhc / np.linalg.norm(fhc)
                D = rot_between(fh, fhc)
                Rh = Rh @ D.T
        ang = an.angles_R(Ru, Rl, Rh)
        info['orientation_error_deg'] = float(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rh.T @ Rh_des) - 1) / 2)))))
        Pl = S + Ru @ (self.rh[f'lowerarm_{side}'] - self.rh[f'upperarm_{side}'])
        Ph = Pl + Rl @ (self.rh[f'hand_{side}'] - self.rh[f'lowerarm_{side}'])
        info['angles'] = ang
        info['wrist_error_m'] = float(np.linalg.norm(Ph - Wpos))
        return ii, (S, Pl, Ph), (Ru, Rl, Rh), info

    def _limit_cost(self, ang):
        L = self.lim
        c = 0.0
        for k, (lo, hi) in (('elbow_flex_deg', L.elbow_flex), ('wrist_flex_deg', L.wrist_flex),
                            ('wrist_dev_deg', L.wrist_dev), ('pronation_deg', L.pronation),
                            ('wrist_twist_deg', L.wrist_twist), ('shoulder_elevation_deg', L.shoulder_elevation)):
            v = ang[k]
            c += max(0.0, lo - v) ** 2 + max(0.0, v - hi) ** 2
        return c

    def plan_swivel(self, now, side, pos, R):
        """Search the elbow swivel (deg, 0 = elbow toward down/outward reference) for a hand-frame goal using the
        CURRENT recorded pose s(now) (causal).  Returns (swivel_deg, info)."""
        P, Q = self.rec.at(self.tm(now))
        Rhf, chf = self.hf[side]
        Rh_des = R @ Rhf.T
        Wpos = pos - Rh_des @ chf
        best = None
        for sw in np.arange(-90.0, 90.1, 5.0):
            _, _, _, inf = self.solve_arm(P, Q, side, Wpos, Rh_des, sw, clamp=False)
            cost = self._limit_cost(inf['angles']) + 0.02 * abs(sw) + 50 * max(0.0, inf['reach_clamped_m']) * 100
            if best is None or cost < best[0]:
                best = (cost, sw, inf)
        return best[1], {'cost': best[0], 'angles_unclamped': best[2]['angles'], 'reach_clamped_m': best[2]['reach_clamped_m']}

    # ---------------------------------------------------------------- goals
    def _duration(self, side, now, p1, R1, v0=None, extra=0.0):
        st = self.goal[side].state(now)
        p0, R0 = (st[0], st[3]) if st is not None else self.hand_frame(now, side, recorded=True)
        D = float(np.linalg.norm(np.asarray(p1) - p0)) + extra
        th = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ R1) - 1) / 2))))
        vm, wm = self.lim.v_hand * self.lim.margin, self.lim.w_hand * self.lim.margin
        return max(0.25, 1.875 * D / vm, 1.875 * th / wm)

    def reach(self, now, pos, R=None, duration=None, swivel_deg=None, side='l', note=''):
        """Hand-frame goal (world position, world rotation; R=None keeps the current hand orientation).  If the arm
        is not engaged it is blended in (min-jerk) while the goal is held; the duration respects the speed limits."""
        pos = np.asarray(pos, float)
        self._cache_t = None
        w_now = self.w[side](now)
        p_cur, R_cur = self.hand_frame(now, side)
        if R is None:
            R = R_cur
        sw, sinfo = (swivel_deg, {}) if swivel_deg is not None else self.plan_swivel(now, side, pos, R)
        if w_now <= 1e-9:
            # engage: goal = target (constant); weight 0 -> 1.  The hand moves from the recording to the goal: the
            # distance to cover is measured against the recorded hand over the next 1.5 s (causal: recording frames
            # beyond s(now) are not used for the decision -> use the current recorded hand only)
            D = float(np.linalg.norm(pos - p_cur))
            th = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R_cur.T @ R) - 1) / 2))))
            vm, wm = self.lim.v_hand * self.lim.margin, self.lim.w_hand * self.lim.margin
            T = max(duration or 0.0, 0.3, 1.875 * D / vm, 1.875 * th / wm)
            self.goal[side].start(now, pos, R, sw)
            self.w[side].set(now, 1.0, T)
            kind = 'engage'
        else:
            T = max(duration or 0.0, self._duration(side, now, pos, R))
            self.goal[side].append(now, T, pos, R, sw, note)
            kind = 'move'
        self.decisions.append({'t': now, 'op': kind, 'side': side, 'pos': pos.tolist(), 'R': np.asarray(R).tolist(),
                               'swivel_deg': float(sw), 'T': float(T), 'note': note, 'info_used': 'recorded pose at s(now) only',
                               'plan': {k: v for k, v in sinfo.items()}})
        self._cache_t = None
        return T

    move = reach

    def release(self, now, duration=None, side='l'):
        p_cur, R_cur = self.hand_frame(now, side)
        p_rec, R_rec = self.hand_frame(now, side, recorded=True)
        D = float(np.linalg.norm(p_cur - p_rec))
        th = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R_cur.T @ R_rec) - 1) / 2))))
        vm, wm = self.lim.v_hand * self.lim.margin, self.lim.w_hand * self.lim.margin
        T = max(duration or 0.0, 0.5, 1.875 * D / vm, 1.875 * th / wm)
        self._cache_t = None
        self.w[side].set(now, 0.0, T)
        self.decisions.append({'t': now, 'op': 'release', 'side': side, 'T': T})
        return T

    # ---------------------------------------------------------------- evaluation
    def hand_frame(self, t, side='l', recorded=False):
        if recorded:
            P, Q = self.rec.at(self.tm(t))
        else:
            P, Q, _, _ = self.full_pose(t)
        ih = self.rec.idx[f'hand_{side}']
        Rh = q2R(Q[ih])
        Rhf, chf = self.hf[side]
        return P[ih] + Rh @ chf, Rh @ Rhf

    def full_pose(self, t):
        if self._cache_t == t:
            return self._cache
        s = self.tm(t)
        P, Q = self.rec.at(s)
        info = {'source_frame': s}
        wmax = 0.0
        for side in ('l', 'r'):
            w = self.w[side](t)
            info[f'w_{side}'] = w
            wmax = max(wmax, w)
            if w <= 1e-12 or not self.goal[side].segs:
                continue
            ih = self.rec.idx[f'hand_{side}']
            Rhf, chf = self.hf[side]
            Rh0 = q2R(Q[ih])
            p_rec, R_rec = P[ih] + Rh0 @ chf, Rh0 @ Rhf
            pg, _, _, Rg, _, swg, _ = self.goal[side].state(t)
            p = (1 - w) * p_rec + w * pg
            Rrel = R_rec.T @ Rg
            R = R_rec @ exp_R(w * rotvec(Rrel))
            # recorded swivel about the current axis
            Rh_des = R @ Rhf.T
            Wpos = p - Rh_des @ chf
            ii, Pa, Qa = self._rec_arm(P, Q, side)
            dh = (Wpos - Pa[0]) / np.linalg.norm(Wpos - Pa[0])
            ref = self._swivel_ref(P, Q, side, dh)
            e = Pa[1] - Pa[0]
            e = e - (e @ dh) * dh
            sw_rec = math.degrees(math.atan2(np.cross(ref, e) @ dh, ref @ e)) if np.linalg.norm(e) > 1e-9 else 0.0
            dsw = (swg - sw_rec + 180) % 360 - 180
            sw = sw_rec + w * dsw
            ii, pos3, rot3, inf = self.solve_arm(P, Q, side, Wpos, Rh_des, sw, clamp=True)
            for k, (i_, p_, R_) in enumerate(zip(ii, pos3, rot3)):
                q = R2q(R_)
                if np.dot(q, Q[i_]) < 0:          # hemisphere consistency with the recording
                    q = -q
                P[i_], Q[i_] = p_, q
            info[f'arm_{side}'] = inf
        self._cache_t, self._cache = t, (P, Q, s, wmax)
        self.last_info = info
        return P, Q, s, wmax

    def angles(self, t, side='l'):
        P, Q, _, _ = self.full_pose(t)
        return self.anat[side].angles(Q)

    def mocap(self, t):
        P, Q, s, w = self.full_pose(t)
        out = {}
        for b in self.mocap_bodies:
            bone = b.replace('recipient_', '')
            i = self.rec.idx[bone]
            out[b] = (P[i].copy(), Q[i].copy())
        return out

    # ---------------------------------------------------------------- export
    def schedules_json(self):
        return {'time_map': self.tm.to_json(), 'goal_l': self.goal['l'].to_json(), 'goal_r': self.goal['r'].to_json(),
                'weight_l': self.w['l'].to_json(), 'weight_r': self.w['r'].to_json(), 'limits': asdict(self.lim),
                'hand_frame_local': {s: {'R': self.hf[s][0].tolist(), 'c': self.hf[s][1].tolist()} for s in ('l', 'r')},
                'decisions': self.decisions, 'hand': 'skin'}

    def write_motion_json(self, path, t_end, fps=FPS, extra=None):
        raw = self.rec.raw
        out = {k: copy.deepcopy(v) for k, v in raw.items() if k != 'frames'}
        frames, mapping = [], []
        n = int(math.floor(t_end * fps + 1e-9)) + 1
        for k in range(n):
            t = k / fps
            P, Q, s, w = self.full_pose(t)
            fr = {'frame': k, 'wall_ns': raw['frames'][min(int(s), self.rec.nframes - 1)]['wall_ns'], 'valid': True,
                  'rotation_wxyz': [[round(float(x), 7) for x in q] for q in Q],
                  'position': [[round(float(x), 7) for x in p] for p in P],
                  'source_frame': round(float(s), 6), 'robot_time_s': round(t, 6),
                  'left_arm_adaptation_weight': round(float(self.w['l'](t)), 6),
                  'right_arm_adaptation_weight': round(float(self.w['r'](t)), 6)}
            frames.append(fr)
            mapping.append([round(t, 6), round(float(s), 6)])
        out['frames'] = frames
        out['robot_time'] = {'fps': fps, 'frame_k_time_s': 'k / fps', 'source_recording': str(self.rec.path),
                             'source_fps': FPS, 'robot_time_to_source_frame': mapping}
        out['adaptation'] = {'kind': 'v5: replayed recording with idle ping-pong retiming (holds <= 0.5 s) and '
                                     'anatomically limited 2-bone IK of the arms toward quintic hand-frame goals decided '
                                     'online from the current state; the colliders are skin-matched convex pieces in the '
                                     'bone frames (recipient_{upperarm,lowerarm,hand}_{l,r})',
                             'schedules': self.schedules_json(), 'hand_model': 'skin'}
        if extra:
            out['adaptation'].update(extra)
        Path(path).write_text(json.dumps(out, separators=(',', ':')) + '\n')
        return out


# ------------------------------------------------------------------------------------------------ validation
def validate(out=HERE / 'recipient5_check.json'):
    res = {}
    r = Recipient5()
    # identity: no adaptation, play from f120 -> bones == recording at integer frames
    P, Q, s, w = r.full_pose(1.0)
    Pr, Qr = r.rec.at(150)
    res['unadapted_equals_recording_f150'] = bool(np.allclose(P, Pr) and np.allclose(Q, Qr))
    # scenario: play 0-2 s, idle from 2 s, engage a reach at 6 s, move after it, release at 9 s (still idling)
    r = Recipient5()
    r.idle(2.0)
    S = r.full_pose(6.0)[0][r.rec.idx['upperarm_l']]
    p_cur, R_cur = r.hand_frame(6.0)
    fwd = np.array([1.0, 0, 0])
    tgt = S + np.array([0.0, 0.0, -0.40]) + 0.30 * (p_cur - S - ((p_cur - S) @ [0, 0, 1]) * np.array([0, 0, 1])) / max(
        1e-6, np.linalg.norm((p_cur - S)[:2]))
    T1 = r.reach(6.0, tgt, None)
    T2 = r.reach(6.0 + T1 + 0.5, tgt + np.array([0.05, -0.05, 0.08]), R_cur @ axis_angle_R([1, 0, 0], 0.4))
    r.release(9.0)                       # release while idling (a release during his recorded reach inherits its speed)
    ts = np.arange(0, 12.0, 0.002)
    hp, hq, ang, src = [], [], [], []
    for t in ts:
        p, R = r.hand_frame(float(t))
        hp.append(p)
        hq.append(R2q(R))
        ang.append(r.angles(float(t)))
        src.append(r.tm(float(t)))
    hp = np.array(hp)
    wl = np.array([r.w['l'](float(t)) for t in ts])
    ad = (wl[1:] > 0) | (wl[:-1] > 0)
    v = np.linalg.norm(np.diff(hp, axis=0), axis=1) / 0.002
    hq = np.array(hq)
    dq = np.array([SKN.quat_angle_deg(hq[i], hq[i + 1]) for i in range(len(hq) - 1)]) / 0.002
    res['scenario'] = {'engage_T': T1, 'move_T': T2,
                       'max_hand_frame_speed_while_adapted_mps': float(v[ad].max()),
                       't_of_max_speed': float(ts[1:][ad][int(np.argmax(v[ad]))]),
                       'max_hand_ang_speed_while_adapted_dps': float(dq[ad].max()),
                       't_of_max_ang': float(ts[1:][ad][int(np.argmax(dq[ad]))]),
                       'max_hand_speed_all_incl_recorded_mps': float(v.max()),
                       'angles_range': {k: [float(min(a[k] for a in ang)), float(max(a[k] for a in ang))] for k in ang[0]},
                       'source_frame_range': [float(min(src)), float(max(src))]}
    res['idle_window'] = r.default_idle_window()
    out.write_text(json.dumps(res, indent=1, default=float) + '\n')
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    a = ap.parse_args()
    if a.validate:
        validate()
