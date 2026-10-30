#!/usr/bin/env python3
"""v6 recipient (the person) = recipient5.Recipient5 (anatomical 2-bone IK, quintic hand goals, idle ping-pong, causal,
append-only schedules) + a FINGER CURL of the left hand (curl.py) whose physics colliders are the curled visible hand
(build_variant_scene.py --hand curl, curl_colliders.py).

Curl schedule (append-only, decided online):
    rcp.curl(now, target, duration=None, note='')   # quintic from the CURRENT (c, dc/dt, d2c/dt2) to (target, 0, 0)
        duration = max(duration, 0.25 s, 1.875 |dc| * tip_speed_per_c / v_tip_max)   (min-jerk peak speed limit)
        v_tip_max (Limits6.v_tip, default 0.35 m/s) bounds the fastest skin vertex of the hand (the fingertips),
        relative to the hand; tip_speed_per_c is measured on the rig (~0.18 m per unit curl).
    rcp.curl_value(t), rcp.curl_rate(t)
Mocap: every scene body recipient_hand_l[_<segment>] is posed from the evaluated hand bone (p_h, R_h) and c(t):
    pos = p_h + R_h (R_B rh + t_B - rh),  R = R_h R_B   (T_B(c) = rig.segment_T(segment, c))
recipient-motion.json: every frame carries `curl_l` (the value the website morph uses) and adaptation.curl_l holds
the rig definition (curl_rig, the exact formula and parameters), the schedule and the decisions.

Overhand receive helpers (all causal: they read only the current state and the given measurements):
    R = rcp.overhand_R(bar_dir, away_dir)          # hand-frame rotation: palm down, fingers horizontal across the bar
                                                   #   pointing away from him (over the bar), so curling hooks it
    p = rcp.hook_offset(c=1.0)                     # hand-frame vector palm centre -> hook point (inside the curled
                                                   #   fingers, between the proximal and distal segments)
    ok, info = rcp.check_goal(now, pos, R)         # anatomical feasibility of a hand-frame goal at s(now): planned
                                                   #   swivel, unclamped angles, orientation the clamps would give way
    T = rcp.reach(now, pos, R, ...)                # (Recipient5) quintic hand goal, natural swivel search
    rcp.idle(now)                                  # (Recipient5) ping-pong his quiet frames while waiting
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

import curl as CU
import curl_colliders as CC
import recipient5 as RCP5
import skin as SKN
from recipient import q2R

HERE = Path(__file__).resolve().parent
DEFAULT_SCENE = HERE / 'g1_umi_room_v9_body.xml'     # v9 (v7: g1_umi_room_v7_body.xml, v6: g1_umi_room_v6_curl.xml)


@dataclass
class Limits6(RCP5.Limits):
    v_tip: float = 0.35          # m/s, fastest hand skin vertex relative to the hand during a curl (planned peak)
    min_curl_T: float = 0.25


class CurlTrack:
    """scalar quintic chain, append-only; each segment starts from the current (c, v, a) -> C2 re-plans."""

    def __init__(self):
        self.segs = [(0.0, 1e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 'start')]     # (t0, T, c0, v0, a0, c1, decided_at, note)

    @staticmethod
    def _q(x0, v0, a0, x1, T, tau):
        tau = min(max(tau, 0.0), T)
        T2, T3, T4, T5 = T * T, T ** 3, T ** 4, T ** 5
        d = x1 - x0 - v0 * T - a0 * T2 / 2
        M = np.array([[T3, T4, T5], [3 * T2, 4 * T3, 5 * T4], [6 * T, 12 * T2, 20 * T3]])
        c3, c4, c5 = np.linalg.solve(M, np.array([d, -v0 - a0 * T, -a0]))
        x = x0 + v0 * tau + a0 / 2 * tau ** 2 + c3 * tau ** 3 + c4 * tau ** 4 + c5 * tau ** 5
        v = v0 + a0 * tau + 3 * c3 * tau ** 2 + 4 * c4 * tau ** 3 + 5 * c5 * tau ** 4
        a = a0 + 6 * c3 * tau + 12 * c4 * tau ** 2 + 20 * c5 * tau ** 3
        return x, v, a

    def state(self, t):
        i = len(self.segs) - 1
        while i > 0 and self.segs[i][0] > t:
            i -= 1
        t0, T, c0, v0, a0, c1, _, _ = self.segs[i]
        return self._q(c0, v0, a0, c1, T, t - t0)

    def append(self, now, T, c1, note=''):
        if now < self.segs[-1][0] - 1e-12:
            raise ValueError('curl schedule is append-only')
        c, v, a = self.state(now)
        self.segs.append((float(now), float(T), float(c), float(v), float(a), float(c1), float(now), note))

    def to_json(self):
        return [{'t0': s[0], 'T': s[1], 'c0': s[2], 'v0': s[3], 'a0': s[4], 'c1': s[5], 'decided_at': s[6], 'note': s[7]}
                for s in self.segs]


class Recipient6(RCP5.Recipient5):
    hand = 'curl'

    def __init__(self, start_frame=RCP5.START_FRAME, limits: Limits6 | None = None, mocap_bodies=None, curl_bodies=None,
                 curl_rig=None, idle_window=None, skin: SKN.Skin | None = None, **kw):
        self.sk = skin or SKN.Skin()
        lim = limits or Limits6()
        curl_bodies = dict(curl_bodies or {})
        bone_bodies = [b for b in (mocap_bodies or []) if b not in curl_bodies]
        super().__init__(start_frame=start_frame, limits=lim, mocap_bodies=bone_bodies or None, idle_window=idle_window,
                         skin=self.sk, **kw)
        self.rig = CU.CurlRig.from_params(curl_rig, self.sk) if curl_rig is not None else CU.CurlRig.build(self.sk)
        self.curl_bodies = curl_bodies
        self.bone_bodies = tuple(bone_bodies)
        self.mocap_bodies = tuple(bone_bodies) + tuple(curl_bodies)
        self.ctrack = CurlTrack()
        self.tip_per_c = self.rig.tip_speed_per_c()
        self.hb_idx = self.rec.idx['hand_l']

    @classmethod
    def for_scene(cls, xml=DEFAULT_SCENE, **kw):
        side = json.loads((Path(xml).parent / (Path(xml).stem + '.recipient.json')).read_text())
        assert side.get('hand') == 'curl', f'{xml} is not a curl scene'
        return cls(mocap_bodies=side['bodies'], curl_bodies=side['curl_bodies'], curl_rig=side['curl_rig'], **kw)

    # ---------------------------------------------------------------- curl schedule
    def curl_value(self, t):
        return float(min(1.0, max(0.0, self.ctrack.state(t)[0])))

    def curl_rate(self, t):
        return float(self.ctrack.state(t)[1])

    def curl(self, now, target, duration=None, note=''):
        target = float(min(1.0, max(0.0, target)))
        c0 = self.ctrack.state(now)[0]
        T = max(duration or 0.0, self.lim.min_curl_T, 1.875 * abs(target - c0) * self.tip_per_c / self.lim.v_tip)
        self.ctrack.append(now, T, target, note)
        self._cache_t = None
        self.decisions.append({'t': now, 'op': 'curl', 'side': 'l', 'from': float(c0), 'to': target, 'T': float(T),
                               'note': note, 'info_used': 'current curl state only',
                               'planned_peak_tip_speed_mps': 1.875 * abs(target - c0) * self.tip_per_c / T})
        return T

    # ---------------------------------------------------------------- evaluation
    def mocap(self, t):
        P, Q, s, w = self.full_pose(t)
        out = {}
        for b in self.bone_bodies:
            i = self.rec.idx[b.replace('recipient_', '')]
            out[b] = (P[i].copy(), Q[i].copy())
        c = self.curl_value(t)
        ph, qh = P[self.hb_idx], Q[self.hb_idx]
        Rh = q2R(qh)
        for b, seg in self.curl_bodies.items():
            out[b] = CC.body_pose(self.rig, seg, ph, Rh, c, q_ref=qh)
        return out

    def skin_vertices(self, t):
        """posed, curled skin (all body meshes) at robot time t (what the website shows)."""
        P, Q, _, _ = self.full_pose(t)
        return self.rig.pose(self.sk, P, Q, self.curl_value(t))

    # ---------------------------------------------------------------- overhand helpers
    def overhand_R(self, bar_dir, away_dir):
        """hand-frame rotation (columns x = palm normal, y, z = fingers): palm facing down, fingers horizontal,
        perpendicular to the bar, pointing away from him (over the bar)."""
        b = np.asarray(bar_dir, float)
        b = b - b[2] * np.array([0, 0, 1.0])
        b /= np.linalg.norm(b)
        z = np.asarray(away_dir, float)
        z = z - (z @ b) * b
        z[2] = 0
        z /= np.linalg.norm(z)
        x = np.array([0, 0, -1.0])
        y = np.cross(z, x)
        return np.stack([x, y, z], 1)

    def hook_offset(self, c=1.0, finger='middle'):
        """palm-centre hand frame -> hook point at curl c: midpoint of the palmar surfaces of the curled proximal and
        distal segments of `finger` (where a bar of a carry strap sits)."""
        rig = self.rig
        seg = rig.vertex_segments()
        Vc = rig.curled(c)
        an = self.anat['l']
        n = an.palm_n_rest
        pts = []
        for part in ('prox', 'dist'):
            sel = seg == f'{finger}_{part}'
            R, _ = rig.segment_T(f'{finger}_{part}', c)
            nn = R @ n
            X = Vc[sel]
            pal = X[(X - X.mean(0)) @ nn > np.quantile((X - X.mean(0)) @ nn, 0.8)]
            pts.append(pal.mean(0))
        hook_rest = 0.5 * (pts[0] + pts[1])
        Rhf, chf = self.hf['l']
        wr = self.sk.rest_head[rig.hb]
        return Rhf.T @ (hook_rest - wr - chf)

    def check_goal(self, now, pos, R, side='l'):
        sw, info = self.plan_swivel(now, side, np.asarray(pos, float), np.asarray(R, float))
        P, Q = self.rec.at(self.tm(now))
        Rhf, chf = self.hf[side]
        Rh_des = R @ Rhf.T
        Wpos = pos - Rh_des @ chf
        _, _, _, inf = self.solve_arm(P, Q, side, Wpos, Rh_des, sw, clamp=True)
        ok = inf['orientation_error_deg'] < 1.0 and inf['wrist_error_m'] < 0.002
        return ok, {'swivel_deg': sw, 'unclamped_angles': info['angles_unclamped'], 'clamped_angles': inf['angles'],
                    'orientation_given_way_deg': inf['orientation_error_deg'], 'wrist_error_m': inf['wrist_error_m'],
                    'reach_clamped_m': inf['reach_clamped_m']}

    # ---------------------------------------------------------------- export
    def schedules_json(self):
        out = super().schedules_json()
        out['curl_l'] = self.ctrack.to_json()
        out['hand'] = 'curl'
        return out

    def write_motion_json(self, path, t_end, fps=RCP5.FPS, extra=None):
        ex = {'curl_l': {'rig': self.rig.params, 'schedule': self.ctrack.to_json(),
                         'limits': {'v_tip_mps': self.lim.v_tip, 'tip_speed_per_unit_curl_mps': self.tip_per_c},
                         'frame_field': 'frames[i].curl_l = curl of the left hand at robot_time_s (0 open .. 1 hooked)',
                         'colliders': {b: s for b, s in self.curl_bodies.items()}},
              'hand_model': 'curl'}
        if extra:
            ex.update(extra)
        out = super().write_motion_json(path, t_end, fps=fps, extra=ex)
        for fr in out['frames']:
            fr['curl_l'] = round(self.curl_value(fr['robot_time_s']), 6)
        out['adaptation']['kind'] = out['adaptation']['kind'].replace('v5:', 'v6:') + \
            '; left-hand finger curl (curl.py rig) with per-segment collider bodies'
        Path(path).write_text(json.dumps(out, separators=(',', ':')) + '\n')
        return out


def validate(out=HERE / 'recipient6_check.json'):
    import time
    t0 = time.time()
    r = Recipient6.for_scene()
    res = {'tip_speed_per_unit_curl_mps': r.tip_per_c, 'n_curl_bodies': len(r.curl_bodies)}
    # identity at c = 0: every segment body == hand bone pose
    P, Q, _, _ = r.full_pose(0.5)
    mc = r.mocap(0.5)
    ih = r.rec.idx['hand_l']
    res['c0_segments_equal_hand_bone'] = bool(all(np.allclose(mc[b][0], P[ih], atol=1e-12) and
                                                  abs(abs(np.dot(mc[b][1], Q[ih])) - 1) < 1e-12 for b in r.curl_bodies))
    # schedule: idle, curl to 1 at 3 s, re-plan to 0.6 mid-way, open at 6 s
    r.idle(1.5)
    T1 = r.curl(3.0, 1.0, note='test close')
    T2 = r.curl(3.0 + 0.5 * T1, 0.6, note='test re-plan')
    T3 = r.curl(6.0, 0.0, note='test open')
    ts = np.arange(0, 8.0, 0.002)
    cv = np.array([r.curl_value(float(t)) for t in ts])
    dc = np.diff(cv) / 0.002
    res['schedule'] = {'T_close': T1, 'T_replan': T2, 'T_open': T3, 'c_range': [float(cv.min()), float(cv.max())],
                       'max_rate_per_s': float(np.abs(dc).max()),
                       'max_rate_jump_per_step': float(np.abs(np.diff(dc)).max()),
                       'max_tip_speed_rel_hand_mps': float(np.abs(dc).max() * r.tip_per_c)}
    # mocap segment bodies vs the curled skin they represent (vertex-level rigid error) at a few times
    errs = []
    for t in (3.2, 3.6, 4.5, 6.3):
        P, Q, _, _ = r.full_pose(t)
        c = r.curl_value(t)
        V = r.rig.pose(r.sk, P, Q, c)
        mcp = r.mocap(t)
        seg = r.rig.vertex_segments()
        rh = r.sk.rest_head[r.rig.hb]
        e = 0.0
        for b, sg in r.curl_bodies.items():
            ii = np.flatnonzero(seg == sg)
            if not len(ii):
                continue
            pos, q = mcp[b]
            Xw = (r.rig.V0r[ii] - rh) @ q2R(q).T + pos
            e = max(e, float(np.linalg.norm(Xw - V[r.rig.vid[ii]], axis=1).max()))
        errs.append((t, c, e * 1000))
    res['segment_body_vs_curled_skin_vertices_mm'] = errs
    res['hook_offset_hand_frame_m'] = r.hook_offset(1.0).tolist()
    res['time_s'] = time.time() - t0
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
