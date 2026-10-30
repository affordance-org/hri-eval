#!/usr/bin/env python3
"""v6 finger-curl rig for the person's LEFT hand (the avatar has no finger bones).

The rig deforms the avatar's REST (bind) vertices of the left hand; skinning is unchanged.  Because every vertex the
rig moves is skinned 100 % to hand_l (verified at build time, `weights_check`), the curled skin at any pose is
    V(P, Q, c) = LBS(V0 + delta(c))  =  LBS(V0) + R_hand_l @ delta(c)        (exact; also = glTF morph before skinning)

Geometry (all measured on the avatar mesh, hand_l rest frame = room rest pose, rest rotation I):
  * hand frame from skin.HandAnatomy: h = wrist -> middle fingertip, n = palm normal (palm side), r = radial (thumb).
  * fingers: the four connected components of the hand mesh beyond the finger webs (index, middle, ring, little;
    sorted by the radial coordinate); each finger's web level is the lowest cut at which its component separates
    from its neighbours; finger axis a_k = PCA of the component; centreline through its centroid (mid-thickness).
  * proximal phalanx length P1 = visible length / 1.8 (P1 : P2 : P3 ~ 1 : 0.6 : 0.45, web ~ 0.35 P1 distal of the
    MCP joint);  MCP_k = web - 0.35 P1,  PIP_k = MCP_k + P1, both on the finger centreline.
  * MCP hinge: ONE line through the four MCP points (least squares; the knuckle line is slanted across the hand),
    axis e = line direction oriented along h x n so that +angle flexes toward the palm.  Band coordinate
    L_mcp(x) = (x - p_mcp) . d_mcp, d_mcp = unit(e x n) (distal).
  * PIP hinge per finger: line through PIP_k along e_k = unit(a_k x n_k) (n_k = n made orthogonal to a_k),
    band coordinate L_pip,k(x) = (x - PIP_k) . a_k.
  * thumb: vertices radial of the index column (lateral coordinate along e beyond the index centreline + 9..15 mm,
    smoothstep) are excluded from the MCP rotation (thumb rigid with the palm; no opposition in this rig).
Deformation (curl c in [0, 1]):
    phi_pip(x) = c * PIP_DEG * S((L_pip,k(x) - lo2) / (hi2 - lo2))   only for vertices of finger k's component
    phi_mcp(x) = c * MCP_DEG * S((L_mcp(x) - lo1) / (hi1 - lo1)) * (1 - thumb(x))
    x' = Rot(p_mcp, e, phi_mcp) [ Rot(PIP_k, e_k, phi_pip) x ]       (S = smoothstep; bands +-w/2 around the hinges)
  The angle varies smoothly across each band, so the skin stays continuous (no torn or duplicated triangles).
Rigid parts for the physics colliders: every vertex with the same (L_mcp, L_pip) moves rigidly, so a SLICE of a band
(thin in L) moves rigidly with the slice-centre angles; the rig exports these segment transforms
    T_seg(c) = Rot(p_mcp, e, c*MCP*s1_seg) o Rot(PIP_k, e_k, c*PIP*s2_seg)
for: palm (identity), mcp slices j (common to all fingers), finger k proximal (s1 = 1, s2 = 0), finger k PIP slices,
finger k distal (s1 = s2 = 1).  `segment_of_points(X)` assigns rest points to segments.

API
    rig = CurlRig.build(sk)                    # from the avatar (or CurlRig.from_params(json_dict, sk))
    rig.params                                 # json-able rig definition (written to recipient-motion.json)
    D  = rig.delta(c)                          # (n_rig, 3) bind-space delta of rig.vid (mesh_0 vertex ids)
    V0c = rig.rest_vertices(c, V0)             # V0 with the curl applied (any array indexed like sk.V0)
    V  = rig.pose(sk, P, Q, c)                 # posed skin with curl (all body meshes)
    R, t = rig.segment_T(seg, c)               # rigid rest-space transform of a segment: x' = R x + t
    seg = rig.segment_of_points(X)             # segment name per rest point (palm / mcpJ / <finger>_prox|pipJ|dist)
    rig.tip_speed_per_c()                      # max |d vertex / d c| (m per unit curl), for speed limits
CLI: python curl.py --report [--png DIR]  -> curl_rig.json (params + checks), PNG renders at c = 0, 0.5, 1
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import skin as SKN

HERE = Path(__file__).resolve().parent
FINGERS = ('index', 'middle', 'ring', 'little')
MCP_DEG = 70.0
PIP_DEG = 80.0
W_MCP = 0.024
W_PIP = 0.016
K_MCP = 8            # rigid slices per band (collider bodies)
K_PIP = 8


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def rot_axis(a, phi):
    """rotation matrices about unit axis a by angles phi (n,) -> (n, 3, 3) (Rodrigues)."""
    a = np.asarray(a, float)
    phi = np.atleast_1d(phi)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s, c = np.sin(phi)[:, None, None], np.cos(phi)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


def rotate_about_line(X, p, a, phi):
    """rotate points X (n,3) about the line (p, a) by per-point angles phi (n,)."""
    R = rot_axis(a, phi)
    return np.einsum('nij,nj->ni', R, X - p) + p


def deform(V, params, c):
    """Pure formula used by the harness AND the exporter: curled rest positions of the rig vertices V (n,3) (rest
    positions of params['vertex_ids'], in that order) at curl c.  Returns (n,3)."""
    V = np.asarray(V, float)
    lab = np.asarray(params['label'], int)
    thumb = np.asarray(params['thumb_weight'], float)
    out = V.copy()
    for k, f in enumerate(params['pip']):
        sel = lab == k
        if not sel.any():
            continue
        p, e, d = np.array(f['point']), np.array(f['axis']), np.array(f['dir'])
        lo, hi = f['band']
        s2 = smoothstep(((V[sel] - p) @ d - lo) / (hi - lo))
        out[sel] = rotate_about_line(out[sel], p, e, math.radians(f['angle_deg']) * c * s2)
    m = params['mcp']
    p, e, d = np.array(m['point']), np.array(m['axis']), np.array(m['dir'])
    lo, hi = m['band']
    s1 = smoothstep(((V - p) @ d - lo) / (hi - lo)) * (1.0 - thumb)
    return rotate_about_line(out, p, e, math.radians(m['angle_deg']) * c * s1)


class CurlRig:
    def __init__(self, params, sk: SKN.Skin):
        self.params = params
        self.sk = sk
        self.vid = np.asarray(params['vertex_ids'], int)        # global ids into sk.V0 (mesh_0 is first: = local ids)
        self.V0r = sk.V0[self.vid]
        self.label = np.asarray(params['label'], int)
        self.thumb = np.asarray(params['thumb_weight'], float)
        self.hb = sk.idx[params['bone']]
        self._segments()

    # ------------------------------------------------------------------------------------------------ build
    @classmethod
    def build(cls, sk: SKN.Skin | None = None, side='l', mcp_deg=MCP_DEG, pip_deg=PIP_DEG, w_mcp=W_MCP, w_pip=W_PIP,
              k_mcp=K_MCP, k_pip=K_PIP, verbose=False):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        sk = sk or SKN.Skin()
        bone = f'hand_{side}'
        hb = sk.idx[bone]
        an = SKN.HandAnatomy(sk, side)
        h, n, r = an.hand_axis_rest, an.palm_n_rest, an.radial_rest
        wr = sk.rest_head[hb]
        V = sk.V0
        cand = np.flatnonzero((sk.part == 0) & (sk.dom == hb))          # skin mesh vertices of the hand
        X = V - wr
        L, Y = X @ h, X @ r
        F = sk.F[sk.face_part == 0]
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        nv = len(V)
        is_c = np.zeros(nv, bool)
        is_c[cand] = True

        def components(thr):
            sel = is_c & (L > thr)
            e = E[sel[E[:, 0]] & sel[E[:, 1]]]
            A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
            _, lab = connected_components(A, directed=False)
            return sel, lab
        # four finger tips: components beyond a cut where exactly four large components exist
        cut0 = None
        for thr in np.arange(0.10, 0.14, 0.002):
            sel, lab = components(thr)
            ids, cnt = np.unique(lab[sel], return_counts=True)
            big = ids[cnt >= 30]
            if len(big) == 4:
                cut0 = thr
                break
        assert cut0 is not None, 'could not find four separate fingers'
        comps = []
        for cid in big:
            vs = np.flatnonzero((lab == cid) & sel)
            comps.append(vs)
        comps.sort(key=lambda vs: -float(Y[vs].mean()))                 # radial (thumb side) first: index ... little
        tips = [int(vs[np.argmax(L[vs])]) for vs in comps]
        # web level of each finger: lowest cut at which its tip's component contains no other tip
        webs, members = [], []
        for k in range(4):
            best = cut0
            for thr in np.arange(cut0, 0.05, -0.001):
                sel, lab = components(thr)
                others = [lab[t] for j, t in enumerate(tips) if j != k]
                if lab[tips[k]] in others:
                    break
                best = thr
            sel, lab = components(best)
            vs = np.flatnonzero((lab == lab[tips[k]]) & sel)
            webs.append(float(best))
            members.append(vs)
        fingers = []
        for k, vs in enumerate(members):
            P_ = V[vs]
            cen = P_.mean(0)
            _, _, Vt = np.linalg.svd(P_ - cen, full_matrices=False)
            a = Vt[0] * np.sign(Vt[0] @ h)
            s = (P_ - cen) @ a
            s_tip, s_web = float(s.max()), float(s.min())
            ell = s_tip - s_web
            P1 = ell / 1.8
            s_mcp = s_web - 0.35 * P1
            s_pip = s_mcp + P1
            nk = n - (n @ a) * a
            nk /= np.linalg.norm(nk)
            ek = np.cross(a, nk)
            fingers.append({'name': FINGERS[k], 'n_vertices': int(len(vs)), 'web_cut_L_m': webs[k],
                            'axis_dir': a.tolist(), 'centroid': cen.tolist(), 'visible_length_m': ell,
                            'proximal_phalanx_m': P1,
                            'mcp_point': (cen + s_mcp * a).tolist(), 'pip_point': (cen + s_pip * a).tolist(),
                            'pip_axis': ek.tolist(), 'tip_vertex': tips[k],
                            'members': vs})
        # common MCP line through the four MCP points
        M = np.array([f['mcp_point'] for f in fingers])
        mc = M.mean(0)
        _, _, Vt = np.linalg.svd(M - mc, full_matrices=False)
        e = Vt[0]
        if e @ np.cross(h, n) < 0:
            e = -e
        d1 = np.cross(e, n)
        d1 /= np.linalg.norm(d1)
        if d1 @ h < 0:
            d1 = -d1
        mcp_resid = float(np.max(np.linalg.norm((M - mc) - np.outer((M - mc) @ e, e), axis=1)))
        # hinge at mid-thickness of the hand along the knuckle line (palm normal n), thumb excluded below
        lat_dir0 = -e if (-e) @ r > 0 else e
        u_all = (V[cand] - mc) @ lat_dir0
        u_idx = float((np.array(fingers[0]['mcp_point']) - mc) @ lat_dir0)
        near = (np.abs((V[cand] - mc) @ d1) < 0.005) & (u_all < u_idx + 0.009)
        Nn = (V[cand][near] - mc) @ n
        mid_shift = 0.5 * (float(Nn.min()) + float(Nn.max()))
        mc = mc + mid_shift * n
        # thumb exclusion: lateral coordinate along +r side of the index centreline
        lat_dir = -e if (-e) @ r > 0 else e                               # toward the thumb
        u_index = float((np.array(fingers[0]['mcp_point']) - mc) @ lat_dir)
        u = (V[cand] - mc) @ lat_dir
        thumb_w = smoothstep((u - (u_index + 0.009)) / 0.006)
        # labels: finger k for its component, -1 otherwise
        label = np.full(len(cand), -1, int)
        pos = {int(v): i for i, v in enumerate(cand)}
        for k, f in enumerate(fingers):
            for v in f['members']:
                label[pos[int(v)]] = k
                thumb_w[pos[int(v)]] = 0.0
        params = {'format': 'h2r-hand-curl-rig/1', 'bone': bone, 'mesh_key': 'mesh_0',
                  'frame': 'avatar rest (bind) coordinates = room rest pose; the bone rest rotation is identity',
                  'formula': ("x' = Rot(mcp.point, mcp.axis, c*mcp.angle*S((x-mcp.point).mcp.dir in band)*(1-thumb)) "
                              "o Rot(pip[k].point, pip[k].axis, c*pip[k].angle*S((x-pip[k].point).pip[k].dir in band)) "
                              "[pip only for label k]; S = smoothstep (3u^2 - 2u^3) of (s - lo)/(hi - lo)"),
                  'vertex_ids': cand.tolist(), 'label': label.tolist(),
                  'thumb_weight': np.round(thumb_w, 6).tolist(),
                  'finger_names': list(FINGERS),
                  'mcp': {'point': mc.tolist(), 'axis': e.tolist(), 'dir': d1.tolist(), 'band': [-w_mcp / 2, w_mcp / 2],
                          'angle_deg': mcp_deg, 'line_fit_residual_m': mcp_resid, 'slices': k_mcp,
                          'hinge_shift_to_mid_thickness_m': mid_shift,
                          'hand_thickness_at_knuckles_m': [float(Nn.min() - mid_shift), float(Nn.max() - mid_shift)]},
                  'pip': [{'name': f['name'], 'point': f['pip_point'], 'axis': f['pip_axis'], 'dir': f['axis_dir'],
                           'band': [-w_pip / 2, w_pip / 2], 'angle_deg': pip_deg, 'slices': k_pip} for f in fingers],
                  'fingers': [{k_: v for k_, v in f.items() if k_ != 'members'} for f in fingers]}
        rig = cls(params, sk)
        if verbose:
            print(json.dumps({k: v for k, v in params.items() if k not in ('vertex_ids', 'label', 'thumb_weight')}, indent=1))
        return rig

    @classmethod
    def from_params(cls, params, sk=None):
        return cls(params, sk or SKN.Skin())

    # ------------------------------------------------------------------------------------------------ deformation
    def curled(self, c):
        return deform(self.V0r, self.params, float(c))

    def delta(self, c):
        return self.curled(c) - self.V0r

    def rest_vertices(self, c, V0=None):
        V = (self.sk.V0 if V0 is None else V0).copy()
        V[self.vid] = self.curled(c)
        return V

    def pose(self, sk, P, Q, c, V=None):
        """posed vertices (all body meshes) with the curl: LBS(V0) + sum_i w_i R_i delta (exact LBS of V0 + delta)."""
        V = sk.pose(P, Q) if V is None else V.copy()
        if c == 0:
            return V
        D = self.delta(c)
        R = SKN.q2R_batch(Q)
        W = sk.W[self.vid]
        moved = np.flatnonzero(np.abs(D).max(1) > 0)
        acc = np.zeros((len(moved), 3))
        for b in np.flatnonzero(W[moved].max(0) > 0):
            acc += W[moved, b][:, None] * (D[moved] @ R[b].T)
        V[self.vid[moved]] += acc
        return V

    # ------------------------------------------------------------------------------------------------ segments
    def _segments(self):
        """Rigid segments = bins of the blend values (s1_eff = S(L_mcp) (1 - thumb), s2 = S(L_pip,k)).  Every point
        with the same (s1_eff, s2, finger) moves rigidly; a bin of width 1/K moves with its centre value, so the rigid
        approximation error of a bin is <= angle / (2K) times the distance from the hinge."""
        p = self.params
        m = p['mcp']
        self.mcp_p, self.mcp_e, self.mcp_d = (np.array(m[k], float) for k in ('point', 'axis', 'dir'))
        self.mcp_band = tuple(m['band'])
        self.km = int(m.get('slices', K_MCP))
        self.seg = {'palm': {'s1': 0.0, 's2': 0.0, 'finger': -1}}
        for j in range(self.km):
            self.seg[f'mcp{j}'] = {'s1': (j + 0.5) / self.km, 's2': 0.0, 'finger': -1,
                                   's1_bin': [j / self.km, (j + 1) / self.km]}
        self.kp = []
        for k, f in enumerate(p['pip']):
            nm = f['name']
            kp = int(f.get('slices', K_PIP))
            self.kp.append(kp)
            self.seg[f'{nm}_prox'] = {'s1': 1.0, 's2': 0.0, 'finger': k}
            for j in range(kp):
                self.seg[f'{nm}_pip{j}'] = {'s1': 1.0, 's2': (j + 0.5) / kp, 'finger': k, 's2_bin': [j / kp, (j + 1) / kp]}
            self.seg[f'{nm}_dist'] = {'s1': 1.0, 's2': 1.0, 'finger': k}
        self.segment_names = list(self.seg)
        self.vs1, self.vs2 = self.blend_values(self.V0r, self.label, self.thumb)

    def blend_values(self, X, label, thumb):
        """(s1_eff, s2) of rest points X with finger label (-1 none) and thumb weight."""
        X = np.atleast_2d(X)
        lo, hi = self.mcp_band
        s1 = smoothstep(((X - self.mcp_p) @ self.mcp_d - lo) / (hi - lo)) * (1.0 - np.asarray(thumb, float))
        s2 = np.zeros(len(X))
        for k, f in enumerate(self.params['pip']):
            sel = np.asarray(label) == k
            if sel.any():
                s2[sel] = smoothstep(((X[sel] - np.array(f['point'])) @ np.array(f['dir']) - f['band'][0]) /
                                     (f['band'][1] - f['band'][0]))
        return s1, s2

    def segment_T(self, name, c):
        """rest-space rigid transform of a segment at curl c: x' = R x + t."""
        s = self.seg[name]
        R = np.eye(3)
        t = np.zeros(3)
        if s['finger'] >= 0 and s['s2'] > 0:
            f = self.params['pip'][s['finger']]
            R2 = rot_axis(np.array(f['axis']), math.radians(f['angle_deg']) * c * s['s2'])[0]
            pk = np.array(f['point'])
            R, t = R2, pk - R2 @ pk
        if s['s1'] > 0:
            R1 = rot_axis(self.mcp_e, math.radians(self.params['mcp']['angle_deg']) * c * s['s1'])[0]
            R, t = R1 @ R, R1 @ (t - self.mcp_p) + self.mcp_p
        return R, t

    def segment_of_s(self, s1, s2, finger, X=None):
        """segment names from blend values (vertices: rig.vs1/vs2/label; triangle samples: barycentric values,
        finger = label of the dominant vertex).  X (rest points) picks the nearest finger for skin beyond the MCP band
        that is not on a finger component (palm skin at the finger bases: rigid with the proximal phalanges)."""
        s1, s2, finger = np.atleast_1d(s1), np.atleast_1d(s2), np.atleast_1d(finger)
        out = np.empty(len(s1), object)
        eps = 1e-9
        for i in range(len(s1)):
            if s1[i] <= eps:
                out[i] = 'palm'
            elif s1[i] < 1 - eps:
                # a point with both blends active exists only inside a long triangle spanning both bands (never a
                # vertex: the bands are disjoint along every finger); it is not rigid with any segment
                out[i] = 'mixed' if s2[i] > eps else f'mcp{min(self.km - 1, int(s1[i] * self.km))}'
            else:
                k = int(finger[i])
                if k < 0:
                    k = self._nearest_finger(X[i]) if X is not None else 0
                    out[i] = f'{FINGERS[k]}_prox'
                    continue
                nm, kp = self.params['pip'][k]['name'], self.kp[k]
                if s2[i] <= eps:
                    out[i] = f'{nm}_prox'
                elif s2[i] < 1 - eps:
                    out[i] = f'{nm}_pip{min(kp - 1, int(s2[i] * kp))}'
                else:
                    out[i] = f'{nm}_dist'
        return out

    def segment_of_points(self, X):
        """segment per rest point (labels/thumb weight of the nearest rig vertex)."""
        from scipy.spatial import cKDTree
        X = np.atleast_2d(X)
        _, ii = cKDTree(self.V0r).query(X)
        s1, s2 = self.blend_values(X, self.label[ii], self.thumb[ii])
        return self.segment_of_s(s1, s2, self.label[ii], X)

    def vertex_segments(self):
        return self.segment_of_s(self.vs1, self.vs2, self.label, self.V0r)

    def _nearest_finger(self, x):
        best, bk = 1e9, 0
        for k, f in enumerate(self.params['fingers']):
            a, c0 = np.array(f['axis_dir']), np.array(f['centroid'])
            d = np.linalg.norm((x - c0) - ((x - c0) @ a) * a)
            if d < best:
                best, bk = d, k
        return bk

    # ------------------------------------------------------------------------------------------------ checks
    def tip_speed_per_c(self, n=21):
        cs = np.linspace(0, 1, n)
        Vs = np.array([self.curled(c) for c in cs])
        return float((np.linalg.norm(np.diff(Vs, axis=0), axis=2).max(1) / (cs[1] - cs[0])).max())

    def weights_check(self):
        D = self.delta(1.0)
        moved = np.abs(D).max(1) > 1e-9
        w = self.sk.W[self.vid[moved], self.hb]
        others = np.delete(self.sk.W[self.vid[moved]], self.hb, axis=1).max(1)
        return {'moved_vertices': int(moved.sum()), 'min_hand_weight_of_moved': float(w.min()),
                'max_other_bone_weight_of_moved': float(others.max()),
                'all_moved_skinned_to_hand_only': bool(w.min() >= 1 - 1e-6),
                'blend_vertices_of_hand_bone_total': int(((self.sk.W[:, self.hb] > 1e-6) & (self.sk.W[:, self.hb] < 1 - 1e-6)).sum()),
                'blend_vertices_moved': int((moved & (self.sk.W[self.vid, self.hb] < 1 - 1e-6)).sum())}


# ------------------------------------------------------------------------------------------------ reports / renders
def fold_check(sk, rig, c):
    """faces of the hand whose orientation flips relative to the rigidly rotated rest face (folds), area ratios."""
    V0 = sk.V0
    V1 = rig.rest_vertices(c)
    m = np.zeros(len(V0), bool)
    m[rig.vid] = True
    F = sk.F[m[sk.F].all(1)]
    n0 = np.cross(V0[F[:, 1]] - V0[F[:, 0]], V0[F[:, 2]] - V0[F[:, 0]])
    n1 = np.cross(V1[F[:, 1]] - V1[F[:, 0]], V1[F[:, 2]] - V1[F[:, 0]])
    a0, a1 = np.linalg.norm(n0, axis=1), np.linalg.norm(n1, axis=1)
    # local rotation of each face: average rotation of its rig vertices ~ use the segment transform of the centroid
    vseg = np.empty(len(V0), object)
    vseg[rig.vid] = rig.vertex_segments()
    seg = vseg[F[:, 0]]
    flips = 0
    cosmin = 1.0
    for i in range(len(F)):
        R, _ = rig.segment_T(seg[i], c)
        cs = float((R @ n0[i]) @ n1[i] / max(1e-12, a0[i] * a1[i]))
        cosmin = min(cosmin, cs)
        flips += cs < 0
    ratio = a1 / np.maximum(a0, 1e-12)
    return {'faces': int(len(F)), 'flipped_faces': int(flips), 'min_cos_to_rigid_normal': cosmin,
            'area_ratio_min': float(ratio.min()), 'area_ratio_max': float(ratio.max())}


def self_contact(sk, rig, c):
    """closest approach of the curled distal finger segments to the palm (rig vertices that do not move)."""
    from scipy.spatial import cKDTree
    Vc = rig.curled(c)
    seg = rig.vertex_segments()
    dist = np.array([s.endswith('_dist') for s in seg])
    palm = np.array([s == 'palm' for s in seg]) & (rig.thumb < 0.5)
    d, _ = cKDTree(Vc[palm]).query(Vc[dist])
    return {'min_fingertip_segment_to_palm_vertex_mm': float(d.min() * 1000)}


def render_pngs(sk, rig, out_dir, cs=(0.0, 0.5, 1.0)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    an = SKN.HandAnatomy(sk, 'l')
    h, n, r = an.hand_axis_rest, an.palm_n_rest, an.radial_rest
    wr = sk.rest_head[rig.hb]
    reg = np.zeros(len(sk.V0), bool)
    reg[sk.bone_vertices('hand_l', 0.01)] = True
    reg[sk.bone_vertices('lowerarm_l', 0.5)] = True
    F = sk.F[reg[sk.F].all(1) & (sk.face_part[np.arange(len(sk.F))] == 0)]
    views = [('dorsal', None), ('radial side', None), ('oblique', None)]
    files = []
    light = np.array([0.3, -0.5, 0.8])
    light /= np.linalg.norm(light)
    fig, axes = plt.subplots(len(views), len(cs), figsize=(4.2 * len(cs), 4.0 * len(views)), subplot_kw={'projection': '3d'})
    for ci, c in enumerate(cs):
        V = rig.rest_vertices(c) - wr
        # hand coordinates (x = along the hand, y = palm normal, z = radial)
        Hc = np.stack([V @ h, V @ n, V @ r], 1)
        for vi, (vname, _) in enumerate(views):
            ax = axes[vi, ci]
            tri = Hc[F]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            if vi == 0:
                ax.view_init(elev=0, azim=-90)      # look along +z (radial side)
            elif vi == 1:
                ax.view_init(elev=90, azim=-90)     # look from the palm side
            else:
                ax.view_init(elev=25, azim=-55)
            sh = 0.35 + 0.65 * np.clip(np.abs(nrm @ np.array([0.2, 0.6, 0.77])), 0, 1)
            col = np.c_[0.86 * sh, 0.68 * sh, 0.6 * sh, np.ones(len(sh))]
            pc = Poly3DCollection(tri, facecolors=col, edgecolors='none')
            ax.add_collection3d(pc)
            ax.set_xlim(-0.02, 0.19)
            ax.set_ylim(-0.105, 0.105)
            ax.set_zlim(-0.105, 0.105)
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            ax.set_title(f'c = {c:.1f}  ({vname})', fontsize=10)
    fig.tight_layout()
    p = out_dir / 'curl_c0_c05_c1.png'
    fig.savefig(p, dpi=110)
    plt.close(fig)
    files.append(str(p))
    for c in cs:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        V = rig.rest_vertices(c) - wr
        Hc = np.stack([V @ h, V @ n, V @ r], 1)
        tri = Hc[F]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        sh = 0.35 + 0.65 * np.clip(np.abs(nrm @ np.array([0.2, 0.6, 0.77])), 0, 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.c_[0.86 * sh, 0.68 * sh, 0.6 * sh, np.ones(len(sh))], edgecolors='none'))
        ax.view_init(elev=20, azim=-70)
        ax.set_xlim(-0.02, 0.19)
        ax.set_ylim(-0.105, 0.105)
        ax.set_zlim(-0.105, 0.105)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(f'left hand curl c = {c:.1f}')
        p = out_dir / f'curl_c{c:.1f}.png'
        fig.savefig(p, dpi=110)
        plt.close(fig)
        files.append(str(p))
    return files


def report(out=HERE / 'curl_rig.json', png_dir=HERE / 'curl_renders'):
    import time
    t0 = time.time()
    sk = SKN.Skin()
    rig = CurlRig.build(sk)
    res = {'params_summary': {k: v for k, v in rig.params.items() if k not in ('vertex_ids', 'label', 'thumb_weight')},
           'n_rig_vertices': int(len(rig.vid)), 'n_finger_vertices': int((rig.label >= 0).sum()),
           'n_thumb_excluded': int((rig.thumb > 0.5).sum()),
           'weights_check': rig.weights_check(),
           'tip_speed_per_unit_curl_mps': rig.tip_speed_per_c(),
           'folds': {str(c): fold_check(sk, rig, c) for c in (0.5, 1.0)},
           'self_contact': {str(c): self_contact(sk, rig, c) for c in (0.5, 1.0)},
           'segments': {k: {kk: vv for kk, vv in v.items()} for k, v in rig.seg.items()},
           'vertex_segment_counts': {k: int(v) for k, v in zip(*np.unique(rig.vertex_segments().astype(str), return_counts=True))}}
    # identity: c = 0 is the rest mesh exactly
    res['c0_is_rest_exact'] = bool(np.array_equal(rig.curled(0.0), rig.V0r))
    # segment rigidity of the rig vertices (vertex vs its segment transform), mm
    seg = rig.vertex_segments()
    for c in (0.5, 1.0):
        Vc = rig.curled(c)
        err = np.zeros(len(seg))
        for s in set(seg):
            ii = np.flatnonzero(seg == s)
            R, t = rig.segment_T(s, c)
            err[ii] = np.linalg.norm(Vc[ii] - (rig.V0r[ii] @ R.T + t), axis=1)
        res[f'vertex_vs_segment_rigid_max_mm_c{c}'] = float(err.max() * 1000)
    res['pngs'] = render_pngs(sk, rig, png_dir) if png_dir else []
    res['time_s'] = time.time() - t0
    full = dict(res)
    full['params'] = rig.params
    Path(out).write_text(json.dumps(full, indent=1, default=float) + '\n')
    print(json.dumps(res, indent=1, default=float))
    return rig, res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--png', type=Path, default=HERE / 'curl_renders')
    a = ap.parse_args()
    if a.report:
        report(png_dir=a.png)
