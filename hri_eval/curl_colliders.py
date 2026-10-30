#!/usr/bin/env python3
"""v6 physics colliders that EQUAL the visible curled left hand (curl.py).

Segments (curl.CurlRig): palm (rigid with hand_l), mcp0..mcpK-1 (slices of the knuckle band, common to all fingers),
<finger>_prox, <finger>_pip0..K-1, <finger>_dist.  Each segment is ONE mocap body; its convex pieces are fitted to the
segment's own skin points (rig vertices + 2.5 mm area samples of the hand triangles, assigned by their barycentric
blend values, + points of the neighbouring segments within 1 mm so adjacent pieces overlap) with the v5 adaptive
quadtree (skin_colliders.fit_points_quadtree, outward bulge <= tol against the rest skin).

Body pose of segment B at hand-bone pose (p_h, R_h) and curl c (T_B(c) = (R_B, t_B), rest-space rigid transform):
    rest point x -> curled rest point R_B x + t_B -> world R_h (R_B x + t_B - rh) + p_h       (rh = hand rest head)
  mesh vertices are stored as x - rh (bone-local, as in v5), so
    body.pos = p_h + R_h (R_B rh + t_B - rh),   body.R = R_h R_B.
Names: recipient_hand_l (palm), recipient_hand_l_<segment> for the others.

API
    segs = segment_point_sets(sk, rig)                    # {segment: rest points}
    pieces = fit_segments(sk, rig, tol=0.0022)            # {segment: [piece dicts]}
    pos, quat = body_pose(rig, seg, p_h, R_h, c)
    rep = deviation(sk, rig, pieces, P, Q, c)             # outward / inward (mm) vs the posed CURLED skin
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

import skin as SKN
import skin_colliders as SC
from skin import _closest_on_triangles


def body_name(seg):
    return 'recipient_hand_l' if seg == 'palm' else f'recipient_hand_l_{seg}'


def seg_of_body(name):
    return 'palm' if name == 'recipient_hand_l' else name[len('recipient_hand_l_'):]


def R2q(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def body_pose(rig, seg, p_h, R_h, c, q_ref=None):
    rh = rig.sk.rest_head[rig.hb]
    RB, tB = rig.segment_T(seg, c)
    pos = p_h + R_h @ (RB @ rh + tB - rh)
    q = R2q(R_h @ RB)
    if q_ref is not None and np.dot(q, q_ref) < 0:
        q = -q
    return pos, q


def hand_faces(sk, rig):
    m = np.zeros(len(sk.V0), bool)
    m[rig.vid] = True
    return sk.F[m[sk.F].all(1)]


def segment_point_sets(sk, rig, spacing=0.0025, neighbour=0.001, seed=11):
    """rest points of every segment (vertices + triangle samples with barycentric blend values)."""
    rng = np.random.default_rng(seed)
    pos = -np.ones(len(sk.V0), int)
    pos[rig.vid] = np.arange(len(rig.vid))
    F = hand_faces(sk, rig)
    V = sk.V0
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = int(area.sum() / (spacing * spacing))
    f = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    X = bary[:, :1] * a[f] + bary[:, 1:2] * b[f] + bary[:, 2:3] * c[f]
    loc = pos[F[f]]
    s1 = (bary * rig.vs1[loc]).sum(1)
    s2 = (bary * rig.vs2[loc]).sum(1)
    lab = rig.label[loc].max(1)
    seg_s = rig.segment_of_s(s1, s2, lab, X)
    seg_v = rig.vertex_segments()
    assert 'mixed' not in set(seg_v.astype(str)), 'a rig vertex lies in both bands'
    keep = seg_s.astype(str) != 'mixed'
    allX = np.concatenate([rig.V0r, X[keep]])
    allS = np.concatenate([seg_v.astype(str), seg_s[keep].astype(str)])
    tree = cKDTree(allX)
    out = {}
    for sg in rig.segment_names:
        own = allS == sg
        if not own.any():
            continue
        nb = tree.query_ball_point(allX[own], neighbour)
        jj = np.unique(np.concatenate([np.asarray(x, int) for x in nb]))
        out[sg] = np.unique(np.concatenate([allX[own], allX[jj]]), axis=0)
    return out


def fit_segments(sk, rig, tol=0.0016, fit_curls=(0.0, 0.5, 1.0), min_cell=0.0025, verbose=False):
    """pieces per segment; a piece is accepted when its hull bulges <= tol outside the curled skin at EVERY curl in
    fit_curls (the segment moved rigidly with T_seg(c); rest arm pose)."""
    rh = sk.rest_head[rig.hb]
    surfs = [(c, sk.surface(rig.rest_vertices(c))) for c in fit_curls]
    sets = segment_point_sets(sk, rig)
    out = {}
    for sg, X in sets.items():
        Ts = [rig.segment_T(sg, c) for c, _ in surfs]

        def dev_fn(S, Ts=Ts):
            return max(float(su.signed_distance(S @ R.T + t)[0].max()) for (R, t), (_, su) in zip(Ts, surfs))
        pcs = SC.fit_points_quadtree(X, body_name(sg), rh, None, tol=tol, min_cell=min_cell, dev_fn=dev_fn)
        if pcs:
            out[sg] = pcs
        if verbose:
            print(f'  {sg}: {len(X)} points -> {len(pcs)} pieces', flush=True)
    return out


def deviation(sk, rig, pieces, P, Q, c, h_out=0.001, h_in=0.002, rng=None, extra_bones=None):
    """outward (hull surface outside the posed curled skin) and inward (posed curled hand skin not covered by the
    hulls) deviations in mm, per segment and overall.  P, Q: bone poses; c: curl."""
    rng = rng or np.random.default_rng(3)
    V = rig.pose(sk, P, Q, c)
    surf = sk.surface(V)
    ih = rig.hb
    Rh, ph = SKN.q2R(Q[ih]), P[ih]
    worst_out, per = 0.0, {}
    eqs = []
    for sg, pcs in pieces.items():
        pos, q = body_pose(rig, sg, ph, Rh, c)
        RB = SKN.q2R(q)
        o_sg = 0.0
        for pc in pcs:
            hl = ConvexHull(pc['verts_local'])
            Vw = pc['verts_local'] @ RB.T + pos
            S = SC.sample_tris(Vw, hl.simplices, h_out, rng)
            sd, _, _ = surf.signed_distance(S)
            o_sg = max(o_sg, float(max(0.0, sd.max())))
            A = hl.equations[:, :3] @ RB.T
            bb = hl.equations[:, 3] - A @ pos
            eqs.append((A, bb, Vw, hl.simplices))
        per[sg] = round(o_sg * 1000, 2)
        worst_out = max(worst_out, o_sg)
    F = hand_faces(sk, rig)
    S = SC.sample_tris(V, F, h_in, rng)
    inside = np.zeros(len(S), bool)
    for A, bb, _, _ in eqs:
        inside |= (S @ A.T + bb).max(1) <= 1e-9
    dist = np.zeros(len(S))
    ci = np.flatnonzero(~inside)
    if len(ci):
        best = np.full(len(ci), np.inf)
        cen = np.array([Vw.mean(0) for _, _, Vw, _ in eqs])
        rad = np.array([np.linalg.norm(Vw - Vw.mean(0), axis=1).max() for _, _, Vw, _ in eqs])
        for i, (A, bb, Vw, simp) in enumerate(eqs):
            near = np.linalg.norm(S[ci] - cen[i], axis=1) < rad[i] + 0.01
            if not near.any():
                continue
            cj = ci[near]
            a_, b_, c_ = Vw[simp[:, 0]], Vw[simp[:, 1]], Vw[simp[:, 2]]
            for j0 in range(0, len(cj), 256):
                ck = cj[j0:j0 + 256]
                pp = np.repeat(S[ck], len(simp), 0)
                qq, _ = _closest_on_triangles(pp, np.tile(a_, (len(ck), 1)), np.tile(b_, (len(ck), 1)), np.tile(c_, (len(ck), 1)))
                dd = np.linalg.norm(pp - qq, axis=1).reshape(len(ck), len(simp)).min(1)
                idx = np.flatnonzero(near)[j0:j0 + 256]
                best[idx] = np.minimum(best[idx], dd)
        best[~np.isfinite(best)] = 0.05
        dist[ci] = best
    worst_seg_in = None
    if len(S):
        k = int(np.argmax(dist))
        worst_seg_in = rig.segment_of_points(_rest_of_posed(sk, rig, S[k:k + 1], P, Q, c))[0]
    return {'c': c, 'outward_max_mm': round(worst_out * 1000, 2), 'inward_max_mm': round(float(dist.max()) * 1000, 2),
            'inward_p99_mm': round(float(np.quantile(dist, 0.99)) * 1000, 2), 'inward_worst_segment': worst_seg_in,
            'outward_per_segment_mm': per, 'n_pieces': int(sum(len(v) for v in pieces.values()))}


def _rest_of_posed(sk, rig, X, P, Q, c):
    """approximate rest point of a posed curled point (nearest posed rig vertex -> its rest position)."""
    V = rig.pose(sk, P, Q, c)
    _, i = cKDTree(V[rig.vid]).query(X)
    return rig.V0r[i]
