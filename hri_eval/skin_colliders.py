#!/usr/bin/env python3
"""v5 skin-matched recipient colliders: convex pieces fitted to the person's own avatar surface, in BONE frames.

For each bone B in (hand_l, lowerarm_l, upperarm_l, hand_r, lowerarm_r, upperarm_r) the vertices of all body meshes
(skin + clothing) whose DOMINANT skinning weight is B are expressed in B's local frame (rest vertex - rest_head_B; the
skinning formula maps it to the world by R_B x + p_B), so a mocap body posed at (p_B, R_B) carries them exactly.
The vertex set is decomposed into convex pieces by a deviation-driven recursive split:
    piece = convex hull of a vertex cluster (+ its 1-ring neighbours, so neighbouring pieces overlap)
    outward deviation = max signed distance (to the rest-pose skin union) of points sampled on the hull surface
                        (1 mm spacing) -> how far the collider bulges OUTSIDE the visible surface
    if outward > tol: split the cluster in two (2-means in 3D) and recurse (max depth / min size bounded)
Inward deviation (visible surface not covered by any piece) = max distance from skin surface samples of the bone's
region to the union of the pieces (0 inside any piece).  Both are reported at the rest pose (exact for the rigid part
of each bone) and at given poses (recorded frames; the wrist/elbow blend bands deform with the joint angle).

API
    pieces = fit_bone(skin, 'hand_l', tol=0.003)          # [{'name', 'verts_local' (k,3), 'n_src'}]
    rep = deviation_report(skin, {bone: pieces}, poses=[(P, Q), ...])
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

import skin as SKN

BONES = ('hand_l', 'lowerarm_l', 'upperarm_l', 'hand_r', 'lowerarm_r', 'upperarm_r')


def hull_tris(pts):
    h = ConvexHull(pts)
    return pts, h.simplices, h.equations


def sample_tris(V, F, h=0.001, rng=None):
    rng = rng or np.random.default_rng(0)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = int(max(50, area.sum() / (h * h)))
    f = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    P = (1 - s)[:, None] * a[f] + (s * (1 - r2))[:, None] * b[f] + (s * r2)[:, None] * c[f]
    return np.concatenate([V[np.unique(F)], P])


def _neighbours(F, idx, nv):
    m = np.zeros(nv, bool)
    m[idx] = True
    touch = m[F].any(1)
    out = np.unique(F[touch])
    return out


def fit_bone(sk: SKN.Skin, bone, tol=0.003, min_pts=12, max_depth=7, max_pieces=40, surf_rest=None, verbose=False):
    b = sk.idx[bone]
    rh = sk.rest_head[b]
    vid = np.flatnonzero(sk.dom == b)
    if surf_rest is None:
        surf_rest = sk.surface(sk.V0)
    nvt = len(sk.V0)
    pieces = []

    def outward(pts_world):
        P, F, _ = hull_tris(pts_world)
        S = sample_tris(P, F, 0.0015)
        sd, _, _ = surf_rest.signed_distance(S)
        return float(sd.max()), S

    def rec(ids, depth):
        if len(pieces) >= max_pieces:
            return
        ring = _neighbours(sk.F, ids, nvt)
        ring = ring[sk.dom[ring] == b] if depth > 99 else ring
        pts = sk.V0[ring]
        try:
            dev, _ = outward(pts)
        except Exception:  # degenerate (flat) cluster
            dev = 1.0
        if (dev <= tol or depth >= max_depth or len(ids) < 2 * min_pts):
            pieces.append({'verts_world_rest': pts, 'n_src': int(len(ids)), 'outward_rest_m': dev, 'depth': depth})
            if verbose:
                print(f'  {bone} piece {len(pieces)} n={len(ids)} depth={depth} outward={dev*1000:.1f} mm')
            return
        X = sk.V0[ids]
        # 2-means initialised along the main PCA axis
        c = X.mean(0)
        _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
        s = (X - c) @ Vt[0]
        lab = s > np.median(s)
        for _ in range(20):
            c0, c1 = X[~lab].mean(0), X[lab].mean(0)
            new = np.linalg.norm(X - c1, axis=1) < np.linalg.norm(X - c0, axis=1)
            if np.array_equal(new, lab) or new.all() or (~new).all():
                break
            lab = new
        if lab.all() or (~lab).all():
            pieces.append({'verts_world_rest': pts, 'n_src': int(len(ids)), 'outward_rest_m': dev, 'depth': depth})
            return
        rec(ids[~lab], depth + 1)
        rec(ids[lab], depth + 1)

    rec(vid, 0)
    out = []
    for k, p in enumerate(pieces):
        P, F, _ = hull_tris(p['verts_world_rest'])
        hv = P[np.unique(F)]
        out.append({'name': f'recipient_{bone}_p{k:02d}', 'verts_local': hv - rh, 'n_src': p['n_src'],
                    'outward_rest_m': p['outward_rest_m'], 'depth': p['depth']})
    return out


def _hull_eq(vl):
    h = ConvexHull(vl)
    return h.equations, vl[h.vertices], h.simplices, vl


def deviation_report(sk: SKN.Skin, pieces_by_bone, poses, h=0.001, region_min_weight=0.5):
    """Outward / inward deviation of the collider pieces from the posed skin at each (P, Q) pose."""
    from skin import q2R
    rng = np.random.default_rng(3)
    rep = {}
    hulls = {bone: [ConvexHull(p['verts_local']) for p in pcs] for bone, pcs in pieces_by_bone.items()}
    for pi, (P, Q) in enumerate(poses):
        V = sk.pose(P, Q)
        surf = sk.surface(V)
        for bone, pcs in pieces_by_bone.items():
            b = sk.idx[bone]
            R, p = q2R(Q[b]), P[b]
            # outward: hull surface samples in world
            worst_out = 0.0
            per_piece = []
            eqs = []
            for pc, hl in zip(pcs, hulls[bone]):
                Vw = pc['verts_local'] @ R.T + p
                S = sample_tris(Vw, hl.simplices, h, rng)
                sd, _, _ = surf.signed_distance(S)
                o = float(max(0.0, sd.max()))
                per_piece.append(round(o * 1000, 2))
                worst_out = max(worst_out, o)
                A = hl.equations[:, :3] @ R.T
                bb = hl.equations[:, 3] - A @ p
                eqs.append((A, bb, Vw, hl.simplices))
            # inward: skin surface samples of the bone's visible region not covered by any piece
            vid = np.flatnonzero(sk.dom == b)
            Fm = sk.F[np.isin(sk.F, vid).all(1)]
            S = sample_tris(V, Fm, 0.002, rng)
            inside_any = np.zeros(len(S), bool)
            for A, bb, Vw, simp in eqs:
                inside_any |= (S @ A.T + bb).max(1) <= 1e-9
            outside = np.zeros(len(S))
            ci = np.flatnonzero(~inside_any)
            if len(ci):
                from skin import _closest_on_triangles
                best = np.full(len(ci), np.inf)
                for A, bb, Vw, simp in eqs:
                    a_, b_, c_ = Vw[simp[:, 0]], Vw[simp[:, 1]], Vw[simp[:, 2]]
                    for j0 in range(0, len(ci), 256):
                        cj = ci[j0:j0 + 256]
                        pp = np.repeat(S[cj], len(simp), 0)
                        q, _ = _closest_on_triangles(pp, np.tile(a_, (len(cj), 1)), np.tile(b_, (len(cj), 1)),
                                                     np.tile(c_, (len(cj), 1)))
                        dd = np.linalg.norm(pp - q, axis=1).reshape(len(cj), len(simp)).min(1)
                        best[j0:j0 + 256] = np.minimum(best[j0:j0 + 256], dd)
                outside[ci] = best
            outside[~np.isfinite(outside)] = 0.05
            rep.setdefault(bone, []).append({'pose': pi, 'outward_max_mm': round(worst_out * 1000, 2),
                                             'inward_max_mm': round(float(outside.max()) * 1000, 2),
                                             'inward_p99_mm': round(float(np.quantile(outside, 0.99)) * 1000, 2),
                                             'outward_per_piece_mm': per_piece})
    return rep


def fit_bone_quadtree(sk: SKN.Skin, bone, tol=0.0025, min_cell=0.004, overlap=0.001, surf_rest=None,
                      max_pieces=400, verbose=False, meshes=('mesh_0',)):
    """Convex pieces by an adaptive quadtree in the plane of the two largest PCA axes of the bone's vertices: each
    cell's vertices (all depths along the third axis, cell grown by `overlap`) -> convex hull; a cell is split into 4
    while its hull bulges more than `tol` outside the rest-pose skin union (or until min_cell)."""
    b = sk.idx[bone]
    rh = sk.rest_head[b]
    keep = np.isin(sk.part, [i for i, mm in enumerate(sk.meshes) if meshes is None or mm['key'] in meshes])
    vid = np.flatnonzero((sk.dom == b) & keep)
    if surf_rest is None:
        surf_rest = sk.surface(sk.V0)
    # points = the bone's vertices + area samples (3 mm) of its triangles, so the pieces cover the triangle interiors
    m_ = np.zeros(len(sk.V0), bool)
    m_[vid] = True
    Fb = sk.F[m_[sk.F].all(1)]
    X = sample_tris(sk.V0, Fb, 0.003, np.random.default_rng(7)) if len(Fb) else sk.V0[vid]
    vid = np.arange(len(X))                          # indices into X from here on
    tree = cKDTree(X)
    c = X.mean(0)
    _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
    AB = (X - c) @ Vt[:2].T
    pieces = []
    stats = {'hulls_evaluated': 0}

    def dev_of(pts):
        P, F, _ = hull_tris(pts)
        S = sample_tris(P, F, 0.0015)
        sd, _, _ = surf_rest.signed_distance(S)
        stats['hulls_evaluated'] += 1
        return float(sd.max())

    def small_piece(sel, depth, lo, hi):
        # too few vertices for a solid hull: use the cell's vertices + their 1-ring neighbours (never drop skin)
        ids = vid[sel]
        if len(ids) == 0:
            return
        # the cell's points + the bone's points within a small radius (local; mesh edges can be several cm long,
        # so the mesh 1-ring is NOT local enough)
        dev = None
        for rad in (0.004, 0.006, 0.008, 0.012):
            near = tree.query_ball_point(X[ids], rad)
            jj = np.unique(np.concatenate([np.asarray(x, int) for x in near] + [np.zeros(0, int)]))
            pts = np.unique(np.concatenate([X[ids], X[jj]]), axis=0)
            if len(pts) < 4:
                continue
            try:
                dev = dev_of(pts)
                break
            except Exception:
                continue
        if dev is None:
            return
        pieces.append({'verts_world_rest': pts, 'n_src': int(len(ids)), 'outward_rest_m': dev, 'depth': depth,
                       'cell': [lo.tolist(), hi.tolist()], 'ring_patch': True})

    def rec(lo, hi, depth):
        sel = np.all((AB >= lo - overlap) & (AB <= hi + overlap), axis=1)
        if sel.sum() < 8:
            small_piece(sel, depth, lo, hi)
            return
        pts = X[sel]
        try:
            dev = dev_of(pts)
        except Exception:
            small_piece(sel, depth, lo, hi)
            return
        size = hi - lo
        if dev <= tol or size.max() <= min_cell or len(pieces) >= max_pieces:
            pieces.append({'verts_world_rest': pts, 'n_src': int(sel.sum()), 'outward_rest_m': dev, 'depth': depth,
                           'cell': [lo.tolist(), hi.tolist()]})
            return
        mid = 0.5 * (lo + hi)
        if size[0] > 2 * size[1]:
            cells = [(lo, np.array([mid[0], hi[1]])), (np.array([mid[0], lo[1]]), hi)]
        elif size[1] > 2 * size[0]:
            cells = [(lo, np.array([hi[0], mid[1]])), (np.array([lo[0], mid[1]]), hi)]
        else:
            cells = [(lo, mid), (np.array([mid[0], lo[1]]), np.array([hi[0], mid[1]])),
                     (np.array([lo[0], mid[1]]), np.array([mid[0], hi[1]])), (mid, hi)]
        for l2, h2 in cells:
            rec(l2, h2, depth + 1)

    rec(AB.min(0) - 1e-6, AB.max(0) + 1e-6, 0)
    # post-process: a piece that bulges more than hard_tol is dropped when every one of its own (cell) vertices is
    # already within 1 mm of / inside the other pieces; otherwise it is kept (reported)
    hard_tol = 0.003
    hulls = [ConvexHull(p['verts_world_rest']) for p in pieces]
    keep = np.ones(len(pieces), bool)
    for i in np.argsort([-p['outward_rest_m'] for p in pieces]):
        if pieces[i]['outward_rest_m'] <= hard_tol:
            continue
        own = pieces[i]['verts_world_rest']
        cov = np.zeros(len(own), bool)
        for j, h in enumerate(hulls):
            if j == i or not keep[j]:
                continue
            cov |= (own @ h.equations[:, :3].T + h.equations[:, 3]).max(1) <= 0.001
        if cov.all():
            keep[i] = False
    stats['dropped_bulging_pieces'] = int((~keep).sum())
    pieces = [p for p, k in zip(pieces, keep) if k]
    out = []
    for k, p in enumerate(pieces):
        P, F, _ = hull_tris(p['verts_world_rest'])
        hv = P[np.unique(F)]
        out.append({'name': f'recipient_{bone}_p{k:02d}', 'verts_local': hv - rh, 'n_src': p['n_src'],
                    'outward_rest_m': p['outward_rest_m'], 'depth': p['depth']})
    if verbose:
        print(f'{bone}: {len(out)} pieces (dropped {stats["dropped_bulging_pieces"]}), hulls evaluated {stats["hulls_evaluated"]}, max outward '
              f'{max(p["outward_rest_m"] for p in out)*1000:.2f} mm')
    return out


def fit_points_quadtree(X, name_prefix, origin, surf_rest, tol=0.0022, min_cell=0.004, overlap=0.001, max_pieces=200,
                        dev_fn=None):
    """v6: the v5 adaptive-quadtree convex decomposition (fit_bone_quadtree) for an explicit rest point set X (n,3)
    (e.g. one rigid finger segment of the curl rig).  Pieces: [{'name', 'verts_local' (= hull vertices - origin),
    'n_src', 'outward_rest_m', 'depth'}]; outward deviation measured against surf_rest (rest skin surface), or by
    dev_fn(hull surface samples (n,3) in rest coordinates) -> max outward distance (e.g. the max over several curls)."""
    X = np.unique(np.round(np.asarray(X, float), 9), axis=0)
    if len(X) < 4:
        return []
    vid = np.arange(len(X))
    tree = cKDTree(X)
    c = X.mean(0)
    _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
    AB = (X - c) @ Vt[:2].T
    pieces = []

    def dev_of(pts):
        P, F, _ = hull_tris(pts)
        S = sample_tris(P, F, 0.0015)
        if dev_fn is not None:
            return float(dev_fn(S))
        sd, _, _ = surf_rest.signed_distance(S)
        return float(sd.max())

    def small_piece(sel, depth):
        ids = vid[sel]
        if len(ids) == 0:
            return
        dev = None
        for rad in (0.003, 0.004, 0.006, 0.008):
            near = tree.query_ball_point(X[ids], rad)
            jj = np.unique(np.concatenate([np.asarray(x, int) for x in near] + [np.zeros(0, int)]))
            pts = np.unique(np.concatenate([X[ids], X[jj]]), axis=0)
            if len(pts) < 4:
                continue
            try:
                dev = dev_of(pts)
                break
            except Exception:
                continue
        if dev is None:
            return
        pieces.append({'verts_world_rest': pts, 'n_src': int(len(ids)), 'outward_rest_m': dev, 'depth': depth})

    def rec(lo, hi, depth):
        sel = np.all((AB >= lo - overlap) & (AB <= hi + overlap), axis=1)
        if sel.sum() < 8:
            small_piece(sel, depth)
            return
        pts = X[sel]
        try:
            dev = dev_of(pts)
        except Exception:
            small_piece(sel, depth)
            return
        size = hi - lo
        if dev <= tol or size.max() <= min_cell or len(pieces) >= max_pieces:
            pieces.append({'verts_world_rest': pts, 'n_src': int(sel.sum()), 'outward_rest_m': dev, 'depth': depth})
            return
        mid = 0.5 * (lo + hi)
        if size[0] > 2 * size[1]:
            cells = [(lo, np.array([mid[0], hi[1]])), (np.array([mid[0], lo[1]]), hi)]
        elif size[1] > 2 * size[0]:
            cells = [(lo, np.array([hi[0], mid[1]])), (np.array([lo[0], mid[1]]), hi)]
        else:
            cells = [(lo, mid), (np.array([mid[0], lo[1]]), np.array([hi[0], mid[1]])),
                     (np.array([lo[0], mid[1]]), np.array([mid[0], hi[1]])), (mid, hi)]
        for l2, h2 in cells:
            rec(l2, h2, depth + 1)

    rec(AB.min(0) - 1e-6, AB.max(0) + 1e-6, 0)
    hard_tol = 0.003
    hulls = [ConvexHull(p['verts_world_rest']) for p in pieces]
    keep = np.ones(len(pieces), bool)
    for i in np.argsort([-p['outward_rest_m'] for p in pieces]):
        if pieces[i]['outward_rest_m'] <= hard_tol:
            continue
        own = pieces[i]['verts_world_rest']
        cov = np.zeros(len(own), bool)
        for j, h in enumerate(hulls):
            if j == i or not keep[j]:
                continue
            cov |= (own @ h.equations[:, :3].T + h.equations[:, 3]).max(1) <= 0.001
        if cov.all():
            keep[i] = False
    pieces = [p for p, k in zip(pieces, keep) if k]
    out = []
    for k, p in enumerate(pieces):
        P, F, _ = hull_tris(p['verts_world_rest'])
        hv = P[np.unique(F)]
        out.append({'name': f'{name_prefix}_p{k:02d}', 'verts_local': hv - origin, 'n_src': p['n_src'],
                    'outward_rest_m': p['outward_rest_m'], 'depth': p['depth']})
    return out
