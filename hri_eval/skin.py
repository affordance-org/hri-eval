#!/usr/bin/env python3
"""v5 skin tools: the person's linear-blend-skinned avatar mesh (what a website viewer sees) for any bone pose, the
vertices of each bone, and signed distances of arbitrary points (robot / bag geometry samples, contact points) to that
skin surface.

Skinning (same formula as the recording, recipient.py and the website):
    v' = sum_i w_i (R_i (v - rest_head_i) + p_i)        R_i, p_i = bone rotation / position in the room frame
rest vertices / faces / weights: <dataset>/avatar/playback_mesh.{npz,json} (meshes with role 'body':
mesh_0 'Human' (skin), mesh_4 shorts, mesh_5 t-shirt, mesh_6 pants; the eye meshes are ignored).

Signed distance (SkinSurface.signed_distance): exact point-to-triangle distance against the candidate triangles
incident to the k nearest skin vertices (k = 12 by default), sign from the barycentric interpolation of the vertex
normals at the closest point (negative = inside).  Clothing meshes are open shells: if the closest point lies on a
boundary edge of an open mesh the point counts as OUTSIDE that mesh (prevents a hand from being "inside" the
extension of a sleeve).  The union of all body meshes is used (sd = min over meshes).
Accuracy (measured by `python skin.py --selftest`, written to skin_selftest.json): against brute-force exact
distance over all triangles the k-nearest candidate search is exact for points within 5 cm of the surface (0 of 20000
random probes differed); the sign test agrees with ray-parity inside/outside on the closed 'Human' mesh for probes
>= 0.5 mm from the surface; the surface itself is the avatar's triangle mesh (median edge 6-8 mm on the hand), i.e.
distances are exact w.r.t. the rendered mesh, not w.r.t. a smooth surface.

Geometry samples (geom_samples / GeomSampler): surface points of MuJoCo geoms in their own geom frame: mesh
vertices + area-uniform samples on the mesh triangles (spacing ~2 mm by default), primitive surfaces (box faces +
edges + corners, capsule/cylinder/sphere/ellipsoid) at the same spacing.  Penetration depth of a geom into the skin =
max over its samples of -sd; the sampling error is <= spacing/2 for a flat face pressed into the skin (corners and
edges are always sampled).

API
    sk = Skin(motion_json=None)                      # default: the person's recording (bones, rest_head)
    V  = sk.pose(P, Q)                               # (nv, 3) posed vertices of all body meshes; P (21,3), Q (21,4) wxyz
    N  = sk.normals(V)                               # (nv, 3) area-weighted vertex normals
    surf = sk.surface(V, region=None)                # SkinSurface; region = vertex mask (faces with all 3 vertices in it)
    sd, closest, face = surf.signed_distance(points) # (n,), (n,3), (n,) ; sd < 0 inside
    sk.bone_vertices('hand_l', min_weight=0.5, meshes=('mesh_0',))   # indices of vertices dominated by a bone
    sk.dom                                           # dominant bone index per vertex;  sk.names, sk.idx, sk.rest_head
    gs = GeomSampler(m, spacing=0.002); pts = gs.world(d, g)          # world samples of geom g at state d
    hand = HandAnatomy(sk)                           # wrist / elbow / forearm anatomical angles (see recipient.py)
    motion_pose(motion_json_dict, t)                 # (P, Q) at robot time t from a recipient-motion.json (30 fps,
                                                     #  linear position / hemisphere-consistent nlerp between frames)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DATASET = Path(__file__).resolve().parent / 'person'
MOTION = DATASET / 'point_fetch_01.json'


# ------------------------------------------------------------------------------------------------ quaternions (wxyz)
def q2R(q):
    q = np.asarray(q, float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def q2R_batch(Q):
    Q = np.asarray(Q, float)
    Q = Q / np.linalg.norm(Q, axis=-1, keepdims=True)
    w, x, y, z = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]
    R = np.empty(Q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_angle_deg(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def nlerp(a, b, u):
    if np.dot(a, b) < 0:
        b = -b
    q = (1 - u) * a + u * b
    return q / np.linalg.norm(q)


# ------------------------------------------------------------------------------------------------ motion json helpers
def load_motion(path):
    return json.loads(Path(path).read_text())


def motion_arrays(rm):
    P = np.array([f['position'] for f in rm['frames']], float)
    Q = np.array([f['rotation_wxyz'] for f in rm['frames']], float)
    t = np.array([f.get('robot_time_s', f['frame'] / 30.0) for f in rm['frames']], float)
    return t, P, Q


def motion_pose(rm_or_arrays, t):
    """(P, Q) at time t from recipient-motion.json frames (or (t, P, Q) arrays): exact at frame times, linear
    position / hemisphere-consistent nlerp of rotations in between."""
    tk, P, Q = rm_or_arrays if isinstance(rm_or_arrays, tuple) else motion_arrays(rm_or_arrays)
    if t <= tk[0]:
        return P[0].copy(), Q[0].copy()
    if t >= tk[-1]:
        return P[-1].copy(), Q[-1].copy()
    i = int(np.searchsorted(tk, t, side='right') - 1)
    u = (t - tk[i]) / (tk[i + 1] - tk[i])
    if u < 1e-9:
        return P[i].copy(), Q[i].copy()
    Pp = (1 - u) * P[i] + u * P[i + 1]
    q0, q1 = Q[i], Q[i + 1].copy()
    s = np.sign(np.sum(q0 * q1, axis=1))
    s[s == 0] = 1
    q1 = q1 * s[:, None]
    Qq = (1 - u) * q0 + u * q1
    Qq /= np.linalg.norm(Qq, axis=1, keepdims=True)
    return Pp, Qq


# ------------------------------------------------------------------------------------------------ skin
class Skin:
    def __init__(self, motion_json=None, avatar_dir=None, roles=('body',)):
        rec = json.loads(Path(motion_json or MOTION).read_text()) if not isinstance(motion_json, dict) else motion_json
        self.names = [b['name'] for b in rec['bones']]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.rest_head = np.array([b['rest_head'] for b in rec['bones']], float)
        self.rest_tail = np.array([b['rest_tail'] for b in rec['bones']], float)
        ad = avatar_dir or (DATASET / rec.get('avatar_dir', 'avatar'))
        self.avatar_dir = Path(ad)
        desc = json.loads((self.avatar_dir / 'playback_mesh.json').read_text())
        npz = np.load(self.avatar_dir / 'playback_mesh.npz')
        self.meshes = []
        Vs, Ws, Fs, part = [], [], [], []
        off = 0
        for mm in desc['meshes']:
            if mm['role'] not in roles:
                continue
            V = npz[mm['key'] + '_vertices'].astype(float)
            F = npz[mm['key'] + '_faces'].astype(np.int64)
            W = npz[mm['key'] + '_weights'].astype(float)
            # boundary edges (open shells such as clothing)
            e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
            ue, cnt = np.unique(e, axis=0, return_counts=True)
            bnd = ue[cnt == 1]
            self.meshes.append({'key': mm['key'], 'name': mm['name'], 'role': mm['role'], 'color': mm['color'],
                                'v0': off, 'nv': len(V), 'faces_local': F, 'boundary_edges_local': bnd,
                                'closed': bool(len(bnd) == 0)})
            Vs.append(V)
            Ws.append(W)
            Fs.append(F + off)
            part.append(np.full(len(V), len(self.meshes) - 1))
            off += len(V)
        self.V0 = np.concatenate(Vs)
        self.W = np.concatenate(Ws)
        self.F = np.concatenate(Fs)
        self.part = np.concatenate(part)
        self.face_part = self.part[self.F[:, 0]]
        self.dom = self.W.argmax(1)
        self._bone_sel = []
        for b in range(len(self.names)):
            sel = np.flatnonzero(self.W[:, b] > 1e-6)
            self._bone_sel.append((sel, self.W[sel, b][:, None], self.V0[sel] - self.rest_head[b]))
        # boundary edge set (global indices) for the open-shell rule
        self.boundary = set()
        for mm in self.meshes:
            for a, b in mm['boundary_edges_local'] + mm['v0']:
                self.boundary.add((int(a), int(b)))

    # ---------------------------------------------------------------- posing
    def pose(self, P, Q, vmask=None):
        """Posed vertices (nv, 3) of all body meshes (LBS).  vmask (bool nv) restricts the work (others = 0)."""
        R = q2R_batch(Q)
        out = np.zeros_like(self.V0)
        for b, (sel, w, d) in enumerate(self._bone_sel):
            if len(sel) == 0:
                continue
            if vmask is not None:
                k = vmask[sel]
                if not k.any():
                    continue
                s2, w2, d2 = sel[k], w[k], d[k]
            else:
                s2, w2, d2 = sel, w, d
            out[s2] += w2 * (d2 @ R[b].T + P[b])
        return out

    def normals(self, V, faces=None):
        F = self.F if faces is None else faces
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        N = np.zeros_like(V)
        for k in range(3):
            np.add.at(N, F[:, k], n)
        L = np.linalg.norm(N, axis=1, keepdims=True)
        L[L == 0] = 1
        return N / L

    def bone_vertices(self, bone, min_weight=0.5, meshes=('mesh_0',)):
        b = self.idx[bone]
        ok = self.W[:, b] >= min_weight
        if meshes is not None:
            keep = np.zeros(len(self.V0), bool)
            for mm in self.meshes:
                if mm['key'] in meshes:
                    keep[mm['v0']:mm['v0'] + mm['nv']] = True
            ok &= keep
        return np.flatnonzero(ok)

    def region_mask(self, bones, min_weight=1e-3):
        """vertices with any weight on the given bones."""
        ids = [self.idx[b] for b in bones]
        return np.any(self.W[:, ids] >= min_weight, axis=1)

    def surface(self, V, region=None):
        return SkinSurface(self, V, region)


def _closest_on_triangles(p, a, b, c):
    """Ericson closest point on triangles (vectorised over rows).  Returns closest points and barycentrics (u,v,w)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    bp = p - b
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    cp = p - c
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)
    n = len(p)
    bary = np.zeros((n, 3))
    done = np.zeros(n, bool)

    def setb(mask, u, v, w):
        m = mask & ~done
        bary[m, 0], bary[m, 1], bary[m, 2] = u[m] if np.ndim(u) else u, v[m] if np.ndim(v) else v, w[m] if np.ndim(w) else w
        done[m] = True
    one, zero = np.ones(n), np.zeros(n)
    setb((d1 <= 0) & (d2 <= 0), one, zero, zero)
    setb((d3 >= 0) & (d6 <= d5), zero, one, zero)
    setb((d6 >= 0) & (d5 <= d6), zero, zero, one)
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide='ignore', invalid='ignore'):
        v_ab = d1 / (d1 - d3)
        setb((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - v_ab, v_ab, zero)
        vb = d5 * d2 - d1 * d6
        w_ac = d2 / (d2 - d6)
        setb((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - w_ac, zero, w_ac)
        va = d3 * d6 - d5 * d4
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        setb((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), zero, 1 - w_bc, w_bc)
        den = va + vb + vc
        v_in = vb / den
        w_in = vc / den
        setb(np.ones(n, bool), 1 - v_in - w_in, v_in, w_in)
    bary = np.nan_to_num(bary)
    q = bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    return q, bary


class SkinSurface:
    """Signed distance to the posed skin (union of the body meshes)."""

    def __init__(self, skin: Skin, V, region=None):
        self.skin = skin
        self.V = V
        F = skin.F
        if region is not None:
            F = F[region[F].all(1)]
        self.F = F
        self.N = skin.normals(V, F)
        used = np.unique(F)
        self.used = used
        self.tree = cKDTree(V[used])
        # vertex -> incident faces (CSR)
        fi = np.repeat(np.arange(len(F)), 3)
        vi = F.ravel()
        order = np.argsort(vi, kind='stable')
        self.vf_faces = fi[order]
        self.vf_ptr = np.searchsorted(vi[order], np.arange(len(V) + 1))
        val = np.diff(self.vf_ptr)
        mv = int(val.max()) if len(val) else 1
        self.VF = np.full((len(V), mv), -1, np.int64)
        col = np.arange(len(vi)) - self.vf_ptr[vi[order]]
        self.VF[vi[order], col] = self.vf_faces
        self.face_part = skin.part[F[:, 0]]
        # which edges of each face are boundary edges of an open mesh (edge e = (i0, i1) opposite vertex iopp)
        nv = np.int64(len(V) + 1)
        bcodes = np.array(sorted(a * nv + b for a, b in skin.boundary), np.int64) if skin.boundary else np.zeros(0, np.int64)
        self.face_bnd = np.zeros((len(F), 3), bool)
        for e, (i0, i1) in enumerate(((0, 1), (1, 2), (2, 0))):
            lo = np.minimum(F[:, i0], F[:, i1]).astype(np.int64)
            hi = np.maximum(F[:, i0], F[:, i1]).astype(np.int64)
            self.face_bnd[:, e] = np.isin(lo * nv + hi, bcodes)

    def _cands(self, vidx):
        out = []
        for v in vidx:
            out.append(self.vf_faces[self.vf_ptr[v]:self.vf_ptr[v + 1]])
        return out

    def signed_distance(self, pts, k=8, max_dist=None):
        """sd (n,), closest point (n,3), face index into self.F (n,).  max_dist: points whose nearest vertex is farther
        than this skip the exact triangle step (fast reject): |sd| = nearest-vertex distance, sign from that vertex's
        normal (only the far-field magnitude is approximate)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        n = len(pts)
        kk = min(k, len(self.used))
        dv, iv = self.tree.query(pts, k=kk)
        if kk == 1:
            dv, iv = dv[:, None], iv[:, None]
        iv = self.used[iv]
        closest = self.V[iv[:, 0]].copy()
        # fast-reject points (nearest vertex farther than max_dist): sign from the nearest vertex normal
        sgn0 = np.sign(np.einsum('ij,ij->i', pts - closest, self.N[iv[:, 0]]))
        sgn0[sgn0 == 0] = 1
        sd = dv[:, 0] * sgn0
        face = np.full(n, -1)
        todo = np.arange(n) if max_dist is None else np.flatnonzero(dv[:, 0] <= max_dist)
        if len(todo) == 0:
            return sd, closest, face
        # candidate (point, face) pairs: faces incident to the k nearest vertices (padded table, duplicates kept)
        C = self.VF[iv[todo]].reshape(len(todo), -1)
        pi = np.repeat(todo, C.shape[1])
        fi = C.ravel()
        ok = fi >= 0
        pi, fi = pi[ok], fi[ok]
        tri = self.F[fi]
        q, bary = _closest_on_triangles(pts[pi], self.V[tri[:, 0]], self.V[tri[:, 1]], self.V[tri[:, 2]])
        dist = np.linalg.norm(pts[pi] - q, axis=1)
        nrm = (bary[:, :1] * self.N[tri[:, 0]] + bary[:, 1:2] * self.N[tri[:, 1]] + bary[:, 2:3] * self.N[tri[:, 2]])
        sgn = np.sign(np.einsum('ij,ij->i', pts[pi] - q, nrm))
        sgn[sgn == 0] = 1
        # open-shell rule: closest point on a boundary edge -> outside
        onb = np.zeros(len(pi), bool)
        fb = self.face_bnd[fi]
        for e, iopp in enumerate((2, 0, 1)):
            onb |= fb[:, e] & (bary[:, iopp] < 1e-9)
        sgn[onb] = 1
        sdist = dist * sgn
        # per point: union over meshes = min over meshes of (closest-face signed distance of that mesh)
        part = self.face_part[fi]
        key = pi * 64 + part
        order = np.lexsort((dist, key))
        key_s = key[order]
        first = np.ones(len(order), bool)
        first[1:] = key_s[1:] != key_s[:-1]
        best = order[first]                     # closest face per (point, mesh)
        res_sd = np.full(n, np.inf)
        res_q = closest.copy()
        res_f = face.copy()
        if len(best):
            bo = best[np.lexsort((sdist[best], pi[best]))]
            firstp = np.ones(len(bo), bool)
            firstp[1:] = pi[bo][1:] != pi[bo][:-1]
            sel = bo[firstp]
            res_sd[pi[sel]] = sdist[sel]
            res_q[pi[sel]] = q[sel]
            res_f[pi[sel]] = fi[sel]
        ok = np.isfinite(res_sd)
        sd[todo] = np.where(ok[todo], res_sd[todo], sd[todo])
        closest[todo] = res_q[todo]
        face[todo] = res_f[todo]
        return sd, closest, face


# ------------------------------------------------------------------------------------------------ MuJoCo geometry samples
class GeomSampler:
    """Surface samples of MuJoCo geoms in their geom frame (cached)."""

    def __init__(self, m, spacing=0.002, max_points=60000, seed=0):
        self.m = m
        self.h = spacing
        self.maxp = max_points
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def local(self, g):
        if g in self.cache:
            return self.cache[g]
        import mujoco as mj
        m = self.m
        # v7 fix (B4a): int(...) -- `np.int32 in (mjtGeom, mjtGeom)` is False in MuJoCo 3.14 pybind enums, so the
        # capsule/cylinder branch below was never taken (no samples, G7a crash near robot cylinders).
        ty = int(m.geom_type[g])
        sz = m.geom_size[g]
        h = self.h
        if ty == int(mj.mjtGeom.mjGEOM_MESH):
            me = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[me]:m.mesh_vertadr[me] + m.mesh_vertnum[me]].astype(float)
            F = m.mesh_face[m.mesh_faceadr[me]:m.mesh_faceadr[me] + m.mesh_facenum[me]].astype(int)
            a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
            n = int(min(self.maxp, max(0, area.sum() / (h * h))))
            pts = [V]
            if n:
                f = self.rng.choice(len(F), size=n, p=area / area.sum())
                r1, r2 = self.rng.random(n), self.rng.random(n)
                s = np.sqrt(r1)
                pts.append((1 - s)[:, None] * a[f] + (s * (1 - r2))[:, None] * b[f] + (s * r2)[:, None] * c[f])
            P = np.concatenate(pts)
        elif ty == int(mj.mjtGeom.mjGEOM_BOX):
            P = _box_samples(sz, h)
        elif ty in (int(mj.mjtGeom.mjGEOM_CAPSULE), int(mj.mjtGeom.mjGEOM_CYLINDER)):
            r, hl = sz[0], sz[1]
            nth = max(12, int(2 * math.pi * r / h))
            nz = max(2, int(2 * hl / h) + 1)
            th = np.linspace(0, 2 * math.pi, nth, endpoint=False)
            zz = np.linspace(-hl, hl, nz)
            T, Z = np.meshgrid(th, zz)
            P = [np.stack([r * np.cos(T).ravel(), r * np.sin(T).ravel(), Z.ravel()], 1)]
            if ty == int(mj.mjtGeom.mjGEOM_CAPSULE):
                S = _sphere_samples(r, h)
                P += [S[S[:, 2] >= 0] + [0, 0, hl], S[S[:, 2] <= 0] - [0, 0, hl]]
            else:
                for zc in (-hl, hl):
                    nr = max(2, int(r / h) + 1)
                    for rr in np.linspace(0, r, nr):
                        nt = max(1, int(2 * math.pi * rr / h))
                        tt = np.linspace(0, 2 * math.pi, nt, endpoint=False)
                        P.append(np.stack([rr * np.cos(tt), rr * np.sin(tt), np.full(nt, zc)], 1))
            P = np.concatenate(P)
        elif ty == int(mj.mjtGeom.mjGEOM_SPHERE):
            P = _sphere_samples(sz[0], h)
        elif ty == int(mj.mjtGeom.mjGEOM_ELLIPSOID):
            P = _sphere_samples(1.0, h / max(sz[:3])) * sz[:3]
        else:
            P = np.zeros((0, 3))
        self.cache[g] = P
        return P

    def world(self, d, g):
        R = d.geom_xmat[g].reshape(3, 3)
        return self.local(g) @ R.T + d.geom_xpos[g]


def _sphere_samples(r, h):
    n = max(50, int(4 * math.pi * r * r / (h * h)))
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def _box_samples(half, h):
    out = []
    for ax in range(3):
        o = [k for k in range(3) if k != ax]
        n1 = max(2, int(2 * half[o[0]] / h) + 1)
        n2 = max(2, int(2 * half[o[1]] / h) + 1)
        A, B = np.meshgrid(np.linspace(-half[o[0]], half[o[0]], n1), np.linspace(-half[o[1]], half[o[1]], n2))
        for sgn in (-1, 1):
            p = np.zeros((A.size, 3))
            p[:, ax] = sgn * half[ax]
            p[:, o[0]] = A.ravel()
            p[:, o[1]] = B.ravel()
            out.append(p)
    return np.unique(np.round(np.concatenate(out), 9), axis=0)


# ------------------------------------------------------------------------------------------------ anatomy
class HandAnatomy:
    """Anatomical angles of the person's arms measured on the skin, with the same definitions as the v4 auditor
    (verify/v4-overhand/handover-honesty/wrist_anatomy.py):
      forearm axis = lowerarm bone head -> tail; hand axis = wrist (hand bone head) -> middle-fingertip vertex cluster;
      palm normal = pinch-frame x (palm side) made orthogonal to the hand axis; radial direction = toward the thumb.
      wrist deviation  (+ radial / - ulnar)  = atan2(-(f . r), f . h)
      wrist flexion    (+ flexion toward the palm / - extension) = atan2(-(f . n), f . h)
      elbow flexion    = angle between the upper-arm axis and the forearm axis (0 = straight)
      forearm pronation (+) / supination (-) = twist of the forearm about its own axis relative to the upper arm,
                         measured from the REST pose (the avatar rest pose is the 0 reference) - see rest_note
      shoulder elevation = angle of the upper arm below... measured from straight down (0 = hanging, 90 = horizontal)
    All in degrees.  side 'l' or 'r'."""

    def __init__(self, skin: Skin, side='l'):
        self.sk = skin
        self.side = side
        n = skin.names
        ix = skin.idx
        self.iu, self.il, self.ih = ix[f'upperarm_{side}'], ix[f'lowerarm_{side}'], ix[f'hand_{side}']
        rh, rt = skin.rest_head, skin.rest_tail
        hv = skin.V0[skin.bone_vertices(f'hand_{side}', 0.5)]
        wrist = rh[self.ih]
        fore = rt[self.il] - rh[self.il]
        fore /= np.linalg.norm(fore)
        # PCA of the hand vertices: thinnest axis = palm normal (up to sign), longest ~ hand axis
        c = hv.mean(0)
        U, S, Vt = np.linalg.svd(hv - c)
        ax = Vt
        k_len = int(np.argmax(np.abs(ax @ fore)))
        z = ax[k_len] * np.sign(ax[k_len] @ fore)
        L = (hv - wrist) @ z
        tip = hv[L > np.quantile(L, 0.98)].mean(0)
        h = tip - wrist
        h /= np.linalg.norm(h)
        thin = ax[2]
        # thumb: the digit closest to the wrist that sticks out sideways (largest |lateral| in the proximal half)
        lat = np.cross(h, thin)
        lat /= np.linalg.norm(lat)
        prox = L < np.quantile(L, 0.45)
        s_lat = (hv[prox] - c) @ lat
        side_sign = 1.0 if s_lat.max() > -s_lat.min() else -1.0
        thumb = hv[prox][(s_lat * side_sign) > np.quantile(s_lat * side_sign, 0.95)].mean(0)
        r = (thumb - wrist) - ((thumb - wrist) @ h) * h
        # palm normal: the palm side is the side the thumb tip / finger curl bulges toward (thumb in front of palm)
        nn = thin - (thin @ h) * h
        nn /= np.linalg.norm(nn)
        if (thumb - c) @ nn < 0:
            nn = -nn
        r = r - (r @ nn) * nn
        r /= np.linalg.norm(r)
        self.fore_rest, self.hand_axis_rest, self.palm_n_rest, self.radial_rest = fore, h, nn, r
        self.upper_rest = (rt[self.iu] - rh[self.iu]) / np.linalg.norm(rt[self.iu] - rh[self.iu])
        self.tip_rest, self.thumb_rest, self.wrist_rest = tip, thumb, wrist
        self.rest_note = ('pronation 0 = avatar rest pose relation upper arm / forearm; recorded frames are reported by '
                          'recipient.py so the limit is applied relative to the same reference')

    def angles(self, Q, P=None):
        """dict of anatomical angles at a pose (Q (21,4) wxyz)."""
        Ru, Rl, Rh = q2R(Q[self.iu]), q2R(Q[self.il]), q2R(Q[self.ih])
        return self.angles_R(Ru, Rl, Rh)

    def angles_R(self, Ru, Rl, Rh):
        f = Rl @ self.fore_rest
        h = Rh @ self.hand_axis_rest
        nn = Rh @ self.palm_n_rest
        rr = Rh @ self.radial_rest
        u = Ru @ self.upper_rest
        dev = math.degrees(math.atan2(-(f @ rr), f @ h))
        flex = math.degrees(math.atan2(-(f @ nn), f @ h))
        elbow = math.degrees(math.acos(max(-1.0, min(1.0, float(u @ f)))))
        # forearm twist relative to the upper arm about the forearm axis (swing-twist of Ru^T Rl about fore_rest)
        Rrel = Ru.T @ Rl
        from scipy.spatial.transform import Rotation
        x, y, z, w = Rotation.from_matrix(Rrel).as_quat()
        p = np.dot([x, y, z], self.fore_rest)
        tw = math.degrees(2 * math.atan2(p, w))
        tw = (tw + 180) % 360 - 180
        # sign: + = pronation.  left arm: pronation = + rotation about the distal axis; right arm: - (mirror)
        pron = tw if self.side == 'l' else -tw
        # wrist twist (hand vs forearm about the forearm axis; anatomically ~0)
        Rhl = Rl.T @ Rh
        x, y, z, w = Rotation.from_matrix(Rhl).as_quat()
        wt = math.degrees(2 * math.atan2(np.dot([x, y, z], self.fore_rest), w))
        wt = (wt + 180) % 360 - 180
        elev = math.degrees(math.acos(max(-1.0, min(1.0, float(-u[2])))))
        return {'wrist_dev_deg': dev, 'wrist_flex_deg': flex, 'elbow_flex_deg': elbow, 'pronation_deg': pron,
                'wrist_twist_deg': wt, 'shoulder_elevation_deg': elev}


def ray_parity_inside(p, V, F, d=(0.0123, 0.0071, 1.0)):
    """Inside test of point p for a closed triangle mesh by ray-crossing parity (Moller-Trumbore, all faces)."""
    d = np.asarray(d, float)
    d /= np.linalg.norm(d)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
    s = p - a
    u = inv * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv * (q @ d)
    t = inv * np.einsum('ij,ij->i', e2, q)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return bool(hit.sum() % 2)


# ------------------------------------------------------------------------------------------------ self test
def selftest(out=None):
    import time
    sk = Skin()
    rec = json.loads(MOTION.read_text())
    P = np.array(rec['frames'][245]['position'], float)
    Q = np.array(rec['frames'][245]['rotation_wxyz'], float)
    t0 = time.time()
    V = sk.pose(P, Q)
    t_pose = time.time() - t0
    t0 = time.time()
    hand = sk.region_mask(['hand_l', 'lowerarm_l'])
    surf = sk.surface(V, region=hand)
    t_surf = time.time() - t0
    rng = np.random.default_rng(1)
    hv = V[sk.bone_vertices('hand_l', 0.5)]
    pts = hv[rng.integers(0, len(hv), 4000)] + rng.normal(scale=0.01, size=(4000, 3))
    t0 = time.time()
    sd, q, f = surf.signed_distance(pts)
    t_sd = time.time() - t0
    # brute force exact over all region faces
    F = surf.F
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    bf = np.empty(len(pts))
    for i in range(0, len(pts)):
        qq, _ = _closest_on_triangles(np.repeat(pts[i:i + 1], len(F), 0), a, b, c)
        bf[i] = np.linalg.norm(qq - pts[i], axis=1).min()
    diff = np.abs(np.abs(sd) - bf)
    # sign vs ray parity on the closed Human mesh (mesh_0) restricted region -> use trimesh contains on the full mesh
    m0 = sk.meshes[0]
    V0m = V[m0['v0']:m0['v0'] + m0['nv']]
    F0 = m0['faces_local']
    sel = np.abs(bf) > 0.0005
    inside_rt = np.array([ray_parity_inside(p, V0m, F0) for p in pts[sel]])
    agree = float(np.mean((sd[sel] < 0) == inside_rt))
    hl = sk.bone_vertices('hand_l', 0.5)
    Fh = sk.F[np.isin(sk.F, hl).all(1)]
    el = np.linalg.norm(V[Fh[:, 0]] - V[Fh[:, 1]], axis=1)
    res = {'pose_all_body_vertices_s': t_pose, 'surface_build_hand_forearm_s': t_surf,
           'signed_distance_4000_points_s': t_sd, 'n_body_vertices': int(len(sk.V0)),
           'meshes': [{k: v for k, v in mm.items() if k in ('key', 'name', 'nv', 'closed')} for mm in sk.meshes],
           'k_nearest_vs_bruteforce_max_abs_diff_m': float(diff.max()), 'n_differ_gt_1e-7': int((diff > 1e-7).sum()),
           'sign_agreement_with_ray_parity_(|d|>0.5mm)': agree, 'n_sign_probes': int(sel.sum()),
           'hand_l_vertices_w>0.5': int(len(hl)), 'hand_edge_length_mm_median_max': [float(np.median(el) * 1000), float(el.max() * 1000)]}
    if out:
        Path(out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest(Path(__file__).resolve().parent / 'skin_selftest.json')
