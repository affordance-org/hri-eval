"""Damped least-squares differential IK as a small QP (OSQP) over the 14 arm joints.

Tasks: per-side tool position (+ optional full orientation or only the approach axis), posture
regularisation, joint limits (hard box), per-iteration step bound, and collision-avoidance inequalities
from MuJoCo geometry distances (mj_geomDistance) between arm/UMI geoms and the torso, pelvis, legs, head,
the other arm, the table, the resting bag, and the carried-bag ghost. Purely kinematic.
"""
import mujoco as mj
import numpy as np
import osqp
import scipy.sparse as sp

ARM_JOINTS = [f"{s}_{j}_joint" for s in ["left", "right"] for j in
              ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]]


def rot_err(R_target, R_cur):
    """World-frame rotation vector taking R_cur to R_target."""
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R_target @ R_cur.T).as_rotvec()


class CollisionSet:
    """Geom pairs to keep apart, with per-pair safety distance."""

    def __init__(self, kin):
        m = kin.m
        self.kin = kin
        bname = lambda g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        gname = kin.geom_name
        coll = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])]
        ghost = [g for g in range(m.ngeom) if bname(g) == 'bag_carried']
        self.groups = {}
        def sel(pred, pool=coll):
            return [g for g in pool if pred(bname(g), gname(g))]
        for s, o in [('left', 'right'), ('right', 'left')]:
            self.groups[f'{s}_hand'] = sel(lambda b, n: b.startswith(f'{s}_umi') or b in (f'{s}_wrist_pitch_link', f'{s}_wrist_yaw_link', f'{s}_wrist_roll_link'))
            self.groups[f'{s}_forearm'] = sel(lambda b, n: b in (f'{s}_elbow_link',))
        self.groups['torso'] = sel(lambda b, n: b in ('torso_link', 'head_link', 'logo_link', 'pelvis_contour_link'))
        self.groups['legs'] = sel(lambda b, n: ('hip_' in b) or ('knee' in b) or ('ankle' in b))
        self.groups['table'] = sel(lambda b, n: n.startswith('table_'))
        self.groups['bag_rest'] = sel(lambda b, n: b == 'bag_rest')
        self.groups['bag_rest_body'] = sel(lambda b, n: b == 'bag_rest' and n != 'rest_bag_1_15')
        self.groups['bag_carried'] = ghost
        self.groups['fingers_left'] = sel(lambda b, n: b.startswith('left_umi_') and 'finger' in b)

    def pairs(self, spec):
        """spec: list of (groupA, groupB, d_safe). groupA must be on an arm."""
        out = []
        for a, b, ds in spec:
            for ga in self.groups[a]:
                for gb in self.groups[b]:
                    out.append((ga, gb, ds))
        return out


def geom_distance(m, d, g1, g2, distmax):
    fromto = np.zeros(6)
    dist = mj.mj_geomDistance(m, d, g1, g2, distmax, fromto)
    return dist, fromto


class ArmIK:
    def __init__(self, kin, collisions):
        self.kin = kin
        self.col = collisions
        self.n = 14
        # priority: the left (load-carrying) arm never yields to the right arm or to the bag it carries;
        # rows between a right-arm geom and a left-arm/ghost geom only move the right arm.
        m = kin.m
        bname = lambda b: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) or ''
        self.left_side = set(b for b in range(m.nbody) if bname(b).startswith('left_') and
                             any(t in bname(b) for t in ('shoulder', 'elbow', 'wrist', 'umi')) or bname(b) == 'bag_carried')
        self.right_side = set(b for b in range(m.nbody) if bname(b).startswith('right_') and
                              any(t in bname(b) for t in ('shoulder', 'elbow', 'wrist', 'umi')))

    def solve(self, q0, targets, q_nom, pair_list, iters=30, w_post=1e-5, lam=1e-6, step=0.08,
              d_influence=0.10, gain=0.5, tol=1e-4, lo=None, hi=None, fixed=None, max_push=0.01):
        """targets: dict side -> dict(pos, R (optional), axis (optional, desired tool z), wp, wr).
        Returns q, info. Base/legs/waist/fingers must be set on kin.d before the call."""
        k, m, d = self.kin, self.kin.m, self.kin.d
        q = q0.copy()
        lo = k.arm_lo + 0.02 if lo is None else lo
        hi = k.arm_hi - 0.02 if hi is None else hi
        q = np.clip(q, lo, hi)
        mask = np.ones(14, bool)
        if fixed is not None:
            mask[fixed] = False
        info = {}
        prob_cache = None
        for it in range(iters):
            k.set_arm(q); k.fk()
            H = lam * np.eye(14) + w_post * np.eye(14)
            g = w_post * (q - q_nom)
            cost = 0.0
            errs = {}
            for side, t in targets.items():
                p, Rm = k.tool(side)
                Jp, Jr = k.jac_tool(side)
                e = t['pos'] - p
                Wp = t.get('Wp', t.get('wp', 1.0) * np.eye(3))
                H += Jp.T @ Wp @ Jp; g -= Jp.T @ Wp @ e
                cost += e @ Wp @ e
                er = np.zeros(3)
                if t.get('R') is not None:
                    er = rot_err(t['R'], Rm)
                    wr = t.get('wr', 0.1)
                    H += wr * Jr.T @ Jr; g -= wr * Jr.T @ er
                    cost += wr * er @ er
                elif t.get('axis') is not None:
                    z = Rm[:, 2]; a = t['axis'] / np.linalg.norm(t['axis'])
                    er = np.cross(z, a)  # small-angle rotation taking z to a
                    wr = t.get('wr', 0.1)
                    # only the two components orthogonal to a matter
                    P = np.eye(3) - np.outer(a, a)
                    H += wr * Jr.T @ P @ Jr; g -= wr * Jr.T @ P @ er
                    cost += wr * er @ er
                errs[side] = (np.linalg.norm(e), np.linalg.norm(er))
            # collision rows
            rows, lb_rows = [], []
            dmin = {}
            for ga, gb, ds in pair_list:
                dist, ft = geom_distance(m, d, ga, gb, d_influence + ds)
                key = (ga, gb)
                dmin[key] = dist
                if dist < d_influence + ds and dist < 10:
                    pa, pb = ft[:3], ft[3:]
                    nvec = pb - pa
                    nn = np.linalg.norm(nvec)
                    if nn < 1e-9:
                        continue
                    nvec = nvec / nn
                    if dist < 0:
                        nvec = -nvec
                    ba, bb = m.geom_bodyid[ga], m.geom_bodyid[gb]
                    Ja = k.jac_point(ba, pa)
                    Jb = k.jac_point(bb, pb) if bb != 0 else np.zeros((3, 14))
                    if ba in self.left_side and bb in self.right_side:
                        Ja = Ja.copy(); Ja[:, :7] = 0
                    if ba in self.right_side and bb in self.left_side:
                        Jb = Jb.copy(); Jb[:, :7] = 0
                    # d(dist)/dq = n . (Jb - Ja) ; require dist + ddist >= ds - (ds - dist)*(1-gain) ...
                    row = nvec @ (Jb - Ja)
                    rows.append(row)
                    lb_rows.append(min(gain * (ds - dist), max_push))
            dq_lo = np.maximum(lo - q, -step); dq_hi = np.minimum(hi - q, step)
            dq_lo[~mask] = 0; dq_hi[~mask] = 0
            A = np.eye(14); l = dq_lo; u = dq_hi
            if rows:
                A = np.vstack([A, np.array(rows)])
                l = np.concatenate([l, np.array(lb_rows)])
                u = np.concatenate([u, np.full(len(rows), np.inf)])
            # if a collision row is infeasible with the box, OSQP returns the closest; fine for IK
            prob = osqp.OSQP()
            prob.setup(sp.csc_matrix(np.triu(H)), g, sp.csc_matrix(A), l, u, verbose=False, eps_abs=1e-7, eps_rel=1e-7,
                       max_iter=4000, polish=False)
            res = prob.solve()
            ok = res.x is not None and np.all(np.isfinite(res.x)) and res.info.status_val in (1, 2)
            if not ok and rows:  # infeasible collision rows: fall back to the box-only problem this iteration
                prob = osqp.OSQP()
                prob.setup(sp.csc_matrix(np.triu(H)), g, sp.csc_matrix(np.eye(14)), dq_lo, dq_hi, verbose=False,
                           eps_abs=1e-7, eps_rel=1e-7, max_iter=4000, polish=False)
                res = prob.solve()
                ok = res.x is not None and np.all(np.isfinite(res.x))
                info['fallback'] = info.get('fallback', 0) + 1
            dq = res.x if ok else np.zeros(14)
            q = np.clip(q + dq, lo, hi)
            if np.max(np.abs(dq)) < tol:
                break
        k.set_arm(q); k.fk()
        final = {}
        for side, t in targets.items():
            p, Rm = k.tool(side)
            pe = float(np.linalg.norm(t['pos'] - p))
            if t.get('R') is not None:
                re = float(np.linalg.norm(rot_err(t['R'], Rm)))
            elif t.get('axis') is not None:
                a = t['axis'] / np.linalg.norm(t['axis'])
                re = float(np.arccos(np.clip(Rm[:, 2] @ a, -1, 1)))
            else:
                re = 0.0
            final[side] = (pe, re)
        info['err'] = final
        info['iters'] = it + 1
        return q, info

    def min_distances(self, pair_list, distmax=0.3):
        m, d = self.kin.m, self.kin.d
        out = []
        for ga, gb, ds in pair_list:
            dist, ft = geom_distance(m, d, ga, gb, distmax)
            out.append(dist)
        return np.array(out)
