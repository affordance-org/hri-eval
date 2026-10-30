"""Per-episode rows and the batch report (report.json, report.md) from trial folders scored by check_gates.py.

Scores: each is its checker gate (catalog.SCORES), except that no fall also needs the episode to end normally
(end_reason max_time or controller_done, not an exception or a divergence). Totals count a score only in a valid episode
(G0 invariants and G1 motor replay pass); an episode that is not valid counts as failed."""
import datetime
import json
import subprocess
from pathlib import Path

import mujoco as mj
import numpy as np

from catalog import SCORES


def min_person_distance(trial, stride=10):
    """Minimum distance [m] between any robot collider and any collider of the person, from trial.xml and the
    recorded trajectory, every `stride` physics steps (10 = 50 Hz). Negative = penetration."""
    m = mj.MjModel.from_xml_path(str(Path(trial) / 'trial.xml'))
    tr = np.load(Path(trial) / 'trajectory.npz')
    d = mj.MjData(m)
    pel = m.body('pelvis').id

    def robot(b):
        while b > 0 and b != pel:
            b = m.body_parentid[b]
        return b == pel
    coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    rg = np.array([g for g in coll if robot(int(m.geom_bodyid[g]))])
    pg = np.array([g for g in coll if m.body(int(m.geom_bodyid[g])).name.startswith('recipient_')])
    if not len(rg) or not len(pg):
        return None
    rr, rp = m.geom_rbound[rg], m.geom_rbound[pg]
    fromto = np.zeros(6)
    best = np.inf
    for i in range(0, len(tr['time']), stride):
        d.qpos[:] = tr['qpos'][i]
        d.mocap_pos[:] = tr['mocap_pos'][i]
        d.mocap_quat[:] = tr['mocap_quat'][i]
        mj.mj_kinematics(m, d)
        lb = (np.linalg.norm(d.geom_xpos[rg][:, None] - d.geom_xpos[pg][None], axis=2) - rr[:, None] - rp[None]).ravel()
        cand = np.flatnonzero(lb < best)
        for k in cand[np.argsort(lb[cand])]:
            if lb[k] >= best:
                break
            a, b = divmod(int(k), len(pg))
            best = min(best, mj.mj_geomDistance(m, d, int(rg[a]), int(pg[b]), min(best, 10.0), fromto))
    return float(best)


NORMAL_END = ('max_time', 'controller_done')


def episode_row(trial, seed, variation, gates, wall_s=None):
    trial = Path(trial)
    g = gates['gates']
    s = json.loads((trial / 'summary.json').read_text())
    g3, g5 = g['G3_pickup'], g['G5_handover']
    scores = {k: bool(g[gate]['pass']) for k, _, _, gate in SCORES}
    scores['no_fall'] = scores['no_fall'] and s['episode']['end_reason'] in NORMAL_END
    return {'episode': trial.name, 'seed': int(seed), 'variation': {k: variation[k] for k in variation if k != 'seed'},
            'valid': bool(g['G0_invariants']['pass'] and g['G1_motor_replay']['pass']),
            'scores': scores,
            't_pickup_s': round(g3['interval'][0], 3) if g3['pass'] and g3.get('interval') else None,
            't_handover_s': round(g5['t_handover_complete_s'], 3) if g5['pass'] else None,
            'min_robot_person_distance_m': None if (dm := min_person_distance(trial)) is None else round(dm, 4),
            'gates': {k: bool(v['pass']) for k, v in g.items()},
            'end_reason': s['episode']['end_reason'], 'failure': s.get('failure'),
            'sim_time_s': round(s['episode']['sim_time_s'], 3), 'wall_time_s': None if wall_s is None else round(wall_s, 1)}


def error_row(name, seed, variation, err):
    return {'episode': name, 'seed': int(seed), 'variation': {k: variation[k] for k in variation if k != 'seed'},
            'valid': False, 'scores': {k: False for k, _, _, _ in SCORES}, 't_pickup_s': None, 't_handover_s': None,
            'min_robot_person_distance_m': None, 'gates': {}, 'end_reason': 'error', 'failure': err,
            'sim_time_s': None, 'wall_time_s': None}


def totals(rows):
    """Pass counts per score over all episodes; a score counts only in a valid episode."""
    n = len(rows)
    out = {k: {'passed': sum(bool(r['valid'] and r['scores'][k]) for r in rows), 'episodes': n} for k, _, _, _ in SCORES}
    out = {'valid': {'passed': sum(r['valid'] for r in rows), 'episodes': n}} | out
    for v in out.values():
        v['rate'] = round(v['passed'] / n, 4) if n else None
    dists = [r['min_robot_person_distance_m'] for r in rows if r['min_robot_person_distance_m'] is not None]
    out['min_robot_person_distance_m'] = min(dists) if dists else None
    return out


def git_commit(root):
    try:
        return subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:  # noqa
        return None


CHECKED_PATHS = ['hri_eval', 'run.py', 'requirements.txt']


def git_local_changes(root):
    """Tracked files under hri_eval/, run.py and requirements.txt that differ from the commit (git status), or None
    outside a git checkout. Your own new files (a policy module, say) are not listed."""
    try:
        r = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no', '--'] + CHECKED_PATHS,
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        return [x[3:] for x in r.stdout.splitlines() if x.strip()]
    except Exception:  # noqa
        return None


def changes_text(ch):
    if ch is None:
        return 'unknown (not a git checkout)'
    return 'none' if not ch else f"{len(ch)} modified file{'s' if len(ch) != 1 else ''} ({', '.join(ch[:5])}{', ...' if len(ch) > 5 else ''})"


def fmt(x, spec):
    return '' if x is None else format(x, spec)


def var_cells(v):
    if v.get('nominal'):
        return ['nominal', 'nominal', '1.000']
    return [f"{v['robot_dx_m']:+.3f} {v['robot_dy_m']:+.3f} {v['robot_dyaw_deg']:+.1f}",
            f"{v['bag_dx_m']:+.3f} {v['bag_dy_m']:+.3f}", f"{v['person_speed']:.3f}"]


HEAD = ['episode', 'seed', 'robot dx dy [m] yaw [deg]', 'bag dx dy [m]', 'person speed', 'valid'] + \
       [n for _, n, _, _ in SCORES] + ['pickup [s]', 'handover [s]', 'min dist [m]', 'end', 'sim [s]']


def cells(r):
    pf = lambda b: 'PASS' if b else 'FAIL'  # noqa: E731
    return [r['episode'], str(r['seed'])] + var_cells(r['variation']) + [pf(r['valid'])] + \
        [pf(r['scores'][k]) for k, _, _, _ in SCORES] + \
        [fmt(r['t_pickup_s'], '.2f'), fmt(r['t_handover_s'], '.2f'), fmt(r['min_robot_person_distance_m'], '.3f'),
         r['end_reason'], fmt(r['sim_time_s'], '.2f')]


def total_cells(t):
    return [f"{t[k]['passed']}/{t[k]['episodes']}" for k in ['valid'] + [k for k, _, _, _ in SCORES]]


def write(out, meta, rows):
    out = Path(out)
    t = totals(rows)
    rep = dict(meta, created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               episodes=rows, totals=t)
    (out / 'report.json').write_text(json.dumps(rep, indent=1) + '\n')
    md = [f"# HRI Eval report: {meta['policy']}", '',
          f"Scenario `{meta['scenario']}`, policy `{meta['policy']}` ({meta['interface']} level), "
          f"{len(rows)} episode{'s' if len(rows) != 1 else ''} (seeds {meta['seeds'][0]} to {meta['seeds'][-1]}), "
          f"max time {meta['max_time_s']:g} s, repo commit {meta.get('commit') or 'unknown'} (local changes in "
          f"hri_eval/, run.py, requirements.txt: {changes_text(meta.get('local_changes'))}), MuJoCo {mj.__version__}.", '',
          '| ' + ' | '.join(HEAD) + ' |', '|' + '---|' * len(HEAD)]
    md += ['| ' + ' | '.join(cells(r)) + ' |' for r in rows]
    md += ['', '| total | ' + ' | '.join(['valid'] + [n for _, n, _, _ in SCORES]) + ' | min dist [m] |',
           '|' + '---|' * (len(SCORES) + 3), '| passed | ' + ' | '.join(total_cells(t)) +
           f" | {fmt(t['min_robot_person_distance_m'], '.3f')} |", '',
           'valid: the checker re-simulated the trial and its G0 invariants and G1 motor replay pass. Totals count a '
           'score only in a valid episode. no fall also needs the episode to end normally (end max_time or '
           'controller_done). pickup [s]: start of the lift that passed the pickup score. handover [s]: time from '
           'which the person alone holds the bag. min dist: closest any robot collider came to any collider of the '
           'person, sampled at 50 Hz. end: why the episode ended; sim [s]: simulated time. Seed 0 is the nominal '
           'recorded episode; other seeds move the robot start, the backpack start and the playback speed of the '
           'person (variation.json in each episode folder).', '']
    (out / 'report.md').write_text('\n'.join(md))
    return rep


def print_table(rows):
    t = totals(rows)
    data = [HEAD] + [cells(r) for r in rows]
    w = [max(len(x[i]) for x in data) for i in range(len(HEAD))]
    for x in data:
        print('  '.join(c.ljust(w[i]) for i, c in enumerate(x)))
    print('\npass rate  ' + '  '.join(f'{n} {c}' for n, c in zip(['valid'] + [n for _, n, _, _ in SCORES], total_cells(t))) +
          '  (scores count only in valid episodes)')
