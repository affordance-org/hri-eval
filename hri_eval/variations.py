"""Seeded episode variations.

Seed 0 is the nominal recorded episode. Any other seed draws, from numpy.random.default_rng(seed) in this order:
robot start dx, dy in +-0.10 m and yaw in +-10 deg, backpack dx, dy on the table in +-0.05 m, and the playback speed
of the recorded person in [0.9, 1.1]. The two start offsets are written into keyframe 'start' of trial.xml before the
model is loaded, so the runner and the independent checker (check_gates.py, which recomputes the variation from the
seed in variation.json) load the same initial state. The speed scales the person's whole clip, the idle ping-pong
included; the checker replays the recorded mocap poses either way."""
import math
import re

import mujoco as mj
import numpy as np

FPS = 30.0                      # recording frame rate (recipient5.FPS)
IDLE_PEAK_RATE = 0.45           # recipient5.Recipient5.idle default
RANGES = [('robot_dx_m', -0.10, 0.10), ('robot_dy_m', -0.10, 0.10), ('robot_dyaw_deg', -10.0, 10.0),
          ('bag_dx_m', -0.05, 0.05), ('bag_dy_m', -0.05, 0.05), ('person_speed', 0.9, 1.1)]
KEYS = [k for k, _, _ in RANGES]
NOMINAL = {'robot_dx_m': 0.0, 'robot_dy_m': 0.0, 'robot_dyaw_deg': 0.0, 'bag_dx_m': 0.0, 'bag_dy_m': 0.0,
           'person_speed': 1.0}
KEY_RE = re.compile(r'(<key\s+name="start"\s+qpos=")([^"]*)(")')


def sample(seed):
    seed = int(seed)
    if seed == 0:
        return dict(NOMINAL, seed=0, nominal=True)
    g = np.random.default_rng(seed)
    return dict({k: float(g.uniform(lo, hi)) for k, lo, hi in RANGES}, seed=seed, nominal=False)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def apply_key(qpos, m, v):
    """Keyframe qpos with the variation applied: pelvis moved by (dx, dy) in the world frame and turned by dyaw about
    the vertical axis, backpack moved by (dx, dy) on the table."""
    q = np.array(qpos, float)
    if v.get('nominal'):
        return q
    r = int(m.jnt_qposadr[m.joint('floating_base_joint').id])
    b = int(m.jnt_qposadr[m.joint('backpack_free').id])
    q[r] += v['robot_dx_m']
    q[r + 1] += v['robot_dy_m']
    h = math.radians(v['robot_dyaw_deg']) / 2
    q[r + 3:r + 7] = quat_mul([math.cos(h), 0.0, 0.0, math.sin(h)], q[r + 3:r + 7])
    q[b] += v['bag_dx_m']
    q[b + 1] += v['bag_dy_m']
    return q


def set_key_qpos(txt, q):
    assert len(KEY_RE.findall(txt)) == 1, 'expected exactly one <key name="start" qpos=...>'
    return KEY_RE.sub(lambda mo: mo.group(1) + ' '.join(repr(float(x)) for x in q) + mo.group(3), txt)


def xml_editor(v, src_xml):
    """EpisodeRunner(xml_edit=...) callable: rewrites keyframe 'start' of the trial.xml text."""
    if v.get('nominal'):
        return None
    m = mj.MjModel.from_xml_path(str(src_xml))
    q = apply_key(m.key('start').qpos, m, v)
    return lambda txt: set_key_qpos(txt, q)


def start_person(rcp, v):
    """The person's schedule for every policy alike: their clip from frame 120 (their pointing gesture) into their
    quietest recorded window, then ping-pong there. Nominal: exactly Recipient5.idle(0.0). Varied: the same at person_speed."""
    f = float(v['person_speed'])
    if v.get('nominal') or f == 1.0:
        rcp.idle(0.0)
        return
    a, b = rcp.idle_window or rcp.default_idle_window()
    tm = rcp.tm
    s = tm(0.0)
    assert s < a, 'the clip must start before the idle window'
    tm.seg = [('play', 0.0, float(s), f, f, 0.0)]
    tm.log.append({'t': 0.0, 'op': 'play', 'rate': f, 's': float(s), 'note': 'seeded playback speed'})
    tm.pingpong((a - s) / (FPS * f), a, b, IDLE_PEAK_RATE * f)
    rcp._cache_t = None
    rcp.decisions.append({'t': 0.0, 'op': 'idle', 'window': [a, b], 'speed': f})


def describe(v):
    if v.get('nominal'):
        return 'seed 0: the nominal recorded episode (no variation)'
    return (f"seed {v['seed']}: robot start moved by dx {v['robot_dx_m']:+.3f} m, dy {v['robot_dy_m']:+.3f} m, "
            f"yaw {v['robot_dyaw_deg']:+.1f} deg; backpack moved by dx {v['bag_dx_m']:+.3f} m, dy {v['bag_dy_m']:+.3f} m "
            f"on the table; the person's recording plays at {v['person_speed']:.3f} x speed (the idle ping-pong too)")
