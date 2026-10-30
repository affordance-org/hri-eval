"""Roll out a policy in the point-to-fetch scene with a recorded person, then score the trial.

usage: python run.py [--policy MODULE:CLASS] [--params JSON|FILE] [--out DIR] [--max-time 24]

Without --policy the baseline (hri_eval/fetch_present.py) runs. A policy is any class with
policy_step(ctx) -> ControlOutput (see README.md)."""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'hri_eval'))
sys.path.insert(0, str(ROOT))
import check_gates as CG  # noqa: E402
import fetch_present as FP  # noqa: E402
import weights  # noqa: E402

SCENE = ROOT / 'hri_eval' / 'scene' / 'room_point_fetch.xml'
GATES = [('G2_no_fall', 'no fall'), ('G3_pickup', 'pickup'), ('G4_carry', 'carry'), ('G5_handover', 'handover'), ('G6_pace', 'pace')]


def load_params(s):
    return json.loads(Path(s).read_text()) if s.endswith('.json') else json.loads(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default=None, help='MODULE:CLASS (default: the baseline)')
    ap.add_argument('--params', default=None, help='JSON or a .json file, passed to the policy constructor')
    ap.add_argument('--out', default=None, help='trial folder (default: trials/<policy>)')
    ap.add_argument('--max-time', type=float, default=24.0)
    a = ap.parse_args()
    weights.ensure()
    params = load_params(a.params or ('{}' if a.policy else str(ROOT / 'hri_eval' / 'baseline_params.json')))
    a.out = a.out or str(ROOT / 'trials' / (a.policy.split(':')[1] if a.policy else 'baseline'))
    if a.policy:
        mod, cls = a.policy.split(':')
        ctrl = getattr(importlib.import_module(mod), cls)(**params)
        rcp = FP.make_recipient({}, str(SCENE))
    else:
        ctrl = FP.PointFetch(params)
        rcp = FP.make_recipient(ctrl.p, str(SCENE))
    rcp.idle(0.0)      # the person: recorded pointing gesture, then the recorded idle window, for every policy alike
    runner = FP.PointFetchRunner(ctrl, a.out, xml=str(SCENE), recipient=rcp, max_time=a.max_time, check=False,
                                 description=f'policy {a.policy or "baseline fetch_present"} in the point-to-fetch scene')
    runner.run()
    res = CG.check(Path(a.out), frame_stride=1)
    print()
    for g, what in GATES:
        r = res['gates'].get(g, {})
        print(f"{'PASS' if r.get('pass') else 'FAIL'}  {what}")


if __name__ == '__main__':
    main()
