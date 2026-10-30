"""Roll out a policy in the point-to-fetch scene with a recorded person, then score every trial.

usage: python run.py [--policy MODULE:CLASS] [--params JSON|FILE] [--episodes N] [--seed S] [--jobs J]
                     [--out DIR] [--max-time 24] [--scenario point-fetch] [--list]

Without --policy the baseline (hri_eval/fetch_present.py) runs. A policy is a class with reset(ctx) and
policy_step(ctx) -> episode.ControlOutput (joint level), or a subclass of interfaces.CommandPolicy with
act(obs) -> interfaces.Command (command level; see hri_eval/interfaces.py). Episode i runs seed S+i into DIR/ep<i>
(seed 0 is the nominal recorded episode, other seeds vary the start: hri_eval/variations.py). Every trial is scored by
the independent checker (hri_eval/check_gates.py); DIR/report.json and DIR/report.md hold the scores. A run replaces
the epNNN folders and reports an earlier run left in DIR; use another --out to keep them.

exit code: 0 when every episode ran to a normal end (max_time or controller_done) and is valid, whatever the scores;
1 when an episode could not run, ended in an exception or a divergence, or is not valid, or the --policy is bad;
2 for an unknown --scenario."""
import argparse
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'hri_eval'))
sys.path.insert(0, str(ROOT))
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import catalog as C  # noqa: E402
import check_gates as CG  # noqa: E402
import fetch_present as FP  # noqa: E402
import interfaces as I  # noqa: E402
import report as R  # noqa: E402
import variations as V  # noqa: E402
import weights  # noqa: E402

SCENARIOS = {'point-fetch': ROOT / 'hri_eval' / 'scene' / 'room_point_fetch.xml'}
BASELINE_PARAMS = ROOT / 'hri_eval' / 'baseline_params.json'


def load_params(s):
    return json.loads(Path(s).read_text()) if s.endswith('.json') else json.loads(s)


def load_class(spec):
    mod, _, name = spec.rpartition(':')
    if not mod or not name:
        raise SystemExit(f'--policy must be MODULE:CLASS (or path/to/file.py:CLASS), got {spec!r}')
    try:
        if mod.endswith('.py'):
            p = Path(mod).resolve()
            sp = importlib.util.spec_from_file_location(p.stem, p)
            module = importlib.util.module_from_spec(sp)
            sys.modules[p.stem] = module
            sp.loader.exec_module(module)
        else:
            module = importlib.import_module(mod)
    except (ImportError, FileNotFoundError) as e:
        raise SystemExit(f'--policy {spec}: cannot import {mod!r} ({e}). Run from the repo root, or pass '
                         'path/to/file.py:CLASS.')
    if not hasattr(module, name):
        raise SystemExit(f'--policy {spec}: {mod!r} has no class {name!r}')
    return getattr(module, name)


class Runner(FP.PointFetchRunner):
    def __init__(self, *a, variation=None, **kw):
        self.variation = variation
        super().__init__(*a, **kw)

    def disclosures(self):
        v = self.variation
        return super().disclosures() + ([f'Seeded variation, {V.describe(v)}.'] if v and not v['nominal'] else [])


def run_episode(policy, params, out, seed, max_time, scene, verbose=True):
    t0 = time.time()
    v = V.sample(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if policy:
        cls = load_class(policy)
        pol = cls(**params)
        ctrl = I.CommandAdapter(pol) if I.is_command_policy(cls) else pol
        rcp = FP.make_recipient({}, str(scene))
    else:
        ctrl = FP.PointFetch(params)
        rcp = FP.make_recipient(ctrl.p, str(scene))
    V.start_person(rcp, v)    # the person: recorded pointing gesture, then the recorded idle window, for every policy alike
    (out / 'variation.json').write_text(json.dumps(v, indent=1) + '\n')
    runner = Runner(ctrl, out, xml=str(scene), recipient=rcp, max_time=max_time, check=False, verbose=verbose,
                    xml_edit=V.xml_editor(v, scene), variation=v,
                    description=f'policy {policy or "baseline fetch_present"} in the point-to-fetch scene, {V.describe(v)}')
    runner.run()
    res = CG.check(out, frame_stride=1)
    row = R.episode_row(out, seed, v, res)
    row['wall_time_s'] = round(time.time() - t0, 1)     # the episode, the checker and the distance measurement
    return row


def safe_episode(*a, **kw):
    try:
        return run_episode(*a, **kw)
    except Exception:  # noqa
        tb = traceback.format_exc()
        if kw.get('verbose', True):
            print(tb, file=sys.stderr)
        return R.error_row(Path(a[2]).name, a[3], V.sample(a[3]), tb[-2000:])


def clear_old_run(out):
    """Remove the trial folders (epNNN) and reports an earlier run wrote to `out`, so that the folder holds only this
    run. Only folders named epNNN that look like trial folders (or are empty) are removed."""
    n = 0
    for d in sorted(out.iterdir()):
        if d.is_dir() and re.fullmatch(r'ep\d{3}', d.name) and (
                not any(d.iterdir()) or any((d / f).exists() for f in ('variation.json', 'trial.xml', 'summary.json'))):
            shutil.rmtree(d)
            n += 1
    for f in ('report.json', 'report.md'):
        if (out / f).exists():
            (out / f).unlink()
    if n:
        print(f'replacing an earlier run in {out} ({n} episode folder{"s" if n != 1 else ""} removed)', flush=True)


def line(r):
    s = ' '.join(f"{n} {'PASS' if r['scores'][k] else 'FAIL'}" for k, n, _, _ in C.SCORES)
    return f"{r['episode']} seed {r['seed']}: {'valid' if r['valid'] else 'NOT VALID'}, {s}" + \
        (f"  ({r['failure'].strip().splitlines()[-1]})" if r.get('failure') else '')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', default=None, help='MODULE:CLASS or path/to/file.py:CLASS (default: the baseline)')
    ap.add_argument('--params', default=None, help='JSON or a .json file, passed to the policy constructor as keywords')
    ap.add_argument('--episodes', type=int, default=1, help='number of episodes (seeds SEED..SEED+N-1)')
    ap.add_argument('--seed', type=int, default=0, help='first seed; seed 0 is the nominal recorded episode')
    ap.add_argument('--jobs', type=int, default=1, help='episodes run in parallel processes')
    ap.add_argument('--out', default=None, help='output folder (default: trials/<CLASS or baseline>)')
    ap.add_argument('--max-time', type=float, default=24.0, help='episode length [s]')
    ap.add_argument('--scenario', default='point-fetch', help='point-fetch (the only one today; see --list)')
    ap.add_argument('--list', action='store_true', help='print scenarios, interfaces, variations, scores and on-request items')
    a = ap.parse_args()
    if a.list:
        C.print_list()
        return
    if a.scenario not in SCENARIOS:
        print(f"Scenario {a.scenario!r} is not available yet. Available: {', '.join(SCENARIOS)}.\n"
              f"Request it here (opens a pre-filled GitHub issue):\n  {C.scenario_request(a.scenario)}\n"
              f"Scenarios on request: python run.py --list")
        sys.exit(2)
    if a.episodes < 1 or a.jobs < 1 or a.seed < 0:
        ap.error('--episodes and --jobs must be >= 1, --seed >= 0')
    weights.ensure()
    params = load_params(a.params or ('{}' if a.policy else str(BASELINE_PARAMS)))
    interface = 'joint'
    if a.policy:
        cls = load_class(a.policy)                     # fail fast on a bad MODULE:CLASS
        interface = 'command' if I.is_command_policy(cls) else 'joint'
    out = Path(a.out or ROOT / 'trials' / (cls.__name__ if a.policy else 'baseline'))
    out.mkdir(parents=True, exist_ok=True)
    clear_old_run(out)
    seeds = [a.seed + i for i in range(a.episodes)]
    scene = SCENARIOS[a.scenario]
    jobs = [(a.policy, params, str(out / f'ep{i:03d}'), s, a.max_time, scene) for i, s in enumerate(seeds)]
    rows = [None] * len(jobs)
    if a.jobs == 1 or len(jobs) == 1:
        for i, j in enumerate(jobs):
            print(f'episode {i + 1}/{len(jobs)} -> {j[2]}: {V.describe(V.sample(j[3]))}', flush=True)
            rows[i] = safe_episode(*j, verbose=True)
            print(line(rows[i]), flush=True)
    else:
        print(f'{len(jobs)} episodes in {min(a.jobs, len(jobs))} processes -> {out}', flush=True)
        with ProcessPoolExecutor(max_workers=min(a.jobs, len(jobs)), mp_context=mp.get_context('spawn')) as ex:
            fut = {ex.submit(safe_episode, *j, verbose=False): i for i, j in enumerate(jobs)}
            for f in as_completed(fut):
                i = fut[f]
                try:
                    rows[i] = f.result()
                except Exception:  # noqa  (a worker process died)
                    rows[i] = R.error_row(Path(jobs[i][2]).name, seeds[i], V.sample(seeds[i]), traceback.format_exc()[-2000:])
                print(line(rows[i]), flush=True)
    meta = {'scenario': a.scenario, 'policy': a.policy or 'baseline', 'interface': interface,
            'params': str(BASELINE_PARAMS.relative_to(ROOT)) if not a.policy and not a.params else params,
            'seeds': seeds, 'max_time_s': a.max_time, 'commit': R.git_commit(ROOT),
            'local_changes': R.git_local_changes(ROOT)}
    R.write(out, meta, rows)
    print()
    R.print_table(rows)
    print(f"\nreport: {out / 'report.md'} and report.json")
    bad = [r['episode'] for r in rows if not r['valid'] or r['end_reason'] not in R.NORMAL_END]
    if bad:
        print(f"{len(bad)} of {len(rows)} episodes did not run cleanly (not valid, or ended by an error, an exception or "
              f"a divergence): {', '.join(bad)}. See 'end_reason' and 'failure' in report.json.")
        sys.exit(1)


if __name__ == '__main__':
    main()
