"""
The headline table: PILA vs HVAE, per-trait R2 against measured ground truth.

Finds the variant-C runs under saved/greenhyperspectra/, reuses the metrics JSON
that eval_greenhyperspectra.py wrote (running it only if missing), and prints the
per-trait difference. Numbers only -- eval_greenhyperspectra.py draws the
per-model 1:1 plots.

Usage:
    python rebuttal/greenhyperspectra/compare_greenhyperspectra.py
    python rebuttal/greenhyperspectra/compare_greenhyperspectra.py --recompute
    python rebuttal/greenhyperspectra/compare_greenhyperspectra.py --seeds 123 240 318
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_greenhyperspectra import AE_RTM_SUPERVISED, TRAITS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def find_runs(fam, seeds):
    """Locate the newest checkpoint per seed. `None` = the single default run."""
    out = {}
    for seed in seeds:
        exp = f'{fam}_PROSAIL_C_GHS' if seed is None else f'{fam}_PROSAIL_C_GHS_s{seed}'
        hits = sorted(glob.glob(os.path.join(
            ROOT, 'saved', 'greenhyperspectra', exp, '*', 'models', 'model_best.pth')))
        if hits:
            out[seed] = hits[-1]
    return out


def load_metrics(ckpt, recompute):
    cfg = json.load(open(os.path.join(os.path.dirname(ckpt), 'config.json')))
    cached = os.path.join(HERE, 'figures', f"{cfg['name']}_metrics.json")
    if recompute or not os.path.exists(cached):
        subprocess.run([sys.executable,
                        os.path.join(HERE, 'eval_greenhyperspectra.py'),
                        '-r', ckpt], check=True, cwd=ROOT)
    return json.load(open(cached))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='seed sweep to aggregate; omit for the single default run')
    ap.add_argument('--recompute', action='store_true',
                    help='re-run evaluation instead of reusing cached metrics')
    args = ap.parse_args()
    # `None` addresses the single un-suffixed run.
    args.seeds = [None] if args.seeds is None else args.seeds

    data = {}
    for fam in ('PILA', 'HVAE'):
        runs = find_runs(fam, args.seeds)
        missing = [s for s in args.seeds if s not in runs]
        if missing:
            print(f'  ! {fam}: no checkpoint for seed(s) {missing}')
        data[fam] = {s: load_metrics(p, args.recompute) for s, p in runs.items()}
        if not data[fam]:
            print(f'ERROR: no {fam} runs found under saved/greenhyperspectra/')
            return

    seeds = [s for s in args.seeds if s in data['PILA'] and s in data['HVAE']]
    if not seeds:
        print('ERROR: no seed has both PILA and HVAE trained.')
        return
    multi = len(seeds) > 1
    print(f"\nRuns compared: {'seeds ' + str(seeds) if multi else 'single run per setting'}")

    def r2(fam, seed, t):
        tr = data[fam][seed].get('traits')
        return np.nan if tr is None else tr[t]['r2']

    def fmt(a):
        return (f'{np.nanmean(a):>8.3f}+-{np.nanstd(a):<5.3f}' if multi
                else f'{np.nanmean(a):>14.3f}')

    print('\n' + '=' * 78)
    print('Per-trait R2 on labeled_splits/test.csv (both trained unsupervised)')
    print('=' * 78)
    print(f"{'trait':6}{'n':>5}{'PILA':>14}{'HVAE':>14}{'PILA-HVAE':>12}"
          f"{'AE-RTM*':>10}")
    rows = {}
    for t in TRAITS:
        p = np.array([r2('PILA', s, t) for s in seeds], float)
        h = np.array([r2('HVAE', s, t) for s in seeds], float)
        rows[t] = dict(pila=p, hvae=h, diff=p - h)
        n = data['PILA'][seeds[0]]['traits'][t]['n']
        print(f"{t:6}{n:>5}{fmt(p)}{fmt(h)}{np.nanmean(p - h):>+12.3f}"
              f"{AE_RTM_SUPERVISED[t]:>10.3f}")

    allp = np.array([[r2('PILA', s, t) for t in TRAITS] for s in seeds], float)
    allh = np.array([[r2('HVAE', s, t) for t in TRAITS] for s in seeds], float)
    print('-' * 78)
    print(f"{'mean':6}{'':>5}{fmt(np.nanmean(allp, 1))}{fmt(np.nanmean(allh, 1))}"
          f"{np.nanmean(allp) - np.nanmean(allh):>+12.3f}")
    print("\n* AE-RTM is GHS's own semi-supervised model on the same test set:")
    print('  an upper reference, not a like-for-like baseline (it trains on labels).')

    wins = sum(1 for t in TRAITS if np.nanmean(rows[t]['diff']) > 0)
    print(f'\nPILA better on {wins}/{len(TRAITS)} traits.')
    if multi:
        sweeps = sum(1 for i, _ in enumerate(seeds)
                     if all(rows[t]['diff'][i] > 0 for t in TRAITS))
        print(f'PILA better on all {len(TRAITS)} traits in {sweeps}/{len(seeds)} seeds.')

    for fam in ('PILA', 'HVAE'):
        rec = [data[fam][s]['recon_rmse_standardized'] for s in seeds]
        tail = f'+-{np.std(rec):.4f}' if multi else ''
        print(f'{fam} recon RMSE (standardized): {np.mean(rec):.4f}{tail}')

    out = os.path.join(HERE, 'figures', 'comparison_C.json')
    with open(out, 'w') as f:
        json.dump({'seeds': seeds,
                   'per_trait': {t: {k: v.tolist() for k, v in rows[t].items()}
                                 for t in TRAITS},
                   'ae_rtm_supervised_reference': AE_RTM_SUPERVISED}, f, indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
