"""
Compare GHS rebuttal variants against their baseline, on normalized MAE.

Shared by rebuttal/prior_vs_endstop/ and rebuttal/consistency_loss/: both train
variants of a model whose baseline already exists under saved/greenhyperspectra/,
and both score checkpoints with rebuttal/greenhyperspectra/eval_greenhyperspectra.py,
which writes a `<name>_preds.csv` per run. This script reads those CSVs.

**Metric: nMAE**, the definition used in visuals/analysis_rtm_insitu.py -- scale
truth and prediction to [0,1] by the trait's physical range, then take the mean
absolute error. Lower is better.

R2 is deliberately not the headline here. These models are trained fully
unsupervised, so their predictions carry a scale/offset bias against the measured
traits; R2 is dominated by that bias (it goes strongly negative even where the
rank correlation is decent), which makes it a poor instrument for the question
these experiments ask -- namely whether a loss change moves the retrieval closer
to the measurements. Pearson r is reported alongside as a shape-only reference.

Usage:
  python rebuttal/compare_ghs_variants.py --dir rebuttal/consistency_loss/figures \\
      --baseline PILA_PROSAIL_C_GHS
  python rebuttal/compare_ghs_variants.py --dir rebuttal/prior_vs_endstop/figures \\
      --baseline HVAE_PROSAIL_C_GHS
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PARAS = os.path.join(HERE, 'greenhyperspectra/data/ghs_prosail_paras.json')


def nmae(y, y_hat, lo, hi):
    """Mean absolute error after scaling both series to [0,1] by the physical range."""
    return float(np.mean(np.abs((y - y_hat) / (hi - lo))))


def metrics_for(csv_path, ranges):
    df = pd.read_csv(csv_path)
    traits = [c[5:] for c in df.columns if c.startswith('true_')]
    out = {}
    for t in traits:
        y, yh = df[f'true_{t}'].to_numpy(), df[f'pred_{t}'].to_numpy()
        ok = np.isfinite(y) & np.isfinite(yh)
        if ok.sum() < 3:
            continue
        y, yh = y[ok], yh[ok]
        lo, hi = ranges[t]['min'], ranges[t]['max']
        out[t] = {
            'n': int(ok.sum()),
            'nmae': nmae(y, yh, lo, hi),
            'mae': float(np.mean(np.abs(y - yh))),
            'pearson': float(np.corrcoef(y, yh)[0, 1]) if np.std(yh) > 0 else float('nan'),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='figures/ directory holding *_preds.csv')
    ap.add_argument('--baseline', required=True, help='run name to treat as the reference')
    ap.add_argument('--paras', default=PARAS)
    args = ap.parse_args()

    ranges = json.load(open(args.paras))
    runs = {}
    for path in sorted(glob.glob(os.path.join(args.dir, '*_preds.csv'))):
        runs[os.path.basename(path)[:-len('_preds.csv')]] = metrics_for(path, ranges)

    if args.baseline not in runs:
        raise SystemExit(f'baseline {args.baseline!r} not found in {args.dir}; '
                         f'have: {sorted(runs)}')

    traits = list(runs[args.baseline])
    order = [args.baseline] + [r for r in sorted(runs) if r != args.baseline]

    print(f'\nnMAE per trait (lower is better); baseline = {args.baseline}\n')
    head = f'{"run":<42}' + ''.join(f'{t:>10}' for t in traits) + f'{"mean":>10}'
    print(head)
    print('-' * len(head))
    base_mean = None
    for r in order:
        vals = [runs[r][t]['nmae'] for t in traits if t in runs[r]]
        m = float(np.mean(vals))
        if r == args.baseline:
            base_mean = m
        row = f'{r:<42}' + ''.join(f'{runs[r][t]["nmae"]:>10.4f}' for t in traits if t in runs[r])
        print(row + f'{m:>10.4f}' + ('   <- baseline' if r == args.baseline else
                                     f'   ({m - base_mean:+.4f} vs baseline)'))

    print(f'\nPearson r per trait (shape only, ignores bias)\n')
    print(head)
    print('-' * len(head))
    for r in order:
        vals = [runs[r][t]['pearson'] for t in traits if t in runs[r]]
        row = f'{r:<42}' + ''.join(f'{runs[r][t]["pearson"]:>10.3f}' for t in traits if t in runs[r])
        print(row + f'{np.mean(vals):>10.3f}')

    ns = {t: runs[args.baseline][t]['n'] for t in traits}
    print(f'\nlabelled test rows per trait: {ns}')


if __name__ == '__main__':
    main()
