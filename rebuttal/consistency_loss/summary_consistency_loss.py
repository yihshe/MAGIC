"""
Summary for the physical-consistency experiment (reviewer: "could PILA be improved
through addition of a physical consistency loss as in HVAE?").

Two outputs:

  1. mean_mae_table.csv -- mean prediction (physical units) and MAE per trait, one
     row per run plus the measured ground truth. Same format and conventions as
     rebuttal/greenhyperspectra/figures/mean_mae_table.csv: the mean is taken over
     the labelled rows for that trait so it is directly comparable to the
     ground-truth mean, and MAE is on the range-normalized residual
     (pred - true) / (max - min), matching visuals/analysis_rtm_insitu.py, so it is
     comparable across traits with very different units.

     Rows: PILA with no consistency term (the published GHS run, weight 0), then
     the three co-trained weights. The point of the table is the comparison against
     that first row.

  2. loss_balance.csv -- what each weight actually buys, measured from the training
     logs rather than assumed: the consistency term, the reconstruction loss, and
     the ratio (weight x consistency) / rec_loss at the start and end of training.
     This is the evidence behind the choice of 0.1 / 1 / 10.

Usage:
    python rebuttal/consistency_loss/summary_consistency_loss.py
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

TRAITS = ['cab', 'cw', 'cm', 'LAI']

# label -> (preds prefix, consistency weight, training-log glob)
RUNS = [
    ('PILA (no consistency)', 'PILA_PROSAIL_C_GHS', 0.0,
     'saved/greenhyperspectra/PILA_PROSAIL_C_GHS/0728_214108/log/info.log'),
    ('PILA + consistency w=0.1', 'PILA_PROSAIL_C_GHS_S1_w0.1', 0.1,
     'saved/rebuttal/consistency_loss/PILA_PROSAIL_C_GHS_S1_w0.1/*/log/info.log'),
    ('PILA + consistency w=1.0', 'PILA_PROSAIL_C_GHS_S2_w1.0', 1.0,
     'saved/rebuttal/consistency_loss/PILA_PROSAIL_C_GHS_S2_w1.0/*/log/info.log'),
    ('PILA + consistency w=10', 'PILA_PROSAIL_C_GHS_S3_w10', 10.0,
     'saved/rebuttal/consistency_loss/PILA_PROSAIL_C_GHS_S3_w10/*/log/info.log'),
]

BASELINE = 'PILA (no consistency)'


def paired(df, trait):
    """(truth, prediction) on the rows where this trait is labelled."""
    t, p = df[f'true_{trait}'].to_numpy(float), df[f'pred_{trait}'].to_numpy(float)
    m = np.isfinite(t) & np.isfinite(p)
    return t[m], p[m]


def residuals(df, trait, ranges):
    """Range-normalized prediction error, (pred - true) / (max - min)."""
    t, p = paired(df, trait)
    return (p - t) / (ranges[trait]['max'] - ranges[trait]['min'])


def build_table(preds, ranges):
    """Mean (physical units) and MAE (normalized) per run and trait."""
    rows = []
    for label in preds:
        row = {'Model': label}
        for tr in TRAITS:
            t, p = paired(preds[label], tr)
            row[f'{tr}_mean'] = float(np.mean(p))
            row[f'{tr}_mae'] = float(np.mean(np.abs(residuals(preds[label], tr, ranges))))
            row[f'{tr}_n'] = int(len(t))
        rows.append(row)

    ref = preds[BASELINE] if BASELINE in preds else preds[list(preds)[0]]
    row = {'Model': 'Ground Truth'}
    for tr in TRAITS:
        t, _ = paired(ref, tr)
        row[f'{tr}_mean'] = float(np.mean(t))
        row[f'{tr}_mae'] = np.nan
        row[f'{tr}_n'] = int(len(t))
    rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df):
    head = '| Model | ' + ' | '.join(f'{t} Mean | {t} MAE' for t in TRAITS) + ' | mean MAE |'
    sep = '|' + '---|' * (2 + 2 * len(TRAITS))
    out = [head, sep]
    for _, r in df.iterrows():
        name = f"**{r['Model']}**" if r['Model'] == BASELINE else r['Model']
        cells = []
        for t in TRAITS:
            # 4 significant figures, not fixed decimals: cw and cm are O(1e-2).
            cells.append(f"{r[f'{t}_mean']:.4g}")
            cells.append('N/A' if not np.isfinite(r[f'{t}_mae']) else f"{r[f'{t}_mae']:.3f}")
        maes = [r[f'{t}_mae'] for t in TRAITS]
        mean_mae = 'N/A' if not np.isfinite(maes[0]) else f'{np.mean(maes):.3f}'
        out.append(f'| {name} | ' + ' | '.join(cells) + f' | {mean_mae} |')
    out.append('')
    out.append('n per trait: ' + ', '.join(
        f"{t}={int(df.iloc[-1][f'{t}_n'])}" for t in TRAITS)
        + '. Mean is in physical units over the labelled rows; MAE is on the '
          'range-normalized residual.')
    return '\n'.join(out)


def parse_log(path_glob, weight):
    """Per-epoch rec_loss and consistency term from a training log.

    Both are logged once per epoch and in order -- rec_loss on the 'Epoch N
    Summary' line, the consistency term on the 'syn_data_loss' metric line -- so
    they are zipped by position.
    """
    hits = sorted(glob.glob(os.path.join(REPO, path_glob)))
    if not hits:
        return None
    text = open(hits[-1]).read()
    rec = [float(m) for m in re.findall(r'Epoch \d+ Summary - Loss: [\d.eE+-]+, Rec: ([\d.eE+-]+)', text)]
    syn = [float(m) for m in re.findall(r'syn_data_loss\s+:\s+([\d.eE+-]+)', text)]
    if not rec:
        return None
    if weight == 0.0:
        syn = [0.0] * len(rec)          # term never computed in the baseline
    n = min(len(rec), len(syn))
    if n == 0:
        return None
    return np.array(rec[:n]), np.array(syn[:n]), os.path.relpath(hits[-1], REPO)


def build_balance(args):
    """How large the weighted consistency term is relative to rec_loss."""
    rows = []
    for label, _, weight, log_glob in RUNS:
        got = parse_log(log_glob, weight)
        if got is None:
            print(f'  ! no training log for {label} -- skipping in loss_balance')
            continue
        rec, syn, src = got
        first, last = 0, len(rec) - 1
        rows.append({
            'Model': label,
            'weight': weight,
            'epochs': len(rec),
            'consistency_first': syn[first],
            'consistency_last': syn[last],
            'rec_first': rec[first],
            'rec_last': rec[last],
            # what the term is actually worth inside the total loss
            'ratio_first_pct': 100.0 * weight * syn[first] / rec[first],
            'ratio_last_pct': 100.0 * weight * syn[last] / rec[last],
            'log': src,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fig_dir', default=os.path.join(HERE, 'figures'))
    ap.add_argument('--ranges', default=os.path.join(
        REPO, 'rebuttal/greenhyperspectra/data/ghs_prosail_paras.json'))
    args = ap.parse_args()

    ranges = json.load(open(os.path.abspath(args.ranges)))
    fig_dir = os.path.abspath(args.fig_dir)

    preds = {}
    for label, prefix, _, _ in RUNS:
        path = os.path.join(fig_dir, f'{prefix}_preds.csv')
        if os.path.exists(path):
            preds[label] = pd.read_csv(path)
        else:
            print(f'  ! missing {path} -- skipping {label}')
    if not preds:
        raise SystemExit(f'no *_preds.csv found in {fig_dir}')

    table = build_table(preds, ranges)
    csv_path = os.path.join(fig_dir, 'mean_mae_table.csv')
    table.to_csv(csv_path, index=False, float_format='%.6g')
    print('\n' + to_markdown(table) + '\n')

    # Difference against the weight-0 run, which is the whole question.
    if BASELINE in preds:
        base = table[table.Model == BASELINE].iloc[0]
        base_mean = float(np.mean([base[f'{t}_mae'] for t in TRAITS]))
        print(f'change in mean MAE vs "{BASELINE}" (negative = better):')
        for _, r in table.iterrows():
            if r['Model'] in (BASELINE, 'Ground Truth'):
                continue
            m = float(np.mean([r[f'{t}_mae'] for t in TRAITS]))
            print(f'  {r["Model"]:<28} {m:.4f}   {m - base_mean:+.4f}  '
                  f'({100 * (m - base_mean) / base_mean:+.1f}%)')

    balance = build_balance(args)
    if not balance.empty:
        bal_path = os.path.join(fig_dir, 'loss_balance.csv')
        balance.to_csv(bal_path, index=False, float_format='%.6g')
        print('\nweighted consistency term as a share of rec_loss, from the training logs:\n')
        print(f'{"Model":<28}{"weight":>8}{"consistency":>13}{"rec_loss":>11}'
              f'{"share":>9}   (first epoch -> last epoch)')
        for _, r in balance.iterrows():
            print(f'{r["Model"]:<28}{r["weight"]:>8g}'
                  f'{r["consistency_first"]:>13.4f}{r["rec_first"]:>11.4f}{r["ratio_first_pct"]:>8.1f}%')
            print(f'{"":<28}{"":>8}'
                  f'{r["consistency_last"]:>13.4f}{r["rec_last"]:>11.4f}{r["ratio_last_pct"]:>8.1f}%')
        print(f'\nbalance -> {bal_path}')
    print(f'table   -> {csv_path}')


if __name__ == '__main__':
    main()
