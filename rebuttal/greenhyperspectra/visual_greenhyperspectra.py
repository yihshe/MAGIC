"""
Rebuttal figures and table for GreenHyperSpectra.

Two outputs, both driven by the *_preds.csv files eval_greenhyperspectra.py
writes into --fig_dir:

  1. A mean / MAE table per trait, one row per model plus the measured ground
     truth. Mean is in physical units on the subset of the test set that carries
     a label for that trait, so it is directly comparable to the ground-truth
     mean. MAE is on the range-normalized residual, matching the convention in
     visuals/analysis_rtm_insitu.py -- it is comparable across traits with very
     different units. Written as CSV; the markdown rendering is printed for the
     "Results" section of README.md rather than written to its own file.

  2. KDE plots of the normalized residual (y_hat_norm - y_norm), one row of four
     panels in the order cab, cw, cm, LAI. Styling follows the in-situ residual
     figure in visuals/analysis_rtm_insitu.py. Two comparisons are drawn:
     HVAE vs PILA, and HVAE with the end-stop penalty vs PILA.

Usage:
    python rebuttal/greenhyperspectra/visual_greenhyperspectra.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))

# Panel order requested for the figures and the table columns.
TRAITS = ['cab', 'cw', 'cm', 'LAI']

# The three runs, keyed by the prefix of their *_preds.csv.
MODELS = {
    'HVAE': 'HVAE_PROSAIL_C_GHS',
    'HVAE (end-stop)': 'HVAE_PROSAIL_C_GHS_endstop',
    'PILA': 'PILA_PROSAIL_C_GHS',
}

# Row order in the table: baselines first, ours in bold, truth last.
TABLE_ORDER = ['HVAE', 'HVAE (end-stop)', 'PILA']

# Which pairs get a KDE figure, and the file each is written to.
COMPARISONS = [
    ('HVAE', 'PILA', 'kde_hvae_vs_pila'),
    ('HVAE (end-stop)', 'PILA', 'kde_hvae_endstop_vs_pila'),
]

# Reference palette: baseline red, ours blue -- as in analysis_rtm_insitu.py.
COLOR_BASE, COLOR_OURS = 'red', 'blue'


def load_preds(fig_dir):
    """{model: DataFrame} for every run that has a predictions CSV."""
    out = {}
    for label, prefix in MODELS.items():
        path = os.path.join(fig_dir, f'{prefix}_preds.csv')
        if os.path.exists(path):
            out[label] = pd.read_csv(path)
        else:
            print(f'  ! missing {path} -- skipping {label}')
    return out


def paired(df, trait):
    """(truth, prediction) on the rows where this trait is labelled."""
    t, p = df[f'true_{trait}'].to_numpy(float), df[f'pred_{trait}'].to_numpy(float)
    m = np.isfinite(t) & np.isfinite(p)
    return t[m], p[m]


def residuals(df, trait, ranges):
    """Range-normalized prediction error, (pred - true) / (max - min)."""
    t, p = paired(df, trait)
    span = ranges[trait]['max'] - ranges[trait]['min']
    return (p - t) / span


def build_table(preds, ranges):
    """Mean (physical units) and MAE (normalized) per model and trait."""
    rows = []
    for label in TABLE_ORDER:
        if label not in preds:
            continue
        row = {'Model': label}
        for tr in TRAITS:
            t, p = paired(preds[label], tr)
            row[f'{tr}_mean'] = float(np.mean(p))
            row[f'{tr}_mae'] = float(np.mean(np.abs(residuals(preds[label], tr, ranges))))
            row[f'{tr}_n'] = int(len(t))
        rows.append(row)

    # Ground truth: same labelled subsets, so the means line up with the above.
    ref = preds[TABLE_ORDER[-1] if TABLE_ORDER[-1] in preds else list(preds)[0]]
    row = {'Model': 'Ground Truth'}
    for tr in TRAITS:
        t, _ = paired(ref, tr)
        row[f'{tr}_mean'] = float(np.mean(t))
        row[f'{tr}_mae'] = np.nan
        row[f'{tr}_n'] = int(len(t))
    rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df):
    head = '| Model | ' + ' | '.join(f'{t} Mean | {t} MAE' for t in TRAITS) + ' |'
    sep = '|' + '---|' * (1 + 2 * len(TRAITS))
    out = [head, sep]
    for _, r in df.iterrows():
        name = f"**{r['Model']}**" if r['Model'] == 'PILA' else r['Model']
        cells = []
        for t in TRAITS:
            # 4 significant figures, not fixed decimals: cw and cm are O(1e-2),
            # so 3 dp would collapse e.g. cm 0.008711 to 0.009.
            cells.append(f"{r[f'{t}_mean']:.4g}")
            cells.append('N/A' if not np.isfinite(r[f'{t}_mae'])
                         else f"{r[f'{t}_mae']:.3f}")
        out.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    out.append('')
    out.append('n per trait: ' + ', '.join(
        f"{t}={int(df.iloc[-1][f'{t}_n'])}" for t in TRAITS)
        + '. Mean is in physical units over the labelled rows; MAE is on the '
          'range-normalized residual.')
    return '\n'.join(out)


def kde_figure(preds, ranges, base_label, ours_label, out_path, args):
    """One row of four KDE panels, baseline vs ours, in TRAITS order."""
    fig, axs = plt.subplots(1, len(TRAITS), figsize=(6.5 * len(TRAITS), 5))

    for j, tr in enumerate(TRAITS):
        ax = axs[j]
        r_base = residuals(preds[base_label], tr, ranges)
        r_ours = residuals(preds[ours_label], tr, ranges)

        # Shared, symmetric x-range per panel so the two KDEs are comparable.
        x_range = float(np.max(np.abs(np.concatenate([r_base, r_ours])))) * 1.2

        # Baseline first, ours on top -- same z-order convention as the in-situ figure.
        sns.kdeplot(data=r_base, ax=ax, fill=True, alpha=0.5,
                    color=COLOR_BASE, linewidth=2, zorder=1)
        sns.kdeplot(data=r_ours, ax=ax, fill=True, alpha=0.5,
                    color=COLOR_OURS, linewidth=2, zorder=2)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2.0, alpha=1.0, zorder=0)
        ax.set_xlim(-x_range, x_range)

        ax.set_ylabel('Density', fontsize=args.fontsize)
        ax.set_xlabel('Prediction Error', fontsize=args.fontsize)
        ax.tick_params(axis='both', which='major', labelsize=args.ticksize)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.grid(False)
        ax.yaxis.labelpad = 10

        # MAE annotations, baseline above ours, in the matching colours.
        ty, tx = 0.95, args.text_x
        ax.text(tx, ty, f'{base_label}: {np.mean(np.abs(r_base)):.3f}',
                transform=ax.transAxes, fontsize=args.annotsize,
                color=COLOR_BASE, verticalalignment='top', weight='bold')
        ax.text(tx, ty - 0.11, f'{ours_label}: {np.mean(np.abs(r_ours)):.3f}',
                transform=ax.transAxes, fontsize=args.annotsize,
                color=COLOR_OURS, verticalalignment='top', weight='bold')

    plt.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig_dir', default=os.path.join(HERE, 'figures'))
    ap.add_argument('--ranges', default=os.path.join(HERE, 'data',
                                                     'ghs_prosail_paras.json'))
    # styling, mirroring visuals/analysis_rtm_insitu.py
    ap.add_argument('--fontsize', type=int, default=30)
    ap.add_argument('--ticksize', type=int, default=25)
    ap.add_argument('--annotsize', type=int, default=23)
    ap.add_argument('--text_x', type=float, default=0.02)
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    ranges = json.load(open(os.path.abspath(args.ranges)))
    fig_dir = os.path.abspath(args.fig_dir)
    preds = load_preds(fig_dir)
    if not preds:
        raise SystemExit(f'no *_preds.csv found in {fig_dir}')

    table = build_table(preds, ranges)
    csv_path = os.path.join(fig_dir, 'mean_mae_table.csv')
    table.to_csv(csv_path, index=False, float_format='%.6g')

    # Printed, not written: the markdown version lives in README.md.
    print('\n' + to_markdown(table) + '\n')
    print(f'table   -> {csv_path}  (markdown above -> README.md "Results")')

    for base, ours, stem in COMPARISONS:
        if base not in preds or ours not in preds:
            print(f'  ! skipping {stem}: missing a run')
            continue
        p = kde_figure(preds, ranges, base, ours,
                       os.path.join(fig_dir, f'{stem}.png'), args)
        print(f'figure  -> {p}')


if __name__ == '__main__':
    main()
