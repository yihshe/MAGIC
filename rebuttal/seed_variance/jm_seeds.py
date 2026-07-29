"""
Jeffreys-Matusita separability across seeds (reviewer Q2 on Figure 8).

Recomputes the Figure-8 analysis independently for every seed and reports the
spread between runs, so the plot can carry error bars instead of a single number.

Claim under test: the tree species remain *more separable* in PILA's inferred
physical variables than in HVAE's, whichever seed is used -- i.e. the Figure-8
difference is a property of the method, not of one lucky initialisation. The
summary therefore pairs the two families seed by seed and counts how many seeds
PILA wins, alongside the mean +- std over runs.

The JM computation is the one from visuals/analysis_rtm.py (Bhattacharyya ->
JM = 2(1 - exp(-B)), latents min-max normalised by their physical ranges first),
lifted verbatim so the numbers stay comparable with the paper. What is new is
only the aggregation over runs.

Outputs (into figures/):
  jm_<FAM>_mean.csv / jm_<FAM>_std.csv    per-pair mean and std over seeds
  jm_group_stats.csv                      within-C / within-D / between, per seed
  jm_summary.txt                          the numbers to quote in the rebuttal
  jm_seeds.png                            mean heatmap per family + group bar chart

Usage (after run_seed_variance.sh):
  python rebuttal/seed_variance/jm_seeds.py
  python rebuttal/seed_variance/jm_seeds.py --seeds 123 240 318 --families PILA HVAE
"""
import argparse
import glob
import itertools
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(HERE, 'figures')
SAVE_ROOT = os.path.join(REPO, 'saved/rebuttal/seed_variance')

# Species grouping, as in visuals/analysis_rtm.py
CONIFEROUS = ['Pseudotsuga menziesii', 'Picea abies', 'Pinus nigra',
              'Larix decidua', 'Pinus sylvestris']
DECIDUOUS = ['Prunus spp', 'Fagus sylvatica', 'Carpinus betulus', 'Quercus spp',
             'Acer pseudoplatanus', 'Fraxinus excelsior', 'Alnus glutinosa']
SPECIES = CONIFEROUS + DECIDUOUS
SHORT = {s: f'C-{i+1}' for i, s in enumerate(CONIFEROUS)}
SHORT.update({s: f'D-{i+1}' for i, s in enumerate(DECIDUOUS)})


def bhattacharyya_distance(mean1, cov1, mean2, cov2):
    cov_mean = (cov1 + cov2) / 2
    term1 = 0.125 * np.dot(np.dot((mean1 - mean2).T, np.linalg.inv(cov_mean)), (mean1 - mean2))
    term2 = 0.5 * np.log(np.linalg.det(cov_mean) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
    return term1 + term2


def jeffreys_matusita_distance(b_distance):
    return 2 * (1 - np.exp(-b_distance))


def jm_matrix(csv_path, attrs, ranges, space='jm'):
    """Pairwise species distances for one run's analyzer CSV.

    space='jm' -> Jeffreys-Matusita, as plotted in Figure 8.
    space='b'  -> the underlying Bhattacharyya distance, left untransformed.

    JM = 2(1 - exp(-B)) is monotone in B, so the two agree on every *pairwise*
    ordering. They disagree on aggregates: JM saturates at 2, and the
    between-group distances here sit close to that ceiling while the
    within-group ones do not. Contrast measures built on JM (the gap, and
    especially the ratio) are therefore compressed on one side only, which can
    understate -- or even invert -- a difference that is plain in B. Report
    whichever you like, but do not read the JM ratio as if it were unsaturated.
    """
    df = pd.read_csv(csv_path)
    for attr in attrs:
        lo, hi = ranges[attr]['min'], ranges[attr]['max']
        df[f'latent_{attr}'] = (df[f'latent_{attr}'] - lo) / (hi - lo)

    means, covs = {}, {}
    for sp in SPECIES:
        sub = df[df['class'] == sp]
        if len(sub) <= len(attrs):
            raise ValueError(f'{os.path.basename(csv_path)}: species "{sp}" has only {len(sub)} rows')
        cols = [f'latent_{a}' for a in attrs]
        means[sp] = sub[cols].mean().to_numpy()
        covs[sp] = sub[cols].cov().to_numpy()

    jm = pd.DataFrame(np.zeros((len(SPECIES), len(SPECIES))), index=SPECIES, columns=SPECIES)
    for s1, s2 in itertools.combinations(SPECIES, 2):
        b = bhattacharyya_distance(means[s1], covs[s1], means[s2], covs[s2])
        d = jeffreys_matusita_distance(b) if space == 'jm' else b
        jm.at[s1, s2] = jm.at[s2, s1] = d
    return jm


def group_stats(jm):
    """Mean JM within each group and between the two groups."""
    within_c, within_d, between = [], [], []
    for s1, s2 in itertools.combinations(SPECIES, 2):
        d = jm.at[s1, s2]
        if s1 in CONIFEROUS and s2 in CONIFEROUS:
            within_c.append(d)
        elif s1 in DECIDUOUS and s2 in DECIDUOUS:
            within_d.append(d)
        else:
            between.append(d)
    within = within_c + within_d
    return {
        'within_coniferous': float(np.mean(within_c)),
        'within_deciduous': float(np.mean(within_d)),
        'within_all': float(np.mean(within)),
        'between_groups': float(np.mean(between)),
        # The two separability measures. `gap` is the additive contrast between
        # conifer-vs-deciduous pairs and same-group pairs; `ratio` is the same
        # contrast scale-free, so it is not flattered by a model that simply
        # spreads all species further apart. The claim under test is that PILA
        # scores higher than HVAE on these at every seed.
        'gap_between_minus_within': float(np.mean(between) - np.mean(within)),
        'separability_ratio': float(np.mean(between) / np.mean(within)),
    }


def find_csv(fam, seed):
    """Newest analyzer CSV for one family/seed, or None."""
    pat = os.path.join(SAVE_ROOT, f'{fam}_RTM_C_AUSTRIA_s{seed}', '*', 'models',
                       'model_best_testset_analyzer.csv')
    hits = sorted(glob.glob(pat))
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=[123, 240, 318])
    ap.add_argument('--families', nargs='+', default=['PILA', 'HVAE'])
    ap.add_argument('--rtm_paras', default=os.path.join(REPO, 'configs/rtm_paras.json'))
    ap.add_argument('--space', choices=['jm', 'b'], default='jm',
                    help="'jm' = Jeffreys-Matusita as in Figure 8 (saturates at 2); "
                         "'b' = the underlying Bhattacharyya distance, unsaturated")
    ap.add_argument('--suffix', default=None,
                    help='appended to output filenames (defaults to the space)')
    ap.add_argument('--csv', nargs='+', default=None,
                    help='explicit analyzer CSVs (FAM=path), bypassing the saved/ lookup')
    args = ap.parse_args()

    os.makedirs(FIG, exist_ok=True)
    sfx = args.suffix if args.suffix is not None else args.space
    label = 'Jeffreys-Matusita' if args.space == 'jm' else 'Bhattacharyya'
    ranges = json.load(open(args.rtm_paras))
    attrs = list(ranges.keys())

    # Locate one CSV per family/seed
    runs = {fam: {} for fam in args.families}
    if args.csv:
        for item in args.csv:
            fam, path = item.split('=', 1)
            runs.setdefault(fam, {})[len(runs[fam])] = path
    else:
        for fam in args.families:
            for seed in args.seeds:
                path = find_csv(fam, seed)
                if path is None:
                    print(f'  ! {fam} seed {seed}: no analyzer CSV under {SAVE_ROOT}')
                else:
                    runs[fam][seed] = path

    rows, mats = [], {}
    for fam in args.families:
        if not runs[fam]:
            print(f'skipping {fam}: nothing found')
            continue
        per_seed = {}
        for seed, path in sorted(runs[fam].items()):
            jm = jm_matrix(path, attrs, ranges, space=args.space)
            per_seed[seed] = jm
            stats = group_stats(jm)
            stats.update({'family': fam, 'seed': seed})
            rows.append(stats)
            print(f'{fam} seed {seed}: within {stats["within_all"]:.3f}  '
                  f'between {stats["between_groups"]:.3f}  gap {stats["gap_between_minus_within"]:.3f}')
        stack = np.stack([m.to_numpy() for m in per_seed.values()])
        mats[fam] = (pd.DataFrame(stack.mean(0), index=SPECIES, columns=SPECIES),
                     pd.DataFrame(stack.std(0, ddof=1) if len(stack) > 1 else np.zeros_like(stack[0]),
                                  index=SPECIES, columns=SPECIES))
        mats[fam][0].rename(index=SHORT, columns=SHORT).to_csv(os.path.join(FIG, f'{sfx}_{fam}_mean.csv'))
        mats[fam][1].rename(index=SHORT, columns=SHORT).to_csv(os.path.join(FIG, f'{sfx}_{fam}_std.csv'))

    if not rows:
        print('nothing to aggregate -- train the runs first (run_seed_variance.sh)')
        return

    metric_cols = ['within_coniferous', 'within_deciduous', 'within_all',
                   'between_groups', 'gap_between_minus_within', 'separability_ratio']
    stats_df = pd.DataFrame(rows)[['family', 'seed'] + metric_cols]
    stats_df.to_csv(os.path.join(FIG, f'{sfx}_group_stats.csv'), index=False)

    # Summary: mean +- std over seeds per family, then the paired PILA-vs-HVAE
    # comparison -- the claim being tested is that PILA separates the species
    # groups better than HVAE at *every* seed, not just on average.
    lines = [f'{label} separability, mean +- std over seeds',
             f'seeds: {sorted(stats_df["seed"].unique().tolist())}', '=' * 68, '']
    for fam, sub in stats_df.groupby('family'):
        n = len(sub)
        lines.append(f'{fam}  (n = {n} run{"s" if n > 1 else ""})')
        for col in metric_cols:
            sd = sub[col].std(ddof=1) if n > 1 else float('nan')
            lines.append(f'  {col:<26s} {sub[col].mean():.4f} +- {sd:.4f}')
        lines.append('')

    if len(args.families) == 2 and set(stats_df['family']) == set(args.families):
        a, b = args.families
        sa = stats_df[stats_df.family == a].set_index('seed')
        sb = stats_df[stats_df.family == b].set_index('seed')
        shared = sorted(set(sa.index) & set(sb.index))
        if shared:
            lines.append(f'Paired comparison, {a} - {b} (same seed, same data)')
            lines.append('-' * 68)
            for col in metric_cols:
                diffs = np.array([sa.at[s, col] - sb.at[s, col] for s in shared])
                per_seed = '  '.join(f'{s}:{d:+.3f}' for s, d in zip(shared, diffs))
                sd = diffs.std(ddof=1) if len(diffs) > 1 else float('nan')
                wins = int((diffs > 0).sum())
                lines.append(f'  {col}')
                lines.append(f'    per seed:  {per_seed}')
                lines.append(f'    mean +- std: {diffs.mean():+.4f} +- {sd:.4f}')
                lines.append(f'    {a} higher in {wins}/{len(diffs)} seeds')
                if len(diffs) > 1 and abs(diffs.mean()) > 2 * sd / np.sqrt(len(diffs)):
                    lines.append(f'    -> separated by more than 2 standard errors')
                else:
                    lines.append(f'    -> NOT separated by 2 standard errors; report as inconclusive')
                lines.append('')
    summary = '\n'.join(lines)
    open(os.path.join(FIG, f'{sfx}_summary.txt'), 'w').write(summary + '\n')
    print('\n' + summary)

    # Figure: mean heatmap per family + group means with across-seed error bars
    fams = [f for f in args.families if f in mats]
    fig, axes = plt.subplots(1, len(fams) + 1, figsize=(6 * len(fams) + 6, 5.5))
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        mean = mats[fam][0].rename(index=SHORT, columns=SHORT)
        im = ax.imshow(mean.to_numpy(), cmap='viridis', vmin=0,
                       vmax=2 if args.space == 'jm' else None)
        ax.set_xticks(range(len(SPECIES)), mean.columns, rotation=-45, fontsize=8)
        ax.set_yticks(range(len(SPECIES)), mean.index, fontsize=8)
        ax.set_title(f'{fam}: {label}, mean over {len(runs[fam])} seeds')
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[-1]
    keys = ['within_all', 'between_groups', 'gap_between_minus_within', 'separability_ratio']
    width = 0.8 / max(len(fams), 1)
    for i, fam in enumerate(fams):
        sub = stats_df[stats_df.family == fam]
        m = [sub[k].mean() for k in keys]
        e = [sub[k].std(ddof=1) if len(sub) > 1 else 0.0 for k in keys]
        ax.bar(np.arange(len(keys)) + i * width, m, width, yerr=e, capsize=4, label=fam)
    ax.set_xticks(np.arange(len(keys)) + width * (len(fams) - 1) / 2,
                  ['within', 'between', 'gap', 'ratio'], fontsize=9)
    ax.set_ylabel(f'{label} distance  (ratio is unitless)')
    ax.set_title('species separability, mean +- std across seeds')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f'{sfx}_seeds.png'), dpi=200)
    print(f'\nwrote {os.path.relpath(FIG, REPO)}/: {sfx}_seeds.png, {sfx}_group_stats.csv, '
          f'{sfx}_summary.txt, {sfx}_<fam>_{{mean,std}}.csv')


if __name__ == '__main__':
    main()
