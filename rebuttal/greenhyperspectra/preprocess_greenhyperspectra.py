"""
Preprocess GreenHyperSpectra (GHS) into a self-contained dataset for PILA/HVAE.

Reads the GHS release once and writes everything PILA needs into --out_dir, so
that no later step depends on the HyspectraSSL project:

    train.npz / valid.npz / test.npz   standardized spectra + (sparse) traits
    train_x_mean.npy / train_x_scale.npy   per-band StandardScaler stats
    ghs_prosail_paras.json             sigmoid ranges for the learnable latents
    ghs_prosail_constants.json         fixed leaf parameters (car, anth)
    meta.json                          provenance and row counts

Train pool = labeled_splits/train.csv + the first --n_splits shards of the
unlabeled pool, merged into one array with a `source` flag. Labels are NOT used
for training (PILA/HVAE are unsupervised here); they are carried along for
diagnostics and are used to derive the latent ranges.

Usage:
    python rebuttal/greenhyperspectra/preprocess_greenhyperspectra.py --ghs_root /maps/ys611/HyspectraSSL \
        --n_splits 2 --out_dir rebuttal/greenhyperspectra/data
"""
import argparse
import hashlib
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# The 8 GHS traits, in the column order used by the released CSVs.
TRAITS = ['cab', 'cw', 'cm', 'LAI', 'cp', 'cbc', 'car', 'anth']

# Free PROSPECT-D/4SAIL latents, in the order the encoder emits them.
#
# Only the four traits that carry the reconstruction. Measured by clamping each
# latent to its training median and re-running the physics: LAI +171%, cbc +96%,
# cw +78%, cab +22% -- against cp +2.8%, car +4.9%, anth +0.0%.
#
# `cm` replaces the (cp, cbc) pair. PROSPECT-PRO resolves dry matter into
# protein and carbon-based constituents, but that second degree of freedom is
# not identifiable here: cos(K_cbc, K_cm) = 0.9986, and weighted by the observed
# contents (cp is 9.4x scarcer than cbc) the unique protein signal is 4.7% of
# the cbc-driven one. Its matched-filter SNR is 0.48, i.e. a full p10-p90 swing
# in cp moves the spectrum half as much as the model's own residual. The labels
# agree: cm == cp + cbc to 1e-7, so only two degrees of freedom are present.
# PROSPECT-D takes cm directly, which also removes the negative-cbc NaN path in
# the GHS implementation (they derive cbc = cm - cp).
#
# `car` and `anth` are fixed instead of inferred -- see FIXED below.
#
# `N` (leaf structure) is absent because GHS does not learn it either; omitting
# it means nothing overwrites RTM.para_dict['N'], so it keeps the default 1.5.
Z_PARAS = ['cab', 'cw', 'cm', 'LAI']

# Non-retrievable leaf parameters, held at the (rounded) training-label median.
#
# Carotenoid and chlorophyll absorb almost the same way: their specific
# absorption coefficients peak 4 nm apart (436 vs 440 nm) with cosine 0.776 over
# 400-750 nm, and chlorophyll saturates the blue, so d(trans)/dk there is 0.081
# against 0.306 at 550 nm. Left free, the car latent came out 0.755-correlated
# with the inferred cab -- a chlorophyll shadow rather than a retrieval.
# Anthocyanin is weaker still (0.22 vs car's 0.75 at peak) and 0.924-collinear
# with cbrown, which is itself frozen. Neither is a defensible retrieval here;
# GHS's own *semi-supervised* AE-RTM scores anth R2 = 0.335, its worst trait.
#
# Rounded from the train-label medians (car 7.97, anth 1.26), which the report
# prints so the rounding stays auditable. Deliberately not the RTM defaults
# (car 10, anth 2): those sit at the 76th and 93rd percentile of these labels.
FIXED = {'car': 8.0, 'anth': 1.25}

# Ranges hardcoded in the GHS code base (AE_RTM_architectures.py). These are the
# min/max of the full labeled pool (train + test), except for cbc which is wrong
# on both ends. Kept here as the documented starting point.
GHS_DECLARED = {
    'cab': (4.44830458, 229.4974769),          # ug/cm2
    'cw': (0.000048333333, 0.08062008166),     # cm
    'cm': (0.0000136666667, 0.06638073277),    # g/cm2
    'LAI': (0.063333333, 8.77),                # m2/m2
    'cp': (3.1320099999999997e-06, 0.00420423377267),   # g/cm2
    'cbc': (0.001350791, 0.046180855),         # g/cm2
    'car': (1.182576236, 40.44321686),         # ug/cm2
    'anth': (0.561035015, 2.981116889),        # ug/cm2
}

# Overrides applied on top of GHS_DECLARED, each with the reason it is needed.
# Verified against the pooled train+test labels (see --report output).
RANGE_OVERRIDES = {
    # cbc: declared min excludes 159 pooled samples and declared max excludes 8.
    # Replaced by the observed pool range.
    'cbc': (1.05e-05, 0.063579,
            'declared range excludes 159 samples below min and 8 above max'),
    # cab: declared max is a single train outlier (q99.9 = 98.3, test max 78.1).
    'cab': (4.44830458, 120.0,
            'declared max 229.5 is one train-only outlier; q99.9 = 98.3'),
    # car: declared max is a single train outlier (q99.9 = 20.5, test max 19.6).
    'car': (1.182576236, 30.0,
            'declared max 40.4 is one train-only outlier; q99.9 = 20.5'),
    # anth: only 289 labeled train rows, almost certainly one source dataset.
    # Binding the sigmoid to [0.56, 2.98] is an unjustified prior; widen.
    'anth': (0.05, 4.0,
             'declared range reflects 289 rows from one source; widened'),
}

# Lower bound per latent, rounded to clean decimals. Zero is fine for the
# pigments (a fully senescent leaf has no chlorophyll) but three parameters need
# a strictly positive floor, each verified against PROSAIL directly:
#   LAI  - 4SAIL divides by ks*LAI, so LAI=0 is NaN in forward and backward.
#          Keeps the 0.01 floor PILA already uses in configs/rtm_paras.json.
#   cw   - with cw=0 AND cm=0 there is no absorber left anywhere in the SWIR,
#   cm     and PROSPECT's plate model degenerates to NaN. Either one alone is
#          safe; the pair is not, and HVAE reaches it because draw() clamps
#          sampled latents to exactly [0, 1]. Floors sit just below the smallest
#          observed label (cw 4.83e-5, cm 1.37e-5).
FLOORS = {'cab': 0.0, 'cw': 0.00001, 'cm': 0.00001, 'LAI': 0.01,
          'cp': 0.000001, 'cbc': 0.00001, 'car': 0.0, 'anth': 0.0}

# Headroom above the largest observed label, so the data does not sit on the
# sigmoid's upper end-stop. Rounded to clean decimals afterwards.
UPPER_PAD = 0.05

# Decimals to round each upper bound to, chosen per magnitude so the published
# ranges read like the ones in the GHS code base rather than padded floats.
ROUND_TO = {'cab': 0, 'cw': 3, 'cm': 3, 'LAI': 1, 'cp': 4, 'cbc': 3,
            'car': 0, 'anth': 1}


def band_columns(df):
    """Band columns are the purely numeric column names (wavelength in nm)."""
    return [c for c in df.columns if str(c).isdigit()]


def build_ranges(pool, report):
    """Derive the sigmoid ranges: declared -> overridden -> floor + padded max."""
    ranges = OrderedDict()
    notes = {}
    for name in Z_PARAS:
        if name in RANGE_OVERRIDES:
            lo_raw, hi_raw, why = RANGE_OVERRIDES[name]
        else:
            lo_raw, hi_raw = GHS_DECLARED[name]
            why = 'GHS declared range (= pooled label min/max)'

        lo = FLOORS[name]
        hi = round(hi_raw * (1.0 + UPPER_PAD), ROUND_TO[name])
        assert hi > hi_raw, f'{name}: rounded max {hi} below observed {hi_raw}'
        ranges[name] = {'min': float(lo), 'max': float(hi)}
        notes[name] = {'raw_min': float(lo_raw), 'raw_max': float(hi_raw),
                       'floor': float(lo), 'reason': why}

    # Verify every observed label lands strictly inside its range, and record
    # where the observed extremes sit in z-space.
    report.append('')
    report.append('Latent ranges (z = sigmoid output, rescaled to [min, max]):')
    header = (f"  {'para':5} {'min':>12} {'max':>12} | "
              f"{'z(obs min)':>10} {'z(obs max)':>10} | outside")
    report.append(header)
    for name in Z_PARAS:
        lo, hi = ranges[name]['min'], ranges[name]['max']
        obs = pool[name].dropna()
        z_lo = (obs.min() - lo) / (hi - lo)
        z_hi = (obs.max() - lo) / (hi - lo)
        n_out = int(((obs < lo) | (obs > hi)).sum())
        report.append(f"  {name:5} {lo:12.6g} {hi:12.6g} | "
                      f"{z_lo:10.3f} {z_hi:10.3f} | {n_out}")

    return ranges, notes


def build_constants(train_labels, report):
    """Fixed leaf parameters, with the medians they were rounded from."""
    report.append('')
    report.append('Fixed (non-retrievable) leaf parameters:')
    report.append(f"  {'para':5} {'value':>8} {'train median':>14} "
                  f"{'RTM default':>12}   percentile of the fixed value")
    for name, value in FIXED.items():
        obs = train_labels[name].dropna()
        pct = float((obs < value).mean() * 100)
        default = {'car': 10, 'anth': 2}[name]
        report.append(f'  {name:5} {value:8.4g} {obs.median():14.4g} '
                      f'{default:12g}   {pct:.1f}%  (n={len(obs)})')
    report.append('  N is not listed: it keeps the RTM default of 1.5, as in GHS.')
    return dict(FIXED)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ghs_root', default='/maps/ys611/HyspectraSSL',
                    help='GreenHyperSpectra checkout (read once, then unused)')
    ap.add_argument('--out_dir', default='rebuttal/greenhyperspectra/data')
    ap.add_argument('--n_splits', type=int, default=2,
                    help='number of unlabeled shards from Splits/ to merge')
    ap.add_argument('--val_frac', type=float, default=0.1,
                    help='fraction of the train pool held out for val_rec_loss')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    report = []

    lb_dir = os.path.join(args.ghs_root, 'Datasets', 'labeled_splits')
    train_csv = os.path.join(lb_dir, 'train.csv')
    test_csv = os.path.join(lb_dir, 'test.csv')

    print(f'Reading {train_csv} ...')
    lb_train = pd.read_csv(train_csv)
    print(f'Reading {test_csv} ...')
    lb_test = pd.read_csv(test_csv)

    bands = band_columns(lb_train)
    assert band_columns(lb_test) == bands, 'test.csv band columns differ'
    n_bands = len(bands)
    report.append(f'Bands: {n_bands} ({bands[0]}-{bands[-1]} nm, '
                  'water-absorption regions removed by GHS)')

    # --- unlabeled shards -------------------------------------------------
    unlab_frames = []
    split_files = []
    for i in range(1, args.n_splits + 1):
        f = os.path.join(args.ghs_root, 'Splits', f'split_{i}.csv')
        print(f'Reading {f} ...')
        df = pd.read_csv(f)
        assert band_columns(df) == bands, f'{f} band columns differ'
        unlab_frames.append(df[bands])
        split_files.append(os.path.basename(f))
    unlab = pd.concat(unlab_frames, ignore_index=True) if unlab_frames \
        else pd.DataFrame(columns=bands)

    # --- assemble the combined train pool ---------------------------------
    X_lb = lb_train[bands].to_numpy(np.float32)
    Y_lb = lb_train[TRAITS].to_numpy(np.float32)
    ds_lb = lb_train['dataset'].to_numpy(np.int32)

    X_un = unlab.to_numpy(np.float32)
    Y_un = np.full((len(X_un), len(TRAITS)), np.nan, np.float32)
    ds_un = np.full(len(X_un), -1, np.int32)

    X = np.concatenate([X_lb, X_un], 0)
    Y = np.concatenate([Y_lb, Y_un], 0)
    ds = np.concatenate([ds_lb, ds_un], 0)
    # 0 = labeled spectrum (labels held out of training), 1 = unlabeled.
    src = np.concatenate([np.zeros(len(X_lb), np.int8),
                          np.ones(len(X_un), np.int8)])

    report.append(f'Train pool: {len(X_lb)} labeled + {len(X_un)} unlabeled '
                  f'({args.n_splits} shard(s)) = {len(X)} rows')

    # --- train / valid split, stratified by source ------------------------
    val_mask = np.zeros(len(X), bool)
    for s in (0, 1):
        idx = np.where(src == s)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, int(round(len(idx) * args.val_frac)),
                          replace=False)
        val_mask[pick] = True
    tr_mask = ~val_mask
    report.append(f'  -> train {tr_mask.sum()} / valid {val_mask.sum()} '
                  f'(val_frac={args.val_frac}, stratified by source)')

    # --- standardize on the train portion only ----------------------------
    scaler = StandardScaler().fit(X[tr_mask])
    x_mean = scaler.mean_.astype(np.float32)
    x_scale = scaler.scale_.astype(np.float32)
    # Guard against zero-variance bands (would produce inf after scaling).
    n_deg = int((x_scale < 1e-8).sum())
    if n_deg:
        x_scale[x_scale < 1e-8] = 1.0
        report.append(f'  !! {n_deg} band(s) had ~zero variance; scale set to 1')

    def standardize(a):
        return ((a - x_mean) / x_scale).astype(np.float32)

    X_test = standardize(lb_test[bands].to_numpy(np.float32))
    Y_test = lb_test[TRAITS].to_numpy(np.float32)
    ds_test = lb_test['dataset'].to_numpy(np.int32)

    np.save(os.path.join(args.out_dir, 'train_x_mean.npy'), x_mean)
    np.save(os.path.join(args.out_dir, 'train_x_scale.npy'), x_scale)

    common = dict(bands=np.array(bands), traits=np.array(TRAITS))
    np.savez_compressed(os.path.join(args.out_dir, 'train.npz'),
                        X=standardize(X[tr_mask]), Y=Y[tr_mask],
                        source=src[tr_mask], dataset=ds[tr_mask], **common)
    np.savez_compressed(os.path.join(args.out_dir, 'valid.npz'),
                        X=standardize(X[val_mask]), Y=Y[val_mask],
                        source=src[val_mask], dataset=ds[val_mask], **common)
    np.savez_compressed(os.path.join(args.out_dir, 'test.npz'),
                        X=X_test, Y=Y_test,
                        source=np.zeros(len(X_test), np.int8),
                        dataset=ds_test, **common)
    report.append(f'Test set: {len(X_test)} rows from labeled_splits/test.csv '
                  '(row-disjoint from train.csv)')

    # --- latent ranges ----------------------------------------------------
    pool = pd.concat([lb_train[TRAITS], lb_test[TRAITS]], ignore_index=True)
    ranges, notes = build_ranges(pool, report)
    with open(os.path.join(args.out_dir, 'ghs_prosail_paras.json'), 'w') as f:
        json.dump(ranges, f, indent=4)

    # --- fixed leaf parameters (medians taken from train labels only) -----
    constants = build_constants(lb_train[TRAITS], report)
    with open(os.path.join(args.out_dir, 'ghs_prosail_constants.json'), 'w') as f:
        json.dump(constants, f, indent=4)

    # --- per-trait label coverage on the evaluation set -------------------
    report.append('')
    report.append('Test-set label coverage (n per trait, for NaN-aware eval):')
    report.append('  ' + '  '.join(f'{t}={int(lb_test[t].count())}'
                                   for t in TRAITS))

    meta = {
        'source_repo': args.ghs_root,
        'source_files': {
            'labeled_train': train_csv,
            'labeled_test': test_csv,
            'unlabeled_splits': split_files,
        },
        'n_bands': n_bands,
        'bands': bands,
        'traits': TRAITS,
        'z_paras': Z_PARAS,
        'leaf_model': 'prospectD',
        'fixed': {**FIXED, 'N': 1.5, 'tts': 30, 'tto': 0, 'psi': 0,
                  'note': 'car/anth held at rounded train medians; N, geometry '
                          'at the RTM defaults, matching GHS AE_RTM'},
        'rows': {'train': int(tr_mask.sum()), 'valid': int(val_mask.sum()),
                 'test': int(len(X_test)),
                 'train_labeled': int((src[tr_mask] == 0).sum()),
                 'train_unlabeled': int((src[tr_mask] == 1).sum())},
        'val_frac': args.val_frac,
        'seed': args.seed,
        'standardization': 'per-band StandardScaler fit on the train portion',
        'range_notes': notes,
        'upper_pad': UPPER_PAD,
        'floors': FLOORS,
        'labels_used_for': ['latent range derivation', 'evaluation only'],
    }
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    txt = '\n'.join(report)
    print('\n' + txt)
    with open(os.path.join(args.out_dir, 'report.txt'), 'w') as f:
        f.write(txt + '\n')
    print(f'\nWrote dataset to {args.out_dir}')


if __name__ == '__main__':
    main()
