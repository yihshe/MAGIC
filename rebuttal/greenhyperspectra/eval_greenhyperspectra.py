"""
Evaluate a trained PILA or HVAE checkpoint against the GreenHyperSpectra
ground-truth traits, and plot the result.

This is the point of the whole experiment: the models are trained without ever
seeing a label, so the per-trait R2 here is an independent check on whether the
inferred physical variables mean what the RTM says they mean -- the thing the
Austrian dataset cannot provide.

Metrics are NaN-aware per trait: GHS labels are sparse (no row carries all 8
traits, and anth has only 73 test rows), so every number is reported with its n.

Writes <name>_preds.csv, <name>_metrics.json and <name>_scatter.png into
--out_dir. Plot styling is adjustable via --ncols / --dpi / --point_size /
--no-density.

Usage:
    python rebuttal/greenhyperspectra/eval_greenhyperspectra.py \
        -r saved/greenhyperspectra/PILA_PROSAIL_C_GHS/<run_id>/models/model_best.pth
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Retrieved traits, in the order the encoder emits them. The other four GHS
# traits are not retrieved: cp/cbc are replaced by cm (PROSPECT-D), and car/anth
# are held fixed because their absorption is largely masked by chlorophyll.
TRAITS = ['cab', 'cw', 'cm', 'LAI']

# Column order in the released GHS CSVs, which is the layout of Y in the npz.
ALL_TRAITS = ['cab', 'cw', 'cm', 'LAI', 'cp', 'cbc', 'car', 'anth']

UNITS = {'cab': 'ug/cm2', 'cw': 'cm', 'cm': 'g/cm2', 'LAI': 'm2/m2',
         'cp': 'g/cm2', 'cbc': 'g/cm2', 'car': 'ug/cm2', 'anth': 'ug/cm2'}

# GHS's own AE-RTM, seed-averaged, on the same test set. Semi-supervised, so an
# upper reference only -- not a like-for-like baseline.
AE_RTM_SUPERVISED = {'cab': 0.674, 'cw': 0.738, 'cm': 0.738, 'LAI': 0.636,
                     'cp': 0.727, 'cbc': 0.761, 'car': 0.657, 'anth': 0.335}


def metrics(y_true, y_pred):
    """NaN-aware R2 / RMSE / nRMSE / Pearson for one trait."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(m.sum())
    if n < 2:
        return dict(n=n, r2=np.nan, rmse=np.nan, nrmse=np.nan, pearson=np.nan)
    t, p = y_true[m], y_pred[m]
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    rmse = float(np.sqrt(ss_res / n))
    rng = float(t.max() - t.min())
    return dict(
        n=n,
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        rmse=rmse,
        nrmse=float(rmse / rng) if rng > 0 else np.nan,
        # Correlation separates "wrong scale/offset" from "no information",
        # which matters a lot for an unsupervised inversion.
        pearson=float(np.corrcoef(t, p)[0, 1]),
    )


def metrics_table(truth, preds):
    """Per-trait metrics, keyed by trait name."""
    return {t: metrics(truth[:, i], preds[:, i]) for i, t in enumerate(TRAITS)}


def save_predictions(path, truth, preds):
    """One row per test spectrum: measured and inferred value for each trait."""
    pd.DataFrame(
        np.hstack([truth, preds]),
        columns=[f'true_{t}' for t in TRAITS] + [f'pred_{t}' for t in TRAITS]
    ).to_csv(path, index=False)
    return path


def load_predictions(path):
    """Read a predictions CSV back as (truth, preds, label)."""
    df = pd.read_csv(path)
    truth = df[[f'true_{t}' for t in TRAITS]].to_numpy(float)
    preds = df[[f'pred_{t}' for t in TRAITS]].to_numpy(float)
    label = os.path.basename(path).replace('_preds.csv', '')
    return truth, preds, label


def scatter_plot(res, truth, preds, out_path, title, args):
    """Grid of 1:1 inferred-vs-measured panels, coloured by point density."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    traits = [t for t in TRAITS if res[t]['n'] >= 2]
    if not traits:
        return None
    ncol = min(args.ncols, len(traits))
    nrow = int(np.ceil(len(traits) / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(3.3 * ncol, 3.3 * nrow))
    axes = axes.ravel()

    for ax, t in zip(axes, traits):
        i = TRAITS.index(t)
        m = np.isfinite(truth[:, i]) & np.isfinite(preds[:, i])
        x, y = truth[m, i], preds[m, i]

        colour = 'tab:blue'
        if not args.no_density and len(x) > 20:
            try:
                # Colour by point density, the style used in visuals/.
                colour = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
            except Exception:
                pass  # singular covariance, e.g. near-duplicate points
        ax.scatter(x, y, c=colour, s=args.point_size, cmap='viridis',
                   edgecolors='none')

        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        r = res[t]
        ax.set_title(f"{t}  $R^2$={r['r2']:.3f}  r={r['pearson']:.3f}  n={r['n']}",
                     fontsize=10)
        ax.set_xlabel(f'measured [{UNITS[t]}]', fontsize=8)
        ax.set_ylabel('inferred', fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(traits):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--resume', required=True, help='path to model_best.pth')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'figures'))
    # plot styling
    ap.add_argument('--ncols', type=int, default=4)
    ap.add_argument('--dpi', type=int, default=150)
    ap.add_argument('--point_size', type=float, default=10.0)
    ap.add_argument('--no-density', dest='no_density', action='store_true',
                    help='plain markers instead of KDE-density colouring')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    import torch
    import data_loader.data_loaders as module_data
    from model import PHYS_VAE, PHYS_VAE_SMPL

    torch.set_num_threads(args.threads)

    # Resolve before building the model: PROSAIL's InitModel calls os.chdir(),
    # so any relative path stops working once the physics module exists.
    ckpt_path = os.path.abspath(args.resume)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg = json.load(open(os.path.join(os.path.dirname(ckpt_path), 'config.json')))
    name = cfg['name']
    is_pila = cfg['arch']['type'] == 'PHYS_VAE_SMPL'
    no_phy = cfg['arch']['phys_vae']['no_phy']

    loader = getattr(module_data, cfg['data_loader']['type'])(
        cfg['data_loader']['data_dir_test'], batch_size=args.batch_size,
        shuffle=False, validation_split=0.0, num_workers=0)

    model = (PHYS_VAE_SMPL if is_pila else PHYS_VAE)(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    if is_pila and not no_phy and cfg['arch']['phys_vae']['dim_z_aux'] > 0:
        model.dec.set_tau_r_from_checkpoint(ckpt.get('tau_r_values', None))
    model.eval()

    if is_pila:
        pv = cfg['trainer']['phys_vae']
        hard_z_phy = not pv.get('use_kl_term_z_phy', pv.get('use_kl_term', False))
        hard_z_aux = not pv.get('use_kl_term_z_aux', pv.get('use_kl_term', False))

    trait_cols = [ALL_TRAITS.index(t) for t in TRAITS]
    preds, truth, recon, physonly, target = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['spectrum']
            # `traits` carries all 8 GHS columns; keep the retrieved ones.
            truth.append(batch['traits'].numpy()[:, trait_cols])
            if is_pila:
                z_phy, _z_aux, x_PB, x_P = model(
                    x, inference=True, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
                physonly.append(x_P.numpy())
            else:
                z_phy_stat, z_aux2_stat, _u = model.encode(x)
                z_phy, z_aux2 = model.draw(z_phy_stat, z_aux2_stat, hard_z=True)
                out = model.decode(z_phy, z_aux2, full=True)
                x_PB, x_P = out[0], out[1]
                physonly.append(x_P.numpy())
            recon.append(x_PB.numpy())
            target.append(x.numpy())
            if not no_phy:
                phys = model.physics_model.rescale(z_phy)
                preds.append(np.stack([phys[t].numpy() for t in TRAITS], 1))

    truth = np.concatenate(truth)
    recon = np.concatenate(recon)
    physonly = np.concatenate(physonly)
    target = np.concatenate(target)
    rec_rmse = float(np.sqrt(np.mean((recon - target) ** 2)))
    # How much of the fit is the physics doing on its own, before the
    # correction? If this is far worse than rec_rmse the residual is carrying
    # the reconstruction and the latents are only weakly identified.
    phys_rmse = float(np.sqrt(np.mean((physonly - target) ** 2)))
    resid_frac = float(np.mean(np.abs(recon - physonly)) /
                       (np.mean(np.abs(physonly)) + 1e-12))

    print(f'\n=== {name} ===')
    print(f'checkpoint       : {ckpt_path}  (epoch {ckpt.get("epoch", "?")})')
    print(f'test spectra     : {len(truth)}')
    print(f'recon RMSE       : {rec_rmse:.4f}  (physics + correction)')
    print(f'physics-only RMSE: {phys_rmse:.4f}  (correction removed)')
    print(f'|correction|/|physics| : {resid_frac:.1%}')

    out = {'name': name, 'checkpoint': ckpt_path, 'n_test': len(truth),
           'recon_rmse_standardized': rec_rmse,
           'physics_only_rmse_standardized': phys_rmse,
           'correction_rel_magnitude': resid_frac}

    if no_phy:
        print('no_phy variant: no physical latents to score against ground truth.')
        out['traits'] = None
    else:
        preds = np.concatenate(preds)
        res = metrics_table(truth, preds)
        out['traits'] = res
        print(f"\n{'trait':6}{'n':>6}{'R2':>9}{'pearson':>10}{'RMSE':>12}{'nRMSE':>9}")
        for t in TRAITS:
            r = res[t]
            print(f"{t:6}{r['n']:>6}{r['r2']:>9.3f}{r['pearson']:>10.3f}"
                  f"{r['rmse']:>12.5g}{r['nrmse']:>9.3f}")
        valid = [res[t]['r2'] for t in TRAITS if np.isfinite(res[t]['r2'])]
        out['mean_r2'] = float(np.mean(valid))
        print(f"{'mean':6}{'':>6}{out['mean_r2']:>9.3f}")

        csv_path = save_predictions(
            os.path.join(out_dir, f'{name}_preds.csv'), truth, preds)
        print(f'\npreds   -> {csv_path}')
        fig_path = scatter_plot(
            res, truth, preds, os.path.join(out_dir, f'{name}_scatter.png'),
            f'{name} - inferred vs measured', args)
        print(f'figure  -> {fig_path}')

    json_path = os.path.join(out_dir, f'{name}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'metrics -> {json_path}')


if __name__ == '__main__':
    main()
