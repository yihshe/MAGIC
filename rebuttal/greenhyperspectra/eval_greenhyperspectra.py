"""
Evaluate a trained PILA or HVAE checkpoint against the GreenHyperSpectra
ground-truth traits.

This is the point of the whole experiment: the models are trained without ever
seeing a label, so the per-trait R2 here is an independent check on whether the
inferred physical variables mean what the RTM says they mean -- the thing the
Austrian dataset cannot provide.

Metrics are NaN-aware per trait: GHS labels are sparse (no row carries all 8
traits, and anth has only 73 test rows), so every number is reported with its n.

Usage:
    python rebuttal/greenhyperspectra/eval_greenhyperspectra.py \
        -r saved/greenhyperspectra/PILA_PROSAIL_C_GHS/<run_id>/models/model_best.pth
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import data_loader.data_loaders as module_data  # noqa: E402
from model import PHYS_VAE, PHYS_VAE_SMPL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAITS = ['cab', 'cw', 'cm', 'LAI', 'cp', 'cbc', 'car', 'anth']
UNITS = {'cab': 'ug/cm2', 'cw': 'cm', 'cm': 'g/cm2', 'LAI': 'm2/m2',
         'cp': 'g/cm2', 'cbc': 'g/cm2', 'car': 'ug/cm2', 'anth': 'ug/cm2'}


def metrics(y_true, y_pred):
    """NaN-aware R2 / RMSE / nRMSE for one trait."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(m.sum())
    if n < 2:
        return dict(n=n, r2=np.nan, rmse=np.nan, nrmse=np.nan, pearson=np.nan)
    t, p = y_true[m], y_pred[m]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    rmse = float(np.sqrt(ss_res / n))
    rng = t.max() - t.min()
    return dict(
        n=n,
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        rmse=rmse,
        nrmse=float(rmse / rng) if rng > 0 else np.nan,
        # Correlation separates "wrong scale/offset" from "no information",
        # which matters a lot for an unsupervised inversion.
        pearson=float(np.corrcoef(t, p)[0, 1]),
    )


def scatter_plot(res, preds, truth, out_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    cols = [t for t in TRAITS if res[t]['n'] >= 2]
    if not cols:
        return
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.3 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, t in zip(axes, cols):
        i = TRAITS.index(t)
        m = np.isfinite(truth[:, i]) & np.isfinite(preds[:, i])
        x, y = truth[m, i], preds[m, i]
        # Colour by point density (the KDE style used in visuals/), falling
        # back to plain markers when the sample is too small or degenerate.
        c = 'tab:blue'
        if len(x) > 20:
            try:
                c = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
            except Exception:
                pass
        ax.scatter(x, y, c=c, s=10, cmap='viridis', edgecolors='none')
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        r = res[t]
        ax.set_title(f"{t}  $R^2$={r['r2']:.3f}  r={r['pearson']:.3f}  n={r['n']}",
                     fontsize=10)
        ax.set_xlabel(f'measured ({UNITS[t]})', fontsize=8)
        ax.set_ylabel('inferred', fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--resume', required=True, help='path to model_best.pth')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'figures'))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    # Resolve before building the model: PROSAIL's InitModel calls os.chdir(),
    # so any relative path stops working once the physics module exists.
    ckpt_path = os.path.abspath(args.resume)
    args.out_dir = os.path.abspath(args.out_dir)
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

    preds, truth, recon, target = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['spectrum']
            truth.append(batch['traits'].numpy())
            if is_pila:
                z_phy, _z_aux, x_PB, _x_P = model(
                    x, inference=True, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
            else:
                z_phy_stat, z_aux2_stat, _u = model.encode(x)
                z_phy, z_aux2 = model.draw(z_phy_stat, z_aux2_stat, hard_z=True)
                out = model.decode(z_phy, z_aux2, full=False)
                x_PB = out[0] if isinstance(out, tuple) else out
            recon.append(x_PB.numpy())
            target.append(x.numpy())
            if not no_phy:
                phys = model.physics_model.rescale(z_phy)
                preds.append(np.stack([phys[t].numpy() for t in TRAITS], 1))

    truth = np.concatenate(truth)
    recon = np.concatenate(recon)
    target = np.concatenate(target)
    rec_rmse = float(np.sqrt(np.mean((recon - target) ** 2)))

    print(f'\n=== {name} ===')
    print(f'checkpoint      : {ckpt_path}  (epoch {ckpt.get("epoch", "?")})')
    print(f'test spectra    : {len(truth)}')
    print(f'recon RMSE      : {rec_rmse:.4f} (standardized units)')

    out = {'name': name, 'checkpoint': ckpt_path, 'n_test': len(truth),
           'recon_rmse_standardized': rec_rmse}

    if no_phy:
        print('no_phy variant: no physical latents to score against ground truth.')
        out['traits'] = None
    else:
        preds = np.concatenate(preds)
        res = {t: metrics(truth[:, i], preds[:, i]) for i, t in enumerate(TRAITS)}
        out['traits'] = res
        print(f"\n{'trait':6}{'n':>6}{'R2':>9}{'pearson':>10}{'RMSE':>12}{'nRMSE':>9}")
        for t in TRAITS:
            r = res[t]
            print(f"{t:6}{r['n']:>6}{r['r2']:>9.3f}{r['pearson']:>10.3f}"
                  f"{r['rmse']:>12.5g}{r['nrmse']:>9.3f}")
        valid = [res[t]['r2'] for t in TRAITS if np.isfinite(res[t]['r2'])]
        print(f"{'mean':6}{'':>6}{np.mean(valid):>9.3f}")
        out['mean_r2'] = float(np.mean(valid))

        os.makedirs(args.out_dir, exist_ok=True)
        fig_path = os.path.join(args.out_dir, f'{name}_scatter.png')
        scatter_plot(res, preds, truth, fig_path, f'{name} - inferred vs measured')
        print(f'\nfigure  -> {fig_path}')

        csv_path = os.path.join(args.out_dir, f'{name}_preds.csv')
        hdr = ','.join([f'true_{t}' for t in TRAITS] + [f'pred_{t}' for t in TRAITS])
        np.savetxt(csv_path, np.hstack([truth, preds]), delimiter=',',
                   header=hdr, comments='')
        print(f'preds   -> {csv_path}')

    json_path = os.path.join(args.out_dir, f'{name}_metrics.json')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'metrics -> {json_path}')


if __name__ == '__main__':
    main()
