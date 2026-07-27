"""
Plot the recovered Mogi parameters against ground truth, per dataset rank K_true
and model residual rank k. Answers: how accurately is EACH physical variable
estimated, and does getting the rank right improve it?

Produces:
  figures/param_recovery.png   time series of xcen, ycen, d, dV vs ground truth
                               (under-ranked / correct-rank / over-ranked)
  figures/param_accuracy.csv   mean +/- std of each variable vs its true value

Run:  python rebuttal/plot_params.py
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))     # rebuttal/synthetic_control
REPO = os.path.dirname(os.path.dirname(HERE))         # repo root
SAVED = os.path.join(REPO, "saved/mogi_synth")
DATA = os.path.join(REPO, "data/processed/mogi_synth")
FIG = os.path.join(HERE, "figures")
K_LIST = [1, 2, 4]
RANK_LIST = [0, 1, 2, 4, 6, 8]
# (analyzer column, scale to physical, label, ground-truth column or constant)
PARAMS = [
    ("latent_xcen", 1e-3, "xcen [km]", "xcen_true_km"),
    ("latent_ycen", 1e-3, "ycen [km]", "ycen_true_km"),
    ("latent_d",    1e-3, "depth d [km]", "d_true_km"),
    ("latent_dV",   1.0,  "dV [m$^3$]", "dV_true_m3"),
]
SMOOTH = 60  # rolling-median window (days) for legibility


def load(K, k):
    hits = sorted(glob.glob(os.path.join(SAVED, f"SYN_K{K}_r{k}", "*", "models",
                                          "model_best_testset_analyzer.csv")))
    if not hits:
        return None
    ana = pd.read_csv(hits[-1])
    gt = pd.read_csv(os.path.join(DATA, f"K{K}", "groundtruth.csv"))
    return ana.merge(gt, on="date", how="inner").sort_values("date").reset_index(drop=True)


def main():
    os.makedirs(FIG, exist_ok=True)

    # ---- numeric accuracy table ----
    rows = []
    for K in K_LIST:
        for k in RANK_LIST:
            m = load(K, k)
            if m is None:
                continue
            rec = dict(K=K, k=k)
            for col, sc, label, gtcol in PARAMS:
                est = m[col].values * sc
                truth = m[gtcol].values
                rec[f"{label.split(' ')[0]}_mean"] = float(est.mean())
                rec[f"{label.split(' ')[0]}_true"] = float(np.mean(truth))
                rec[f"{label.split(' ')[0]}_bias"] = float(est.mean() - np.mean(truth))
                rec[f"{label.split(' ')[0]}_std"] = float(est.std())
            rows.append(rec)
    acc = pd.DataFrame(rows)
    acc.to_csv(os.path.join(FIG, "param_accuracy.csv"), index=False)
    show = [c for c in acc.columns if c.endswith(("_bias", "_std")) or c in ("K", "k")]
    print(acc[show].to_string(index=False))

    # ---- time-series figure: under-ranked / correct / over-ranked ----
    fig, axes = plt.subplots(len(PARAMS), len(K_LIST),
                             figsize=(5.0 * len(K_LIST), 2.9 * len(PARAMS)))
    for cj, K in enumerate(K_LIST):
        picks = [(0, "C3", f"k=0 (no residual)"),
                 (K, "C2", f"k={K} (= K_true)"),
                 (8, "C0", "k=8 (over-ranked)")]
        for ri, (col, sc, label, gtcol) in enumerate(PARAMS):
            ax = axes[ri, cj]
            truth_drawn = False
            for k, color, lab in picks:
                m = load(K, k)
                if m is None:
                    continue
                t = np.arange(len(m))
                est = pd.Series(m[col].values * sc)
                if not truth_drawn:
                    ax.plot(t, m[gtcol].values, "k--", lw=2, label="ground truth", zorder=5)
                    truth_drawn = True
                ax.plot(t, est.rolling(SMOOTH, center=True, min_periods=1).median(),
                        color=color, lw=1.3, label=lab, alpha=0.9)
            ax.set_ylabel(label, fontsize=9)
            if ri == 0:
                ax.set_title(f"K_true = {K}", fontsize=11)
            if ri == len(PARAMS) - 1:
                ax.set_xlabel("day")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Mogi parameter recovery vs ground truth "
                 f"({SMOOTH}-day rolling median)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG, "param_recovery.png")
    fig.savefig(out, dpi=120)
    print(f"\nfigure -> {out}")
    print(f"table  -> {os.path.join(FIG, 'param_accuracy.csv')}")


if __name__ == "__main__":
    main()
