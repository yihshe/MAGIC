"""
Evaluate the synthetic-control rank sweep against known ground truth and produce
the reviewer-facing metrics + figures.

For each run SYN_K{K}_r{k} it reads the test analyzer CSV (physics/bias/recovered
params), the dataset groundtruth.csv (true dV, x,y,d, bias) and basis.npy (true V),
and computes:

  recon_rms_mm     : reconstruction error (output vs target), mm  -> FIDELITY elbow
  dV_r, dV_nrmse   : recovered dV(t) vs true ramp                 -> INTERPRETABILITY
  drift_{x,y,d}_km : std over time of recovered source geometry   -> IDENTIFIABILITY
                     (true geometry is fixed, so nonzero = leakage)
  subspace_overlap : how well the learned residual subspace spans -> ARCHITECTURAL
                     the true bias subspace V (1 = full)             CLAIM (rank works)
  bias_rms_mm      : recovered bias vs true bias, mm

Expectation: every metric has an elbow at model rank k = K_true (the intrinsic
bias rank); k<K leaves bias uncorrected and leaks into the physics; k>=K recovers.

Run:  python rebuttal/eval_sweep.py
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED = os.path.join(REPO, "saved/mogi_synth")
DATA = os.path.join(REPO, "data/processed/mogi_synth")
OUT_DIR = os.path.join(REPO, "rebuttal/figures")
K_LIST = [1, 2, 4]
RANK_LIST = [0, 1, 2, 4, 6, 8]
STATIONS = list(json.load(open(os.path.join(REPO, "configs/station_info.json"))).keys())
GPS = [f"{d}_{s}" for d in ("ux", "uy", "uz") for s in STATIONS]  # 36, matches gen order


def latest_analyzer(name):
    hits = sorted(glob.glob(os.path.join(SAVED, name, "*", "models",
                                          "model_best_testset_analyzer.csv")))
    return hits[-1] if hits else None


def subspace_overlap(bias_recovered_mm, V_true):
    """Fraction of the true K-dim bias subspace captured by the recovered residual.
    (1/K) * ||V^T Q||_F^2 in [0,1], Q = orthonormal basis of recovered bias rows."""
    K = V_true.shape[1]
    Bc = bias_recovered_mm - bias_recovered_mm.mean(0, keepdims=True)
    # right singular vectors span the row-space (the 36-dim spatial subspace)
    _, s, Vt = np.linalg.svd(Bc, full_matrices=False)
    keep = int(min(K, np.sum(s > 1e-6 * (s[0] if s[0] > 0 else 1))))
    if keep == 0:
        return 0.0
    Q = Vt[:keep].T                       # 36 x keep, orthonormal columns
    return float(np.sum((V_true.T @ Q) ** 2) / K)


def eval_run(K, k):
    name = f"SYN_K{K}_r{k}"
    ana_path = latest_analyzer(name)
    if ana_path is None:
        return None
    ana = pd.read_csv(ana_path)
    gt = pd.read_csv(os.path.join(DATA, f"K{K}", "groundtruth.csv"))
    basis = np.load(os.path.join(DATA, f"K{K}", "basis.npy"), allow_pickle=True).item()
    xs = np.load(os.path.join(DATA, f"K{K}", "train_x_scale.npy"))
    xm = np.load(os.path.join(DATA, f"K{K}", "train_x_mean.npy"))
    V_true = basis["V"]                    # 36 x K, raw-space orthonormal

    m = ana.merge(gt, on="date", how="inner").sort_values("date").reset_index(drop=True)

    out = m[[f"output_{g}" for g in GPS]].values
    tgt = m[[f"target_{g}" for g in GPS]].values
    bias_std = m[[f"bias_{g}" for g in GPS]].values
    recon_rms_mm = float(np.sqrt(np.mean(((out - tgt) * xs) ** 2)))

    # recovered dV(t) vs true ramp
    dV_hat = m["latent_dV"].values
    dV_true = m["dV_true_m3"].values
    dV_r = float(np.corrcoef(dV_hat, dV_true)[0, 1])
    dV_nrmse = float(np.sqrt(np.mean((dV_hat - dV_true) ** 2)) /
                     (dV_true.std() + 1e-12))  # scale-free vs true variation

    # dV and d are NOT separately identifiable from surface displacements
    # (classic Mogi depth-volume trade-off, corr(d_hat, dV_hat) ~ -0.9). The
    # combination dV/d^3 sets the observable amplitude, so track it too.
    d_hat_km = m["latent_d"].values / 1000.0
    dVd3_hat = dV_hat / np.maximum(d_hat_km, 1e-6) ** 3
    dVd3_true = dV_true / (m["d_true_km"].values ** 3)
    dVd3_r = float(np.corrcoef(dVd3_hat, dVd3_true)[0, 1])
    # strength of the depth-volume confounding itself: |corr(d_hat, dV_hat)|.
    # ~1 => the two are traded off against each other; lower => better separated.
    dV_d_confound = float(abs(np.corrcoef(d_hat_km, dV_hat)[0, 1]))

    # Most assumption-free interpretability check: does the PHYSICS branch
    # reproduce the true volcanic signal S_vol(t)? (mm, parameterization-free)
    # init_output is standardized, i.e. (Mogi_raw - x_mean)/x_scale, so the
    # train mean must be added back to recover raw mm before comparing.
    svol_hat = m[[f"init_output_{g}" for g in GPS]].values * xs + xm
    svol_true = m[[f"svol_true_{g}" for g in GPS]].values
    svol_rms_mm = float(np.sqrt(np.mean((svol_hat - svol_true) ** 2)))
    svol_r = float(np.corrcoef(svol_hat.ravel(), svol_true.ravel())[0, 1])

    # source-geometry drift (true geometry fixed -> nonzero std = leakage). meters->km
    drift_x = float(m["latent_xcen"].std() / 1000.0)
    drift_y = float(m["latent_ycen"].std() / 1000.0)
    drift_d = float(m["latent_d"].std() / 1000.0)

    # bias recovery (residual is std-space; convert to mm to compare with true bias)
    bias_mm = bias_std * xs
    true_bias = m[[f"bias_true_{g}" for g in GPS]].values
    bias_rms_mm = float(np.sqrt(np.mean((bias_mm - true_bias) ** 2)))
    overlap = subspace_overlap(bias_mm, V_true) if k > 0 else 0.0

    return dict(K=K, k=k, recon_rms_mm=recon_rms_mm,
                svol_rms_mm=svol_rms_mm, svol_r=svol_r,
                dV_r=dV_r, dVd3_r=dVd3_r, dV_nrmse=dV_nrmse, dV_d_confound=dV_d_confound,
                drift_x_km=drift_x, drift_y_km=drift_y, drift_d_km=drift_d,
                subspace_overlap=overlap, bias_rms_mm=bias_rms_mm)


def main():
    rows = []
    for K in K_LIST:
        for k in RANK_LIST:
            r = eval_run(K, k)
            if r is not None:
                rows.append(r)
    if not rows:
        print("No completed runs yet (no analyzer CSVs found).")
        return
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "sweep_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nmetrics -> {csv_path}")

    # figures: each metric vs model rank, one line per K, vertical marker at k=K
    panels = [
        ("recon_rms_mm", "reconstruction RMS [mm]  (fidelity)", False),
        ("svol_rms_mm", "volcanic-signal recovery RMS [mm] (interpretability)", False),
        ("dVd3_r", "identifiable dV/d^3 correlation (higher=better)", False),
        ("subspace_overlap", "bias-subspace overlap  (1=full)  (architecture)", False),
        ("drift_x_km", "source x-drift std [km]  (identifiability)", False),
        ("bias_rms_mm", "bias recovery RMS [mm]", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title, _) in zip(axes.ravel(), panels):
        for K in K_LIST:
            d = df[df.K == K].sort_values("k")
            ax.plot(d.k, d[col], "o-", label=f"K_true={K}")
            ax.axvline(K, color=f"C{K_LIST.index(K)}", ls=":", lw=1, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("model residual rank k")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=9)
    fig.suptitle("PILA synthetic-control rank sweep — elbow expected at k = K_true "
                 "(dotted verticals)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "sweep_results.png")
    fig.savefig(out, dpi=120)
    print(f"figure   -> {out}")


if __name__ == "__main__":
    main()
