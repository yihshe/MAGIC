"""
Inspect the synthetic Mogi datasets: format parity vs real data, bias/physics
orthogonality, intrinsic rank in raw and standardized (residual) space, and a
visual figure (dV ramp, signal decomposition, bias rank, noise spectrum).

Run:  python rebuttal/inspect_synth.py                 # clean datasets  K{1,2,4}
      python rebuttal/inspect_synth.py --suffix _realistic   # pink+white noise
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "data/processed/mogi_synth")
FIG_DIR = os.path.join(REPO, "rebuttal/figures")
RANKS = [1, 2, 4]
STATIONS = list(json.load(open(os.path.join(REPO, "configs/station_info.json"))).keys())
GPS = [f"{d}_{s}" for d in ("ux", "uy", "uz") for s in STATIONS]


def load_components(K, suffix):
    """Return (dates, dV_true, S_vol, bias, noise) in raw mm on the full timeline."""
    d = os.path.join(ROOT, f"K{K}{suffix}")
    gt = pd.read_csv(os.path.join(d, "groundtruth.csv"))
    test = pd.read_csv(os.path.join(d, "test.csv"))          # full ordered timeline
    xs = np.load(os.path.join(d, "train_x_scale.npy"))
    xm = np.load(os.path.join(d, "train_x_mean.npy"))
    m = test.merge(gt, on="date", how="inner").sort_values("date").reset_index(drop=True)
    observed = m[GPS].values * xs + xm                        # de-standardize -> mm
    svol = m[[f"svol_true_{g}" for g in GPS]].values
    bias = m[[f"bias_true_{g}" for g in GPS]].values
    noise = observed - svol - bias                            # exactly the injected eps
    return m["date"].values, m["dV_true_m3"].values, svol, bias, noise, xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="'' (clean) or '_realistic' (pink+white)")
    args = ap.parse_args()
    sfx = args.suffix
    os.makedirs(FIG_DIR, exist_ok=True)

    real = pd.read_csv(os.path.join(REPO, "data/processed/mogi/train.csv"))
    syn = pd.read_csv(os.path.join(ROOT, f"K4{sfx}", "train.csv"))
    print(f"== dataset K*{sfx} ==")
    print("cols match real :", list(real.columns) == list(syn.columns))
    print("real  std/range :", round(real.iloc[:, :36].values.std(), 3),
          (round(real.iloc[:, :36].values.min(), 2), round(real.iloc[:, :36].values.max(), 2)))
    print("synth std/range :", round(syn.iloc[:, :36].values.std(), 3),
          (round(syn.iloc[:, :36].values.min(), 2), round(syn.iloc[:, :36].values.max(), 2)))

    print("\n== rank / orthogonality / magnitudes ==")
    comps = {}
    for K in RANKS:
        d = os.path.join(ROOT, f"K{K}{sfx}")
        basis = np.load(os.path.join(d, "basis.npy"), allow_pickle=True).item()
        V, mhat = basis["V"], basis["m_hat"]
        dates, dV, svol, bias, noise, xs = load_components(K, sfx)
        comps[K] = (dates, dV, svol, bias, noise)
        s_raw = np.linalg.svd(bias - bias.mean(0), compute_uv=False)
        s_std = np.linalg.svd((bias / xs) - (bias / xs).mean(0), compute_uv=False)
        rms = lambda A: float(np.sqrt(np.mean(A ** 2)))
        print(f"K={K}: max|V^T m_hat|={np.max(np.abs(V.T @ mhat)):.1e} "
              f"rank(raw)={int(np.sum(s_raw > 1e-6 * s_raw[0]))} "
              f"rank(std)={int(np.sum(s_std > 1e-6 * s_std[0]))} | "
              f"vol={rms(svol):.2f} bias={rms(bias):.2f} noise={rms(noise):.2f} mm | "
              f"drivers={basis['drivers']}")

    has_noise = any(np.abs(comps[K][4]).max() > 1e-9 for K in RANKS)
    ncol = 4 if has_noise else 3
    fig, axes = plt.subplots(len(RANKS), ncol, figsize=(4.2 * ncol, 3.2 * len(RANKS)))
    for row, K in enumerate(RANKS):
        dates, dV, svol, bias, noise = comps[K]
        t = np.arange(len(dV))
        j = GPS.index("uz_AKGG")

        ax = axes[row, 0]
        ax.plot(t, dV, "k")
        ax.set_title(f"K={K}: true dV(t)"); ax.set_ylabel("dV [m$^3$]")

        ax = axes[row, 1]
        ax.plot(t, svol[:, j] + bias[:, j] + noise[:, j], color="0.75", lw=0.5,
                label="observed")
        ax.plot(t, bias[:, j], "C1", lw=0.9, label="bias")
        ax.plot(t, svol[:, j], "C0", lw=1.4, label="volcanic")
        ax.set_title(f"K={K}: decomposition (uz_AKGG)"); ax.set_ylabel("mm")
        if row == 0:
            ax.legend(fontsize=8, loc="upper left")

        ax = axes[row, 2]
        s = np.linalg.svd(bias - bias.mean(0), compute_uv=False)
        ax.semilogy(range(1, 9), s[:8] / s[0], "o-")
        ax.axvline(K + 0.5, color="r", ls="--", lw=1)
        ax.set_title(f"K={K}: bias singular values (rank={K})")
        ax.set_xlabel("index"); ax.set_ylabel("s / s$_1$")

        if has_noise:
            ax = axes[row, 3]
            n = noise.shape[0]
            freqs = np.fft.rfftfreq(n, d=1.0)          # cycles/day
            psd = np.mean(np.abs(np.fft.rfft(noise, axis=0)) ** 2, axis=1)
            sl = slice(1, len(freqs))
            ax.loglog(freqs[sl], psd[sl], color="C3", lw=0.7, alpha=0.8)
            ref = psd[1] * (freqs[sl] / freqs[1]) ** -1.0
            ax.loglog(freqs[sl], ref, "k--", lw=1.2, label="1/f (pink)")
            ax.set_title(f"K={K}: noise power spectrum")
            ax.set_xlabel("frequency [1/day]"); ax.set_ylabel("power")
            if row == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f"Synthetic Mogi data{sfx or ' (clean)'} — "
                 f"controllable-rank bias over a known dV(t) ramp", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"synth_inspection{sfx}.png")
    fig.savefig(out, dpi=120)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
