"""
Visualise the synthetic-control rank sweep (the table in figures/sweep_metrics.csv):

  left panel  : reconstruction RMSE   [mm]   (fidelity)
  right panel : dV NRMSE              [-]    (interpretability)

x-axis is the PILA residual rank k; one line per ground-truth intrinsic bias rank
K in {1, 2, 4}. Each curve's ground-truth rank (k = K) is highlighted with a
matching vertical line and an enlarged star marker -- the point where the residual
first has enough spatial degrees of freedom to absorb the whole synthetic bias.

Run:
  conda run -n mres python rebuttal/synthetic_control/plot_sweep_table.py
  ... --noise      # visualise the pink+white noise variant (sweep_metrics_noise.csv)
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "figures")

# Small multiples: K identity is carried by the column title, so every curve
# uses the same colour. A red dotted line marks the ground-truth rank k = K, and
# the largest dot marks the minimum of each curve (its best k).
LINE_COLOR = "#33638D"   # muted blue for all curves
HL_COLOR = "red"         # red for the ground-truth rank line
PANELS = [
    ("recon_rms_mm", "Reconstruction\nRMSE (mm)"),
    ("dV_nrmse", r"$\Delta V$ NRMSE"),
]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", action="store_true",
                    help="use sweep_metrics_noise.csv instead of sweep_metrics.csv")
    ap.add_argument("--highlight", choices=["rank", "min", "both", "none"],
                    default="both",
                    help="mark the ground-truth rank k=K, the curve minimum (star), "
                         "both, or none")
    ap.add_argument("--rank-style", choices=["line", "band", "point"], default="line",
                    help="draw the ground-truth rank as a dotted line, a shaded band, "
                         "or a marker on the curve")
    ap.add_argument("--rank-marker", default="D", help="marker for rank-style=point")
    ap.add_argument("--rank-color", default="red", help="colour of the rank highlight")
    ap.add_argument("--rank-size", type=float, default=13, help="rank point marker size")
    ap.add_argument("--min-marker", default="*", help="marker for the curve minimum")
    ap.add_argument("--min-color", default=LINE_COLOR, help="colour of the minimum marker")
    ap.add_argument("--min-size", type=float, default=20, help="minimum marker size")
    ap.add_argument("--ypad", type=float, default=0.3,
                    help="fractional y-margin per panel; larger flattens the elbow wiggle")
    ap.add_argument("--ranks", type=int, nargs="+", default=None,
                    help="which residual ranks k to display (default: all k > 0)")
    args = ap.parse_args()
    # show_rank = args.highlight in ("rank", "both")
    # show_min = args.highlight in ("min", "both")
    show_rank = True
    show_min = False
    sfx = "_noise" if args.noise else ""
    csv = os.path.join(OUT_DIR, f"sweep_metrics{sfx}.csv")
    df = pd.read_csv(csv)
    args.ranks = (1,2,4,8)
    if args.ranks:
        df = df[df.k.isin(args.ranks)]     # user-selected residual ranks
    else:
        df = df[df.k > 0]                  # default: drop pure-physics point (k = 0)
    K_list = sorted(df.K.unique())
    kticks = sorted(df.k.unique())

    # small multiples: rows = metric, columns = ground-truth rank K. Each cell
    # autoscales its own y so the elbow fills the panel; k = K marked in red.
    fig, axes = plt.subplots(len(PANELS), len(K_list), figsize=(15, 5),
                             sharex=True, sharey=False)
    for r, (col, ylabel) in enumerate(PANELS):
        for c, K in enumerate(K_list):
            ax = axes[r, c]
            d = df[df.K == K].sort_values("k")
            ax.plot(d.k, d[col], "-o", color=LINE_COLOR, markersize=12, linewidth=2, zorder=3)
            # highlight the ground-truth rank k = K (dotted line, band, or a point)
            if show_rank:
                # if args.rank_style == "band":
                #     ax.axvspan(K - 0.4, K + 0.4, color=args.rank_color, alpha=0.35, zorder=1)
                # elif args.rank_style == "point":
                hit = d[d.k == K]
                if not hit.empty:
                    ax.plot(hit.k, hit[col], marker="o",
                            markersize=12, color="red",
                            linestyle="none", zorder=4)
                # else:
                ax.axvline(K, color=args.rank_color, ls=":", lw=2.5, alpha=1.0, zorder=1)
            # marker at the minimum of the curve (its best k)
            if show_min:
                best = d.loc[d[col].idxmin()]
                ax.plot(best.k, best[col], marker="o",
                        markersize=12, color="#33638D",
                        markeredgecolor="white", markeredgewidth=1.2,
                        linestyle="none", zorder=4)
            ax.set_xticks(kticks)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(labelsize=17)
            ax.margins(y=args.ypad)
            ax.grid(alpha=0.3, zorder=0)
            if r == 0:
                ax.set_title(f"Intrinsic rank $r'={K}$", fontsize=18)
            if r == len(PANELS) - 1:
                ax.set_xlabel("PILA residual rank $r$", fontsize=18)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=17)
    # fig.suptitle("PILA rank sweep" + (" (noise)" if args.noise else ""), fontsize=15)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"sweep_rmse_dvnrmse{sfx}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
