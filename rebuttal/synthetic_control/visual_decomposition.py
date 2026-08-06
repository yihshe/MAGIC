"""
Decomposition of the *generated* synthetic-control datasets, in the style of
visuals/analysis_mogi.py.

This figure is about DATA GENERATION ONLY -- no training / no learned model. For
each ground-truth bias rank K in {1, 2, 4} it reads that dataset's
groundtruth.csv and shows, per station, three columns:

  col 1  svol_true            Mogi forward (volcanic signal)     X_F
  col 2  bias_true            generated low-rank residual        Delta'
  col 3  svol_true+bias_true  synthetic observation (the sum)    X'

Two figure kinds are produced:
  * all-stations : 12 x 3 grid, one unified y-scale across the whole figure.
  * single       : one station (default AV13), 1 x 3, shared y-scale.
Panel titles are just the station name (column labels are added by hand later).
The ~9-month 5-95% inflation window (from meta.json) is shaded pink.

Run:
  conda run -n mres python rebuttal/synthetic_control/visual_decomposition.py
  ... --K 4 --station AV08 --noise
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data/processed/mogi_synth")
OUT_DIR = os.path.join(HERE, "figures")

station_info = json.load(open(os.path.join(REPO, "configs/station_info.json")))
STATIONS = list(station_info.keys())
DIRECTIONS = ["ux", "uy", "uz"]
dirs = {"ux": "East (mm)", "uy": "North (mm)", "uz": "Vertical (mm)"}
COMPONENTS = ["svol_true", "bias_true", "obs"]   # X_F | Delta' | X'


def data_ylim(values, pad_frac=0.06):
    """Asymmetric limits hugging the actual data range (e.g. -5..15), not 0-centred."""
    lo, hi = float(np.min(values)), float(np.max(values))
    span = max(hi - lo, 1e-6)
    return lo - pad_frac * span, hi + pad_frac * span


def sym_ylim(values, pad=1.1):
    m = max(float(np.abs(values).max()) * pad, 1e-6)
    return -m, m


def load(K, dsuf):
    dd = f"K{K}{dsuf}"
    gt_path = os.path.join(DATA, dd, "groundtruth.csv")
    if not os.path.exists(gt_path):
        return None
    gt = pd.read_csv(gt_path)
    gt["date"] = pd.to_datetime(gt["date"], format="%Y.%m.%d")
    gt = gt.sort_values("date").reset_index(drop=True)
    for d in DIRECTIONS:
        for s in STATIONS:
            g = f"{d}_{s}"
            gt[f"obs_{g}"] = gt[f"svol_true_{g}"] + gt[f"bias_true_{g}"]
    meta = json.load(open(os.path.join(DATA, dd, "meta.json")))
    infl = (pd.to_datetime(meta["transient_start_date"], format="%Y.%m.%d"),
            pd.to_datetime(meta["transient_end_date"], format="%Y.%m.%d"))
    return gt, infl


# Shared per-panel style so a single gross-figure panel is zoom-comparable to
# the individual figure. The gross figure is 12x3 in figsize (30,60) -> each
# panel is 10x5, so the individual figure (1x3, width 30) uses height 5 to match.
LABEL_FS = 32
TICK_FS = 26
PANEL_W, PANEL_H = 10, 5


def draw_panel(ax, gt, prefix, g, infl, ylim, station):
    ax.axvspan(infl[0], infl[1], color="pink", alpha=0.4)
    sns.scatterplot(x=gt["date"], y=gt[f"{prefix}_{g}"], ax=ax,
                    s=30, alpha=0.75, color="blue", edgecolor="none")
    ax.set_title(station, fontsize=LABEL_FS)
    ax.set_xlabel("Date", fontsize=LABEL_FS)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    for tick in ax.get_xticklabels():
        tick.set_rotation(-20)


def render_all(K, dsuf, direction):
    got = load(K, dsuf)
    if got is None:
        print(f"[skip] no groundtruth for K{K}{dsuf}")
        return
    gt, infl = got
    cols = [f"{p}_{direction}_{s}" for p in COMPONENTS for s in STATIONS]
    ylim = sym_ylim(gt[cols].values)           # symmetric, unified across whole figure
    fig, axs = plt.subplots(12, 3, figsize=(3 * PANEL_W, 12 * PANEL_H))
    for i, station in enumerate(STATIONS):
        for j, prefix in enumerate(COMPONENTS):
            ax = axs[i, j]
            draw_panel(ax, gt, prefix, f"{direction}_{station}", infl, ylim, station)
            ax.set_ylabel(dirs[direction], fontsize=LABEL_FS)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{direction}_datagen_all_K{K}{dsuf}.png")
    plt.savefig(out, dpi=300); plt.close(fig)
    print(f"[ok] all-stations K{K}{dsuf} {direction} -> {out}")


def render_single(K, dsuf, direction, station):
    got = load(K, dsuf)
    if got is None:
        print(f"[skip] no groundtruth for K{K}{dsuf}")
        return
    gt, infl = got
    cols = [f"{p}_{direction}_{station}" for p in COMPONENTS]
    ylim = data_ylim(gt[cols].values)           # shared across the 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(3 * PANEL_W, PANEL_H))
    for j, prefix in enumerate(COMPONENTS):
        draw_panel(axs[j], gt, prefix, f"{direction}_{station}", infl, ylim, station)
        axs[j].set_ylabel(dirs[direction], fontsize=LABEL_FS)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{direction}_datagen_{station}_K{K}{dsuf}.png")
    plt.savefig(out, dpi=300); plt.close(fig)
    print(f"[ok] single {station} K{K}{dsuf} {direction} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=None,
                    help="single dataset rank (default: all of 1, 2, 4)")
    ap.add_argument("--station", default="AV13", choices=STATIONS,
                    help="station for the single-station figure")
    ap.add_argument("--noise", action="store_true", help="use the pink+white noise variant")
    ap.add_argument("--directions", nargs="+", default=DIRECTIONS, choices=DIRECTIONS,
                    help="components to render (default: ux uy uz)")
    args = ap.parse_args()

    dsuf = "_realistic" if args.noise else ""
    K_list = [args.K] if args.K is not None else [1, 2, 4]
    print("columns: svol_true (X_F) | bias_true (Delta') | svol_true+bias_true (X')")
    for K in K_list:
        for d in args.directions:
            render_all(K, dsuf, d)
            render_single(K, dsuf, d, args.station)


if __name__ == "__main__":
    main()
