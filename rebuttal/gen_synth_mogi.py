"""
Synthetic-control data generator for the PILA rank ablation (rebuttal, R1).

Goal
----
Produce GNSS-like datasets in which the *ground truth is fully known* so we can
show, in a controlled setting, that PILA's low-rank neural residual recovers the
true volcanic transient dV(t) **iff** its rank k >= K_true, the intrinsic rank of
an injected systematic bias. This isolates the benefit of the low-rank inductive
bias itself from the specifics of the real Akutan case study.

Generative model (per timestamp t, raw units = mm)
--------------------------------------------------
    u(t) = S_vol(t) + b(t) + eps(t)

  S_vol(t) : volcanic signal from a *fixed* Mogi source (xcen, ycen, d), driven
             by an arctan inflation ramp dV(t). Mogi is linear in dV and the
             geometry is fixed, so S_vol(t) = dV(t) * m_hat lives in a SINGLE
             direction m_hat in R^36.

  b(t)     : systematic bias of controllable intrinsic rank K,
                 b(t) = A * sum_{j=1..K} v_j * g_j(t)              (B = G V^T)
             - v_j : orthonormal spatial patterns, each ORTHOGONAL to m_hat
                     (projected out before QR) so the bias is unreachable by the
                     physics and cannot silently corrupt dV.
             - g_j : distinct temporal drivers (Bevis trajectory-model terms;
                     see DRIVERS). K distinct drivers => rank exactly K. Add more
                     entries to DRIVERS to reach higher K ("more complexity").
             - A   : single scalar so per-entry RMS(b) hits `bias_rms_mm` for
                     EVERY K (total bias energy held constant across K).

  eps(t)   : optional observation noise = white + flicker(pink). Per component,
             spatially independent => spatially full-rank => sets the irreducible
             reconstruction floor (NOT correctable by any low-rank residual).

Magnitude calibration (from the real rank-4 PILA decomposition, in mm RMS):
    physics 2.82 | bias 2.60 | noise(recon resid) 3.14 | target 5.39
    => bias/physics ~ 0.9, noise/physics ~ 1.1.  Defaults below match this.

How the true rank is controlled (and verified)
----------------------------------------------
  rank(B) = rank(G V^T) = K when V has K independent columns (orthonormal via QR)
  and G has K independent columns (distinct drivers). The script SVDs the
  centered bias matrix and asserts exactly K singular values survive.

Run:
    python rebuttal/gen_synth_mogi.py                       # clean, K in {1,2,4}
    python rebuttal/gen_synth_mogi.py --noise realistic     # + pink+white noise
    python rebuttal/gen_synth_mogi.py --ranks 1 2 4 6 8     # higher-rank bias
"""
import os
import sys
import json
import argparse
import datetime
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from physics.mogi.mogi import Mogi  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration (all knobs in one place)
# --------------------------------------------------------------------------- #
CFG = dict(
    seed=0,
    start_date=(2006, 1, 1),
    n_days=6574,                  # 2006-01-01 .. 2023-12-31 (daily)
    ranks=[1, 2, 4],

    # Fixed Mogi source (physical; within configs/mogi_paras.json ranges)
    xcen_km=2.0, ycen_km=0.0, d_km=5.0,

    # Inflation ramp: dV(t) = dV_max * normalize(atan((t - onset)/width))
    ramp_onset_frac=0.40,         # onset (time shift) as fraction of timeline
    ramp_width_frac=0.06,         # transition width as fraction of timeline

    # Target per-entry RMS magnitudes (mm), calibrated to real decomposition.
    vol_rms_mm=2.5,               # volcanic (Mogi) signal
    bias_rms_mm=2.5,              # systematic bias (held constant across K)

    # Observation noise per component (mm RMS). Presets via --noise.
    noise_white_mm=0.0,
    noise_flicker_mm=0.0,
    flicker_beta=1.0,             # PSD ~ 1/f^beta (1=pink/flicker, 2=brownian)

    # Split: random 85/15 train/valid; test = full ordered timeline (for dV(t)).
    split_mode="random",
    valid_frac=0.15,

    out_root="data/processed/mogi_synth",
    poisson_nu=0.25,
)

NOISE_PRESETS = {
    "none":      dict(noise_white_mm=0.0, noise_flicker_mm=0.0),
    "low":       dict(noise_white_mm=0.5, noise_flicker_mm=0.5),
    "moderate":  dict(noise_white_mm=1.0, noise_flicker_mm=1.0),
    "realistic": dict(noise_white_mm=1.8, noise_flicker_mm=2.2),  # ~3.1mm total, real floor
}

# Ordered temporal-driver dictionary (Bevis trajectory-model terms + harmonics).
# K activates the first K entries; extend this list to reach higher ranks.
T_ANN = 365.25
DRIVERS = [
    ("linear",         lambda t: t.astype(np.float64)),
    ("annual_sin",     lambda t: np.sin(2 * np.pi * t / T_ANN)),
    ("annual_cos",     lambda t: np.cos(2 * np.pi * t / T_ANN)),
    ("semiannual_sin", lambda t: np.sin(2 * np.pi * t / (T_ANN / 2))),
    ("semiannual_cos", lambda t: np.cos(2 * np.pi * t / (T_ANN / 2))),
    ("terannual_sin",  lambda t: np.sin(2 * np.pi * t / (T_ANN / 3))),
    ("terannual_cos",  lambda t: np.cos(2 * np.pi * t / (T_ANN / 3))),
    ("quadratic",      lambda t: (t.astype(np.float64)) ** 2),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def unit_rms(v):
    v = v - v.mean()
    r = np.sqrt(np.mean(v ** 2))
    return v / r if r > 0 else v


def colored_series(n, beta, rng):
    """Length-n zero-mean unit-std series with PSD ~ 1/f^beta (Timmer-Koenig)."""
    f = np.fft.rfftfreq(n)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (-beta / 2.0)
    phases = rng.uniform(0, 2 * np.pi, len(f))
    spec = amp * np.exp(1j * phases)
    x = np.fft.irfft(spec, n=n)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x


def load_stations():
    si = json.load(open(os.path.join(REPO, "configs/station_info.json")))
    names = list(si.keys())
    x_km = np.array([si[k]["xE"] for k in names])
    y_km = np.array([si[k]["yN"] for k in names])
    return names, x_km, y_km


def mogi_forward(x_km, y_km, xcen_km, ycen_km, d_km, dV_m3, nu):
    x = torch.tensor(x_km * 1000.0, dtype=torch.float64)
    y = torch.tensor(y_km * 1000.0, dtype=torch.float64)
    model = Mogi(x, y, nu=nu)
    N = len(dV_m3)
    xcen = torch.full((N,), xcen_km * 1000.0, dtype=torch.float64)
    ycen = torch.full((N,), ycen_km * 1000.0, dtype=torch.float64)
    d = torch.full((N,), d_km * 1000.0, dtype=torch.float64)
    dV = torch.tensor(dV_m3, dtype=torch.float64)
    return model.run(xcen, ycen, d, dV).numpy()  # (N,36) mm


def make_dates(start, n):
    d0 = datetime.date(*start)
    dates = [d0 + datetime.timedelta(days=int(i)) for i in range(n)]
    date_str = [f"{d.year:04d}.{d.month:02d}.{d.day:02d}" for d in dates]
    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=np.float64)
    return date_str, np.sin(2 * np.pi * doy / T_ANN), np.cos(2 * np.pi * doy / T_ANN)


def build_spatial_basis(m_hat, K, rng):
    R = rng.standard_normal((36, K))
    R = R - np.outer(m_hat, m_hat @ R)      # project out volcanic direction
    Q, _ = np.linalg.qr(R)
    V = Q[:, :K]
    for j in range(K):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] *= -1
    return V


def rank_check(B, tol_ratio=1e-6):
    Bc = B - B.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Bc, compute_uv=False)
    eff = int(np.sum(s > tol_ratio * s[0])) if s[0] > 0 else 0
    return s, eff


# --------------------------------------------------------------------------- #
# Core generation for one K
# --------------------------------------------------------------------------- #
def generate_for_K(K, cfg, out_suffix):
    assert K <= len(DRIVERS), f"K={K} exceeds available drivers ({len(DRIVERS)})"
    rng = np.random.default_rng(cfg["seed"] + K)
    names, x_km, y_km = load_stations()
    N = cfg["n_days"]
    t = np.arange(N, dtype=np.float64)

    # volcanic direction m_hat (fixed source, unit dV)
    m_raw = mogi_forward(x_km, y_km, cfg["xcen_km"], cfg["ycen_km"],
                         cfg["d_km"], np.ones(1), cfg["poisson_nu"])[0]
    m_hat = m_raw / np.linalg.norm(m_raw)

    # inflation ramp dV(t) -> S_vol, scaled to target volcanic RMS
    onset = cfg["ramp_onset_frac"] * N
    width = cfg["ramp_width_frac"] * N
    ramp = np.arctan((t - onset) / width)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    S_vol_unit = mogi_forward(x_km, y_km, cfg["xcen_km"], cfg["ycen_km"],
                              cfg["d_km"], ramp, cfg["poisson_nu"])
    vol_scale = cfg["vol_rms_mm"] / np.sqrt(np.mean(S_vol_unit ** 2))
    dV = ramp * vol_scale
    S_vol = S_vol_unit * vol_scale

    # systematic bias of intrinsic rank K
    V = build_spatial_basis(m_hat, K, rng)
    G = np.stack([unit_rms(DRIVERS[j][1](t)) for j in range(K)], axis=1)
    B_unit = G @ V.T
    B = B_unit * (cfg["bias_rms_mm"] / np.sqrt(np.mean(B_unit ** 2)))
    svals, eff_rank = rank_check(B)
    assert eff_rank == K, f"K={K}: expected rank {K}, got {eff_rank}"

    # observation noise (white + flicker), per component, spatially independent
    E = np.zeros((N, 36))
    if cfg["noise_white_mm"] > 0:
        E += cfg["noise_white_mm"] * rng.standard_normal((N, 36))
    if cfg["noise_flicker_mm"] > 0:
        E += cfg["noise_flicker_mm"] * np.stack(
            [colored_series(N, cfg["flicker_beta"], rng) for _ in range(36)], axis=1)

    U = S_vol + B + E

    # dates + split
    date_str, sin_d, cos_d = make_dates(cfg["start_date"], N)
    idx = np.arange(N)
    if cfg["split_mode"] == "random":
        perm = rng.permutation(N)
        n_val = int(cfg["valid_frac"] * N)
        valid_idx, train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    else:
        n_val = int(cfg["valid_frac"] * N)
        train_idx, valid_idx = idx[:N - n_val], idx[N - n_val:]
    test_idx = idx

    # standardize with TRAIN per-column stats
    x_mean = U[train_idx].mean(axis=0)
    x_scale = U[train_idx].std(axis=0)
    x_scale[x_scale == 0] = 1.0
    U_std = (U - x_mean) / x_scale

    # write
    import pandas as pd
    out_dir = os.path.join(REPO, cfg["out_root"], f"K{K}{out_suffix}")
    os.makedirs(out_dir, exist_ok=True)
    cols = ([f"ux_{n}" for n in names] + [f"uy_{n}" for n in names]
            + [f"uz_{n}" for n in names])

    def write_split(name, ids):
        df = pd.DataFrame(U_std[ids], columns=cols)
        df["date"] = [date_str[i] for i in ids]
        df["sin_date"] = sin_d[ids]
        df["cos_date"] = cos_d[ids]
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    write_split("train", train_idx)
    write_split("valid", valid_idx)
    write_split("test", test_idx)
    np.save(os.path.join(out_dir, "train_x_mean.npy"), x_mean)
    np.save(os.path.join(out_dir, "train_x_scale.npy"), x_scale)

    gt = pd.DataFrame({"date": date_str, "dV_true_m3": dV,
                       "xcen_true_km": cfg["xcen_km"], "ycen_true_km": cfg["ycen_km"],
                       "d_true_km": cfg["d_km"]})
    for c, col in enumerate(cols):
        gt[f"bias_true_{col}"] = B[:, c]
        gt[f"svol_true_{col}"] = S_vol[:, c]
    gt.to_csv(os.path.join(out_dir, "groundtruth.csv"), index=False)
    np.save(os.path.join(out_dir, "basis.npy"),
            {"V": V, "m_hat": m_hat, "drivers": [DRIVERS[j][0] for j in range(K)]},
            allow_pickle=True)

    # std-space overlap of bias with the physics direction (leakage indicator):
    # fraction of bias energy lying along m_hat AFTER per-column standardization.
    # Raw-space orthogonality (V^T m_hat = 0) does NOT survive diagonal scaling,
    # so this is generally > 0 and is what governs dV/param leakage at low rank.
    m_std = (m_raw / x_scale)
    m_std /= np.linalg.norm(m_std)
    B_std = (B / x_scale)
    bias_phys_overlap = float(np.sqrt(np.sum((B_std @ m_std) ** 2)
                                      / np.sum(B_std ** 2)))

    noise_rms = float(np.sqrt(np.mean(E ** 2))) if np.any(E) else 0.0
    meta = dict(
        K_true=K, effective_rank=eff_rank,
        singular_values=svals[:min(8, len(svals))].tolist(),
        drivers=[DRIVERS[j][0] for j in range(K)],
        n_days=N, n_train=len(train_idx), n_valid=len(valid_idx), n_test=len(test_idx),
        ramp_onset_days=float(onset), ramp_width_days=float(width),
        realized_vol_rms_mm=float(np.sqrt(np.mean(S_vol ** 2))),
        realized_bias_rms_mm=float(np.sqrt(np.mean(B ** 2))),
        realized_noise_rms_mm=noise_rms,
        bias_to_vol_ratio=float(np.sqrt(np.mean(B ** 2)) / np.sqrt(np.mean(S_vol ** 2))),
        noise_to_vol_ratio=(noise_rms / np.sqrt(np.mean(S_vol ** 2))) if noise_rms else 0.0,
        bias_phys_overlap_frac=bias_phys_overlap,
        dV_max_m3=float(dV.max()),
        noise_white_mm=cfg["noise_white_mm"], noise_flicker_mm=cfg["noise_flicker_mm"],
        source_km=dict(xcen=cfg["xcen_km"], ycen=cfg["ycen_km"], d=cfg["d_km"]),
        split_mode=cfg["split_mode"], seed=cfg["seed"],
    )
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    return meta, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=None)
    ap.add_argument("--noise", choices=list(NOISE_PRESETS), default="none")
    ap.add_argument("--suffix", default=None,
                    help="output dir suffix; default '' for none, '_<noise>' otherwise")
    args = ap.parse_args()

    cfg = dict(CFG)
    if args.ranks:
        cfg["ranks"] = args.ranks
    cfg.update(NOISE_PRESETS[args.noise])
    suffix = args.suffix if args.suffix is not None else ("" if args.noise == "none" else f"_{args.noise}")

    print(f"Generating: ranks={cfg['ranks']} noise={args.noise} "
          f"(white={cfg['noise_white_mm']} flicker={cfg['noise_flicker_mm']} mm) suffix='{suffix}'")
    for K in cfg["ranks"]:
        meta, out_dir = generate_for_K(K, cfg, suffix)
        print(f"  K={K}: rank={meta['effective_rank']} "
              f"vol={meta['realized_vol_rms_mm']:.2f} bias={meta['realized_bias_rms_mm']:.2f} "
              f"noise={meta['realized_noise_rms_mm']:.2f} mm | "
              f"bias/vol={meta['bias_to_vol_ratio']:.2f} noise/vol={meta['noise_to_vol_ratio']:.2f} "
              f"bias||phys={meta['bias_phys_overlap_frac']:.3f} -> {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
