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

  b(t)     : systematic bias = K EQUAL-energy common-mode components. Each is one
             orthonormal spatial pattern (a column of V) carrying its own realistic
             Bevis trajectory (linear + annual + semiannual, realistic relative
             sizes, random phases):
                 b(t) = sum_{m=1..K} s_m(t) * V[:,m]           (B = S V^T)
             - V   : K orthonormal spatial patterns, each ORTHOGONAL to m_hat
                     (projected out before QR) so the bias cannot corrupt dV.
             - s_m : K temporal signals, ORTHOGONALIZED (QR) so they are mutually
                     uncorrelated with equal energy => B has a FLAT singular
                     spectrum (all K singular values equal) => a sharp elbow at
                     k=K. All three frequencies live in every mode.
             - scaled so per-entry RMS(B) = bias_total_mm for EVERY K (total bias
                     held constant across K; only the spatial rank changes).
             Why spatial rank: PILA's residual is k fixed spatial patterns with
             FREE time-varying coefficients c(t), so it can represent b iff
             k >= dim(span V) = K, regardless of how many frequencies b contains.
             NOTE: temporal orthogonality (the QR step) is what makes the spectrum
             flat; without it the modes are correlated and the elbow smears out.

  eps(t)   : optional observation noise = white + flicker(pink). Per component,
             spatially independent => spatially full-rank => sets the irreducible
             reconstruction floor (NOT correctable by any low-rank residual).

Magnitude calibration (from the real rank-4 PILA decomposition, in mm RMS):
    physics 2.82 | bias 2.60 | noise(recon resid) 3.14 | target 5.39
    => bias/physics ~ 0.9.  vol_rms_mm and bias_total_mm default to ~1:1.

How the true rank is controlled (and verified)
----------------------------------------------
  B (T x 36) = S V^T with S (T x K) orthonormal columns (QR) and V (36 x K)
  orthonormal columns. Both factors orthonormal => B has exactly K nonzero and
  EQUAL singular values, i.e. spatial rank = K with a flat spectrum. The script
  SVDs the centered bias matrix and ASSERTS exactly K singular values survive --
  so the ground-truth spatial rank is guaranteed to be K, and (being flat) gives
  the sharpest possible reconstruction elbow at k=K.

Run:
    python rebuttal/synthetic_control/gen_synth_mogi.py                    # clean
    python rebuttal/synthetic_control/gen_synth_mogi.py --noise realistic  # +noise
"""
import os
import sys
import json
import argparse
import datetime
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))     # rebuttal/synthetic_control
REPO = os.path.dirname(os.path.dirname(HERE))         # repo root
sys.path.insert(0, REPO)
from physics.mogi.mogi import Mogi  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration (all knobs in one place)
# --------------------------------------------------------------------------- #
CFG = dict(
    seed=0,
    start_date=(2006, 1, 1),
    n_days=2922,                  # ~8 years: enough snapshots for the residual to be
                                  # identifiable (3 yr under-trains it, smearing the elbow)
    ranks=[1, 2, 4],

    # Fixed Mogi source (physical; within configs/mogi_paras.json ranges)
    xcen_km=2.0, ycen_km=0.0, d_km=5.0,

    # Inflation ramp on a nonzero baseline: dV(t) = dV_base + transient(t),
    # transient(t) = A * normalize(atan((t - onset)/width)). The baseline keeps
    # the source always deforming so the geometry stays identifiable throughout
    # (only the relative/transient change is scientifically meaningful). Set
    # dV_base_frac=0 to recover a ramp starting from dV=0.
    ramp_onset_frac=0.40,         # onset (time shift) as fraction of the record
    ramp_width_day=30,            # atan width in days (10-90% rise ~6*width ~ 6 months)
    dV_base_frac=1.0,             # baseline as a multiple of the transient amplitude
                                  # (=1 => dV doubles across the step: prominent)

    # Volcanic transient VARIATION (temporal std of the Mogi signal, mm) -- the
    # part that survives standardization and competes with the bias.
    vol_rms_mm=2.5,

    # Systematic bias = K EQUAL-energy common-mode components, each a realistic
    # Bevis trajectory (linear + annual + semiannual, realistic relative sizes,
    # random phases). Equal energy => flat singular spectrum => sharp elbow at
    # k=K; total bias RMS is held at bias_total_mm (~= vol, 1:1) for every K.
    bias_total_mm=2.5,            # total systematic-bias per-entry RMS (constant across K)
    bias_rel=(1.0, 2.0, 1.0),     # relative sizes: (linear, annual, semiannual)

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

T_ANN = 365.25


def bevis_trajectory(t, rel, rng):
    """One realistic common-mode signal: linear trend + annual + semiannual
    (relative sizes `rel`, random signs/phases). Returned zero-mean, unit-RMS."""
    r_lin, r_ann, r_semi = rel
    yr = (t - t.mean()) / T_ANN
    s = (r_lin * np.sign(rng.standard_normal()) * yr
         + r_ann * np.cos(2 * np.pi * t / T_ANN + rng.uniform(0, 2 * np.pi))
         + r_semi * np.cos(2 * np.pi * t / (T_ANN / 2) + rng.uniform(0, 2 * np.pi)))
    s = s - s.mean()
    return s / np.sqrt(np.mean(s ** 2))


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
    R = R - np.outer(m_hat, m_hat @ R) #TODO not sure what this means     # project out volcanic direction
    Q, _ = np.linalg.qr(R)
    V = Q[:, :K]
    for j in range(K):
        if V[np.argmax(np.abs(V[:, j])), j] < 0: #TODO: not sure what this means.
            V[:, j] *= -1
    return V


def rank_check(B, tol_ratio=1e-6):
    Bc = B - B.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Bc, compute_uv=False)
    eff = int(np.sum(s > tol_ratio * s[0]))#TODO: how is this tol_ratio decided? if s[0] > 0 else 0
    return s, eff


# --------------------------------------------------------------------------- #
# Core generation for one K
# --------------------------------------------------------------------------- #
def generate_for_K(K, cfg, out_suffix):
    rng = np.random.default_rng(cfg["seed"] + K)
    names, x_km, y_km = load_stations()
    N = cfg["n_days"]
    t = np.arange(N, dtype=np.float64)

    # volcanic direction m_hat (fixed source, unit dV)
    m_raw = mogi_forward(x_km, y_km, cfg["xcen_km"], cfg["ycen_km"],
                         cfg["d_km"], np.ones(1), cfg["poisson_nu"])[0]
    m_hat = m_raw / np.linalg.norm(m_raw)

    # transient inflation ramp on a nonzero baseline: dV(t) = dV_base + A*ramp(t).
    # A is set so the TRANSIENT displacement RMS = vol_rms_mm (the meaningful
    # signal, which is what survives standardization); dV_base keeps |dV| away
    # from zero so the geometry stays identifiable even before/at onset.
    onset = cfg["ramp_onset_frac"] * N
    width = cfg["ramp_width_day"]
    ramp = np.arctan((t - onset) / width)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())        # transient shape [0,1]
    S_trans_unit = mogi_forward(x_km, y_km, cfg["xcen_km"], cfg["ycen_km"],
                                cfg["d_km"], ramp, cfg["poisson_nu"])
    # calibrate the transient VARIATION (temporal std) so vol_rms_mm is directly
    # comparable to the bias amplitudes (both are variations about the mean).
    S_trans_var = S_trans_unit - S_trans_unit.mean(0, keepdims=True)
    vol_scale = cfg["vol_rms_mm"] / np.sqrt(np.mean(S_trans_var ** 2))
    dV_base = cfg["dV_base_frac"] * vol_scale
    dV = dV_base + ramp * vol_scale                               # absolute dV(t)
    S_vol = mogi_forward(x_km, y_km, cfg["xcen_km"], cfg["ycen_km"],
                         cfg["d_km"], dV, cfg["poisson_nu"])       # includes baseline

    # transient timing/magnitude ground truth (dV is monotonic: baseline -> plateau)
    dv_lo, dv_hi = float(dV.min()), float(dV.max())
    _cross = lambda f: int(np.argmax(dV >= dv_lo + f * (dv_hi - dv_lo)))
    t_start, t_end = _cross(0.05), _cross(0.95)                   # 5%-95% of the change

    # systematic bias: K EQUAL-energy common-mode components. Each is one
    # orthonormal spatial pattern (column of V, V^T m_hat = 0) carrying its own
    # realistic Bevis trajectory. Equal energy => flat spectrum => sharp elbow at
    # k=K; total per-entry RMS(B) = bias_total_mm for every K.
    V = build_spatial_basis(m_hat, K, rng)                # 36 x K orthonormal, ⟂ m_hat
    S = np.stack([bevis_trajectory(t, cfg["bias_rel"], rng) for _ in range(K)], axis=1)
    S, _ = np.linalg.qr(S)                                # orthonormal temporal modes
    B0 = S @ V.T                                          # equal singular values (flat)
    B = B0 * (cfg["bias_total_mm"] / np.sqrt(np.mean(B0 ** 2)))   # per-entry RMS=bias_total
    svals, eff_rank = rank_check(B)
    assert eff_rank == K, f"K={K}: expected spatial rank {K}, got {eff_rank}"

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
            {"V": V, "m_hat": m_hat, "freqs": ["linear", "annual", "semiannual"]},
            allow_pickle=True)

    # realized physical magnitudes (for realism check), recovered from B by
    # projection onto each Bevis component.
    yr = (t - t.mean()) / T_ANN
    slope_field = (B * yr[:, None]).sum(0) / (yr ** 2).sum()   # mm/yr per station
    lin_rate_rms = float(np.sqrt(np.mean(slope_field ** 2)))
    lin_rate_max = float(np.max(np.abs(slope_field)))

    def _amp(fs, fc):    # per-station amplitude of a sin/cos pair, RMS over stations
        a = (B * fs[:, None]).sum(0) / (fs ** 2).sum()
        b = (B * fc[:, None]).sum(0) / (fc ** 2).sum()
        return float(np.sqrt(np.mean(a ** 2 + b ** 2)))
    ann_amp = _amp(np.sin(2 * np.pi * t / T_ANN), np.cos(2 * np.pi * t / T_ANN))
    semi_amp = _amp(np.sin(2 * np.pi * t / (T_ANN / 2)), np.cos(2 * np.pi * t / (T_ANN / 2)))

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
    # reference magnitude = TRANSIENT variation (what survives standardization and
    # competes with the bias), not the baseline-inflated absolute volcanic signal.
    vol_tr_rms = float(np.sqrt(np.mean((S_vol - S_vol.mean(0)) ** 2)))
    meta = dict(
        K_true=K, effective_rank=eff_rank,
        singular_values=svals[:min(8, len(svals))].tolist(),
        bias_freqs=["linear", "annual", "semiannual"],
        bias_annual_amp_mm=ann_amp, bias_semiannual_amp_mm=semi_amp,
        bias_linear_rate_rms_mm_yr=lin_rate_rms, bias_linear_rate_max_mm_yr=lin_rate_max,
        n_days=N, n_train=len(train_idx), n_valid=len(valid_idx), n_test=len(test_idx),
        ramp_onset_day=float(onset), ramp_width_day=float(width),
        transient_start_day=t_start, transient_end_day=t_end,
        transient_start_date=date_str[t_start], transient_end_date=date_str[t_end],
        transient_duration_days=int(t_end - t_start),
        dV_baseline_m3=dv_lo, dV_plateau_m3=dv_hi, dV_change_m3=float(dv_hi - dv_lo),
        realized_vol_transient_rms_mm=vol_tr_rms,
        realized_vol_abs_rms_mm=float(np.sqrt(np.mean(S_vol ** 2))),
        realized_bias_rms_mm=float(np.sqrt(np.mean(B ** 2))),
        realized_noise_rms_mm=noise_rms,
        bias_to_vol_ratio=float(np.sqrt(np.mean(B ** 2)) / vol_tr_rms),
        noise_to_vol_ratio=(noise_rms / vol_tr_rms) if noise_rms else 0.0,
        bias_phys_overlap_frac=bias_phys_overlap,
        dV_base_m3=float(dV_base), dV_min_m3=float(dV.min()), dV_max_m3=float(dV.max()),
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
    ap.add_argument("--n-days", type=int, default=None, help="record length in days")
    ap.add_argument("--vol-rms", type=float, default=None, help="transient displacement RMS (mm)")
    ap.add_argument("--dv-base-frac", type=float, default=None,
                    help="baseline dV as a multiple of the transient amplitude")
    args = ap.parse_args()

    cfg = dict(CFG)
    if args.ranks:
        cfg["ranks"] = args.ranks
    if args.n_days:
        cfg["n_days"] = args.n_days
    if args.vol_rms is not None:
        cfg["vol_rms_mm"] = args.vol_rms
    if args.dv_base_frac is not None:
        cfg["dV_base_frac"] = args.dv_base_frac
    cfg.update(NOISE_PRESETS[args.noise])
    suffix = args.suffix if args.suffix is not None else ("" if args.noise == "none" else f"_{args.noise}")

    print(f"Generating: ranks={cfg['ranks']} noise={args.noise} "
          f"(white={cfg['noise_white_mm']} flicker={cfg['noise_flicker_mm']} mm) suffix='{suffix}'")
    for K in cfg["ranks"]:
        meta, out_dir = generate_for_K(K, cfg, suffix)
        print(f"  K={K} (spatial rank): vol_tr={meta['realized_vol_transient_rms_mm']:.2f} "
              f"bias={meta['realized_bias_rms_mm']:.2f} noise={meta['realized_noise_rms_mm']:.2f} mm | "
              f"bias/vol={meta['bias_to_vol_ratio']:.2f} noise/vol={meta['noise_to_vol_ratio']:.2f}")
        print(f"        bias components: linear {meta['bias_linear_rate_rms_mm_yr']:.2f} mm/yr "
              f"(max {meta['bias_linear_rate_max_mm_yr']:.2f}), "
              f"annual {meta['bias_annual_amp_mm']:.2f} mm, semiannual {meta['bias_semiannual_amp_mm']:.2f} mm; "
              f"singular values (norm) {np.array2string(np.array(meta['singular_values'][:K])/meta['singular_values'][0], precision=2)}")
        print(f"        transient: {meta['transient_start_date']} -> {meta['transient_end_date']} "
              f"({meta['transient_duration_days']} d, ~{meta['transient_duration_days']/365:.2f} yr); "
              f"dV {meta['dV_baseline_m3']:.2e} -> {meta['dV_plateau_m3']:.2e} m^3 "
              f"(Δ{meta['dV_change_m3']:.2e}, range [-1e7,1e7]) -> {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
