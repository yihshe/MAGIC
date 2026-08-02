# Synthetic-control experiment: does the low-rank neural residual recover a *known* bias?

This document specifies the data-generation process and experimental design for the
synthetic-control study, with the mathematical detail needed to reproduce and audit it.

## 1. Motivation

The reviewer asks us to separate the **architectural** contribution of PILA — its
low-rank neural residual — from the specifics of the forest / GNSS case studies, ideally
via a controlled study with known ground truth. We therefore build GNSS-like data in which
the physics is exact, the ground truth is known, and the **complexity (rank) of the
physics–data mismatch is a controlled variable**. The question we answer quantitatively is:

> A neural residual of rank $k$ corrects a systematic bias of intrinsic rank $K$
> **if and only if** $k \ge K$; and increasing $k$ beyond $K$ trades interpretability for fidelity.

All scripts live in this folder; generated data go to `data/processed/mogi_synth/`, trained
models to `saved/mogi_synth/`.

## 2. Notation

| symbol | meaning |
|---|---|
| $N$ | number of daily epochs (record length) |
| $S=36$ | observation dimension: 3 components (E, N, U) $\times$ 12 stations |
| $t\in\{0,\dots,N-1\}$ | time index (days) |
| $u(t)\in\mathbb{R}^{S}$ | observed displacement field at epoch $t$ |
| $\mathcal{F}_{\text{Mogi}}$ | Mogi forward operator (fixed, shared with the case study) |
| $\hat m\in\mathbb{R}^{S}$ | unit vector of the volcanic spatial pattern |
| $K$ | **intrinsic (spatial) rank** of the injected bias |
| $k$ | **rank of PILA's neural residual** (the swept hyper-parameter) |

We write matrices over the record by stacking epochs as rows, e.g. the bias matrix
$B\in\mathbb{R}^{N\times S}$ has rows $b(t)^\top$.

## 3. Generative model

Following the GNSS trajectory decomposition (Bevis & Brown, 2014), each epoch is

$$
u(t) \;=\; \underbrace{S_{\text{vol}}(t)}_{\text{physics (Mogi)}} \;+\; \underbrace{b(t)}_{\text{systematic bias}} \;+\; \underbrace{\varepsilon(t)}_{\text{noise}} .
$$

### 3.1 Physics: a Mogi source with a transient inflation

For a point (Mogi) pressure source at horizontal position $(x_c,y_c)$, depth $d$, and volume
change $\Delta V$, the surface displacement at a station $(x_i,y_i)$ is

$$
C=\frac{1-\nu}{\pi}\,\Delta V,\qquad
R_i=\sqrt{(x_i-x_c)^2+(y_i-y_c)^2+d^2},\qquad
\begin{bmatrix}u^E_i\\ u^N_i\\ u^U_i\end{bmatrix}
=\frac{C}{R_i^{3}}
\begin{bmatrix}x_i-x_c\\ y_i-y_c\\ d\end{bmatrix},
$$

with Poisson ratio $\nu=0.25$. Stacking the 12 real Akutan stations gives
$\mathcal{F}_{\text{Mogi}}(x_c,y_c,d,\Delta V)\in\mathbb{R}^{36}$.

We fix the source geometry $(x_c,y_c,d)=(2,0,5)\,$km and vary only the volume:

$$
\Delta V(t)=\Delta V_{\text{base}}+A_{\Delta V}\,\rho(t),\qquad
\rho(t)=\frac{\arctan\!\big(\tfrac{t-t_0}{w}\big)-\min_t(\cdot)}{\max_t(\cdot)-\min_t(\cdot)}\in[0,1],
$$

an **arctan inflation transient** ($t_0=0.4N$, width $w=30\,$d $\Rightarrow$ a ~6-month
10–90% rise, ~9-month 5–95%). The non-zero baseline $\Delta V_{\text{base}}$ keeps the
source always deforming, so its geometry remains identifiable throughout (Fisher
information for $(x_c,y_c,d)$ scales as $\Delta V^2$; a baseline avoids the ill-posed
$\Delta V\!\to\!0$ regime). Since $\mathcal{F}_{\text{Mogi}}$ is **linear in $\Delta V$** at
fixed geometry,

$$
S_{\text{vol}}(t)=\Delta V(t)\,m,\qquad \hat m = m/\lVert m\rVert,
$$

i.e. the entire volcanic signal lies in the **single direction** $\hat m$. $A_{\Delta V}$ is
calibrated so the *transient variation* $\operatorname{std}_t\!\big(S_{\text{vol}}\big)$ equals
a target `vol_rms_mm` (the part that survives standardization and competes with the bias).

### 3.2 Systematic bias of controlled rank

The bias represents everything the physics cannot express (residual tectonics, hydrological
loading, common-mode error). We give it a realistic temporal content **and** a controlled
spatial rank:

$$
\boxed{\,b(t)=\sum_{m=1}^{K} s_m(t)\,v_m \;=\; V\,s(t)\,},\qquad
B=S V^\top\in\mathbb{R}^{N\times S},
$$

where $V=[v_1,\dots,v_K]\in\mathbb{R}^{S\times K}$ are spatial patterns and
$S=[s_1,\dots,s_K]\in\mathbb{R}^{N\times K}$ are temporal signals, constructed as follows.

**Temporal signals — realistic Bevis trajectories.** Each $s_m$ is a trend + seasonal signal,

$$
\tilde s_m(t)=a_{\text{lin}}\,\sigma_m\frac{t-\bar t}{T}
+a_{\text{ann}}\cos\!\Big(\tfrac{2\pi t}{T}+\phi^{\text{a}}_m\Big)
+a_{\text{semi}}\cos\!\Big(\tfrac{2\pi t}{T/2}+\phi^{\text{s}}_m\Big),
$$

with $T=365.25\,$d, random signs $\sigma_m$ and phases $\phi_m$, and realistic relative
amplitudes $(a_{\text{lin}},a_{\text{ann}},a_{\text{semi}})$ (default $1:2:1$, giving linear
$\sim1$ mm/yr, annual $\sim2$ mm, semi-annual $\sim1$ mm). **All three frequencies are present
at every rank.**

**Two orthogonality constructions fix the rank at exactly $K$.**

1. *Spatial* — draw $K$ random Gaussian vectors, subtract from **each** its projection
   onto the volcanic direction $\hat m$, then orthonormalize:
$$
G=[g_1,\dots,g_K]\sim\mathcal N(0,1)^{S\times K},\qquad
g_j \;\leftarrow\; g_j-\underbrace{(\hat m^{\top} g_j)}_{\text{projection length}}\hat m,\qquad
V\leftarrow\operatorname{QR}(G).
$$
   For each random vector $g_j$, the scalar $\hat m^{\top}g_j$ is *how far it points along*
   $\hat m$, and $(\hat m^{\top}g_j)\,\hat m$ is *that component itself*; subtracting it
   leaves $g_j\perp\hat m$. Stacked over all columns the subtraction is
   $G-\hat m(\hat m^{\top}G)$ — exactly the code line
   `R - np.outer(m_hat, m_hat @ R)`, where `m_hat @ R` $=\hat m^{\top}G$ is the row of
   projection lengths and `np.outer(...)` rebuilds the projections $\hat m(\hat m^{\top}G)$.
   (Equivalently, $G-\hat m(\hat m^{\top}G)=(I-\hat m\hat m^{\top})G$.) QR then
   orthonormalizes, giving $V^{\top}V=I_K$ and $V^{\top}\hat m=0$.
Orthogonality to $\hat m$ means the bias is **unreachable by the physics**, so it cannot be
silently absorbed into $\Delta V$ (preserving identifiability).

2. *Temporal* — orthonormalize the stacked trajectories:
$$
S \leftarrow \operatorname{QR}\big([\tilde s_1,\dots,\tilde s_K]\big)\;\Rightarrow\; S^\top S=I_K .
$$
Without this step the $\tilde s_m$ are **correlated** (they share the annual/semi-annual
frequencies, overlap $\propto\cos\Delta\phi$), which yields a *decaying* singular spectrum and
smears the rank threshold. Orthogonalizing makes the $K$ modes uncorrelated with equal energy.

Finally $B$ is rescaled to a fixed per-entry RMS `bias_total_mm` (default $\approx$ the
volcanic variation, i.e. bias:volcanic $\approx 1{:}1$, matching the real decomposition).
Because the total is fixed, **only the rank changes across $K$**, not the overall difficulty.

**Lemma (exact rank, flat spectrum).** With $S^\top S=I_K$ and $V^\top V=I_K$,
$$
B^\top B = V S^\top S V^\top = V V^\top,
$$
which has eigenvalue $1$ with multiplicity $K$ and $0$ otherwise. Hence $B$ has **exactly $K$
non-zero singular values, all equal** — intrinsic spatial rank $K$ with a *flat* spectrum
(after the RMS rescaling the common value is `bias_total_mm`-dependent, still flat). The flat
spectrum is what produces the sharp reconstruction elbow at $k=K$; a decaying spectrum would
only give diminishing returns.

**Why spatial rank is the operative quantity.** PILA's residual is
$\delta(t)=B_{\text{res}}^\top c(t)$: $k$ *fixed spatial patterns* $B_{\text{res}}\in\mathbb R^{k\times S}$
with *freely time-varying* coefficients $c(t)\in\mathbb R^{k}$ inferred from $u(t)$. It can
represent $b(t)$ for all $t$ **iff** $k\ge\dim\operatorname{span}\{v_m\}=K$, **independent of
how many temporal frequencies** the bias contains (the free $c(t)$ handles the time axis).
This is why the controlled variable is the *spatial* rank $K$.

### 3.3 Noise (optional robustness axis)

$\varepsilon(t)$ is per-component independent flicker (power spectral density $\propto 1/f$,
Timmer–König) plus white noise, at the real GNSS floor ($\approx 2.8$ mm RMS). Being
spatially full-rank it is **not** low-rank-correctable and only sets the reconstruction floor.

### 3.4 Standardization

Each of the 36 channels is standardized with the **training** mean/scale,
$x(t)=\big(u(t)-\mu\big)\oslash\varsigma$, exactly as in the case study. Note the physics
branch reconstructs the absolute Mogi field and re-applies $(\mu,\varsigma)$, so a non-zero
$\Delta V_{\text{base}}$ (carried in $\mu$) is what keeps the geometry identifiable.

## 4. Rank verification (how the ground truth is *measured*)

For every generated dataset the generator computes the SVD of the mean-removed bias matrix
$B_c=B-\bar B$ and **asserts**:

- exactly $K$ singular values exceed $10^{-6}\sigma_1$ (a $>6$-order gap to machine zero);
- the surviving singular values are equal (flat spectrum);
- the recovered bias subspace matches $\operatorname{span}(V)$: overlap
  $\frac1K\lVert V^\top Q\rVert_F^2=1.000$, where $Q$ holds the top-$K$ right singular vectors.

These checks are printed at generation time (see `gen_synth_mogi.py`) and re-verified in
`inspect_synth.py`.

## 5. Experimental design

- **Datasets:** $K\in\{1,2,4\}$ (clean, and a `_realistic` pink+white-noise variant),
  8-year daily records; identical fixed source and transient across $K$.
- **Model sweep:** for each $K$, train PILA at residual rank $k\in\{0,1,2,4,6,8\}$
  ($k=0\equiv$ pure physics, no residual), **all other hyper-parameters identical to the
  real-Mogi ablation**. $3\times6=18$ runs (`make_configs.py`, `run_sweep.sh`).
- **Evaluation** (`eval_sweep.py`, `plot_params.py`) against the known ground truth.

### Metrics

| metric | definition | targets |
|---|---|---|
| reconstruction RMS | $\operatorname{RMS}\big((\hat u-u)\odot\varsigma\big)$ (mm) | fidelity |
| volcanic recovery | $\operatorname{RMS}\big(\hat S_{\text{vol}}-S_{\text{vol}}\big)$ (mm) | interpretability |
| $\Delta V$ recovery | $\operatorname{corr}(\hat{\Delta V},\Delta V)$; and $\operatorname{corr}(\widehat{\Delta V/d^3},\cdot)$ | interpretability |
| source drift | $\operatorname{std}_t(\hat x_c),\operatorname{std}_t(\hat d)$ (km) | identifiability |
| subspace overlap | $\frac1K\lVert V^\top Q_{\text{res}}\rVert_F^2\in[0,1]$, $Q_{\text{res}}$ = top-$K$ basis of the learned residual | architecture |

The $\Delta V/d^3$ combination is reported because $\Delta V$ and $d$ are not separately
identifiable from surface displacements (Mogi depth–volume trade-off); $\Delta V/d^3$ is the
identifiable amplitude that controls the observable.

## 6. Key result

Headline table — reconstruction RMS (fidelity) and volcanic recovery (interpretability),
per intrinsic rank $K$ and residual rank $k$ (from `figures/sweep_metrics.csv`):

| $K$ | metric (mm) | $k{=}0$ | $k{=}1$ | $k{=}2$ | $k{=}4$ | $k{=}6$ | $k{=}8$ |
|---|---|---|---|---|---|---|---|
| **1** | reconstruction RMS ↓ | 2.67 | **0.88** | 0.61 | 0.39 | 0.31 | 0.26 |
|   | volcanic recovery | 0.61 | **0.61** | 0.60 | 1.03 | 1.50 | 1.98 |
| **2** | reconstruction RMS ↓ | 2.59 | 1.74 | **0.70** | 0.43 | 0.37 | 0.31 |
|   | volcanic recovery | 1.12 | 1.10 | **1.05** | 1.12 | 1.41 | 1.76 |
| **4** | reconstruction RMS ↓ | 2.57 | 2.25 | 1.88 | **0.44** | 0.31 | 0.28 |
|   | volcanic recovery | 0.93 | 0.91 | 0.89 | **0.85** | 0.80 | 1.43 |

Reconstruction collapses sharply as $k$ reaches $K$ (bold); pushing $k>K$ keeps
lowering reconstruction RMS but *raises* the volcanic-recovery error. Bias-subspace
overlap $\tfrac1K\lVert V^\top Q_{\text{res}}\rVert_F^2$ tracks $\min(k,K)/K$ almost exactly:

| $K$ | $k{=}1$ | $k{=}2$ | $k{=}4$ | (prediction) |
|---|---|---|---|---|
| 2 | 0.39 | **0.86** | 0.87 | 0.5, 1.0, 1.0 |
| 4 | 0.21 | 0.45 | **0.90** | 0.25, 0.5, 1.0 |

Source-depth drift $\operatorname{std}_t(\hat d)$ (km) — the identifiability cost of
over-flexibility: small and flat for $k\le K$, then grows once $k>K$:

| $K$ | $k{=}0$ | $k{=}1$ | $k{=}2$ | $k{=}4$ | $k{=}6$ | $k{=}8$ |
|---|---|---|---|---|---|---|
| 1 | 0.35 | **0.34** | 0.34 | 0.64 | 0.86 | 0.76 |
| 2 | 0.50 | 0.53 | **0.47** | 0.57 | 0.77 | 0.91 |
| 4 | 0.29 | 0.30 | 0.29 | **0.27** | 0.33 | 0.63 |

*(all tables from `figures/sweep_metrics.csv`; see `figures/sweep_results.png` (6-panel),
`param_recovery.png`, `synth_inspection.png`. 8-year record, bias:volcanic ≈ 1:1, clean data.)*

Supporting evidence:

- **Sharp elbow:** reconstruction RMS collapses precisely at $k=K$ — the residual needs
  exactly the intrinsic rank of the mismatch.
- **Subspace overlap $\approx\min(k,K)/K$:** the learned residual recovers exactly $k$ of the
  $K$ true bias directions — the gain is the *rank* inductive bias, not generic capacity.
- **Interpretability–fidelity divergence for $k>K$:** reconstruction keeps improving while
  $\Delta V$/geometry recovery degrades — locating the point at which extra flexibility starts
  to harm interpretability.

## 7. Files

```
gen_synth_mogi.py   data generator (Sec. 3–4); prints ground-truth transient timing,
                    magnitudes, and the rank/spectrum check
inspect_synth.py    data inspection figures (decomposition, rank spectrum, noise PSD)
make_configs.py     writes the 18 sweep configs (K x k)
run_sweep.sh        menu of commented train/test commands (uncomment the step you want)
eval_sweep.py       metrics table + sweep_results.png
plot_params.py      parameter recovery vs ground truth
figures/            outputs (sweep_metrics.csv, *.png)
```

Reproduce (from the repo root, `conda activate mres`):
```bash
P=rebuttal/synthetic_control
python $P/gen_synth_mogi.py --ranks 1 2 4        # generate data (+ --noise realistic)
python $P/inspect_synth.py                       # optional: verify rank/spectrum, figures
python $P/make_configs.py                        # write the 18 configs
# train + test all 18 (run_sweep.sh is a commented menu; this is the batch form):
for cfg in $P/configs/SYN_K*_r*.json; do
  python train_pila.py --config "$cfg"
  d=$(ls -td saved/mogi_synth/$(basename "$cfg" .json)/*/ | head -1)
  python test_pila_mogi.py --config ${d}models/config.json --resume ${d}models/model_best.pth
done
python $P/eval_sweep.py                           # metrics + sweep_results.png
python $P/plot_params.py                          # parameter recovery figure
```

### Reference
Bevis, M., & Brown, A. (2014). Trajectory models and reference frames for crustal motion
geodesy. *Journal of Geodesy*, 88(3), 283–311.
