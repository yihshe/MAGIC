# GreenHyperSpectra experiment (rebuttal)

Answers the reviewer's point that *"the results on the Austrian dataset are hard to
interpret without ground truth"*: the identical PILA pipeline is run on
GreenHyperSpectra (GHS, arXiv:2507.06806), a hyperspectral dataset that **does**
carry measured trait labels, so the inferred physical variables can be scored
directly.

**PILA and HVAE are trained fully unsupervised here** — reconstruction loss only,
exactly as on Wytham and Austria. The labels are used for two things, neither of
which is training:

1. deriving the sigmoid ranges of the physical latents,
2. evaluation.

**The claim is narrow and singular:** on a dataset where the traits are actually
measured, PILA recovers them more accurately than HVAE. That is variant **C** for
both models (physics + correction), one run per setting. Variants A and B have
configs here for reference but are not part of the claim and are not run by
default — the ablation is already reported on the S2 datasets.

> **Not a like-for-like baseline.** GHS's own AE-RTM is *semi-supervised* (Huber
> loss on 7 traits in Box-Cox space plus reconstruction), so its published R²
> (0.63–0.76 on `labeled_splits/test.csv`) is a **supervised upper reference**,
> not a competitor to the numbers here. The comparison that matters is
> PILA vs HVAE: same data, same latent bounds, both unsupervised.

Augmentation and a semi-supervised loss are deliberately left out; both are
documented follow-ups, not part of this experiment.

## Layout

| path | what |
|---|---|
| `preprocess_greenhyperspectra.py` | reads the GHS release **once**, writes everything below |
| `data/` | self-contained dataset; nothing here points at HyspectraSSL afterwards |
| `data/ghs_prosail_paras.json` | sigmoid ranges for the four learnable latents |
| `data/ghs_prosail_constants.json` | fixed leaf parameters (`car`, `anth`) |
| `make_configs.py` | generates the configs so PILA/HVAE stay consistent |
| `configs/` | `{PILA,HVAE}_PROSAIL_{A,B,C}_ghs.json` (C is the one that matters) |
| `run_greenhyperspectra.sh` | every step, commented out, debugger-friendly |
| `eval_greenhyperspectra.py` | checkpoint → metrics, predictions CSV and 1:1 plots |
| `compare_greenhyperspectra.py` | the headline PILA-vs-HVAE table |
| `figures/`, `logs/` | outputs |

## Commands

```bash
# 1. build the dataset (one-off; ~5 min, dominated by reading two 208 MB shards)
python rebuttal/greenhyperspectra/preprocess_greenhyperspectra.py \
    --ghs_root /maps/ys611/HyspectraSSL --n_splits 2

# 2. configs
python rebuttal/greenhyperspectra/make_configs.py

# 3. train PILA-C and HVAE-C (~1.2 h wall-clock, both in parallel)
./rebuttal/greenhyperspectra/run_greenhyperspectra.sh

# 4. the headline table (evaluates both checkpoints and diffs them)
python rebuttal/greenhyperspectra/compare_greenhyperspectra.py

# ...or one checkpoint on its own
python rebuttal/greenhyperspectra/eval_greenhyperspectra.py \
    -r saved/greenhyperspectra/PILA_PROSAIL_C_GHS/<run_id>/models/model_best.pth
```

To add seeds later: set `SEEDS = [240, 318]` in `make_configs.py`, then
`SEEDS="240 318" ./run_greenhyperspectra.sh` and
`compare_greenhyperspectra.py --seeds 123 240 318` for mean ± std.

## Dataset

Train pool = `labeled_splits/train.csv` (4508) + `Splits/split_{1,2}.csv`
(13 948) = 18 456 rows, of which 10% is held out for `val_rec_loss`.
Evaluation set = `labeled_splits/test.csv` (1127 rows).

`labeled_all/all.csv` is **not** used: GHS's released training path splits it and
consumes all of it, which leaks 99.6% of `test.csv`. `train.csv`/`test.csv` are
row-disjoint and distribution-matched over the same 46 source datasets.

Spectra are the 1721 GHS-retained bands (400–1350, 1431–1800, 2051–2450 nm),
standardized per band with a `StandardScaler` fit on the train portion — the same
scheme as Wytham (`train_x_mean.npy` / `train_x_scale.npy`), minus the /10000
factor since GHS reflectance is already in [0, 1]. The physics output is
standardized with the identical statistics.

Labels are sparse: no row carries all 8 traits, and per-trait test coverage runs
from 736 (`cm`) down to 73 (`anth`), so every metric is reported with its own n.

## Physics

PROSPECT-D + 4SAIL (`sensor='default'`, no SRF resampling) instead of INFORM.
PILA's `physics/rtm` already supports this, so `Physics_PROSAIL` differs from
`Physics_RTM` only in `model_init`, the band slice and the fixed leaf
parameters — no new RTM code, and nothing is imported from HyspectraSSL.

Geometry is fixed (`tts=30, tto=0, psi=0`); GHS carries no per-sample angles.

### Latent parameterization

**Four free latents, `[cab, cw, cm, LAI]`**, each a sigmoid rescaled linearly to
its range. `car` and `anth` are held constant; `N` keeps the RTM default 1.5, as
in GHS (their `AE_RTM` never touches it either).

These four are exactly the ones carrying the reconstruction. Clamping each
latent to its training median and re-running the physics costs:

| latent | Δ physics-only RMSE | | latent | Δ |
|---|---|---|---|---|
| LAI | **+170.8%** | | car | +4.9% |
| cbc→cm | **+96.3%** | | cp | +2.8% |
| cw | **+78.3%** | | anth | +0.0% |
| cab | **+22.2%** | | | |

**PROSPECT-D rather than PROSPECT-PRO.** PRO resolves dry matter into protein
(`cp`) and carbon-based constituents (`cbc`), which is why GHS uses it. That
second degree of freedom is not identifiable from fresh canopy spectra:

* `cos(K_cbc, K_cm) = 0.9986` — cbc's coefficient is already PROSPECT-D's dry
  matter. Half of `K_cp` is genuinely independent of `K_cbc`, but weighted by
  observed content (protein is 9.4× scarcer, medians 7.31e-4 vs 6.888e-3 g/cm²)
  the unique protein signal is **4.7%** of the cbc-driven one.
* Matched-filter SNR for `cp` is **0.48** — a full p10–p90 swing moves the
  spectrum half as much as the model's own residual.
* In the labels `cm ≡ cp + cbc` to 1e-7 over 2203 rows, so only two degrees of
  freedom are present regardless.
* The amide features PRO relies on are masked by liquid water, and GHS deletes
  1351–1430 and 1801–2050 nm, removing the 1940 nm band outright.

PROSPECT-D takes `cm` directly, which also removes the failure mode in GHS's own
implementation where `cbc = cm − cp` can go negative and drive PROSAIL to NaN.

**`car` and `anth` are fixed** at the rounded train-label medians (8.0 µg/cm²
and 1.25 µg/cm²) via `data/ghs_prosail_constants.json`. Carotenoid and
chlorophyll absorb almost identically — specific absorption coefficients peaking
4 nm apart (436 vs 440 nm), cosine 0.776 over 400–750 nm — and chlorophyll
saturates the blue, so `d(trans)/dk` there is 0.081 against 0.306 at 550 nm.
Left free, the `car` latent came out **0.755-correlated with the inferred `cab`**:
a chlorophyll shadow, not a retrieval. Anthocyanin is weaker still (0.22 vs
car's 0.75 at peak) and 0.924-collinear with `cbrown`, which is itself frozen;
GHS's own *semi-supervised* AE-RTM scores anth R² = 0.335, its worst trait by a
wide margin. Deliberately **not** the RTM defaults (`car=10`, `anth=2`), which
sit at the 76th and 93rd percentile of these labels rather than the median.

The claim this supports is narrow and stated as such: *chlorophyll is the only
leaf pigment retrievable this way* — alongside water, dry matter and LAI.

### Ranges

Base values are the ones hardcoded in GHS
(`src/AE_RTM/AE_RTM_architectures.py`), which are the pooled train+test min/max.
Verified against the data, with these corrections:

| trait | change | why |
|---|---|---|
| `cab` | max `229.5` → `120` | the max is a single *train* outlier; q99.9 = 98.3, test max = 78.1 |

The capped outlier lives in `train`, not `test`, so evaluation is unaffected.
Each upper bound is then padded 5% above the largest observed label and rounded
to clean decimals, so the data does not sit on the sigmoid's upper end-stop.
Lower bounds are set to zero where that is physically meaningful and verified
NaN-free in PROSAIL, with three exceptions carrying small positive floors:

| latent | floor | why |
|---|---|---|
| `LAI` | `0.01` | 4SAIL divides by `ks*LAI`; `LAI=0` is NaN forward and backward (PILA uses the same floor in `configs/rtm_paras.json`) |
| `cw` | `1e-5` | `cw=0` *and* `cm=0` together leave no SWIR absorber → PROSPECT degenerates to NaN |
| `cm` | `1e-5` | as above |

The mapping stays **linear** — `value = min + (max − min)·z`. A quantile warp
would put z=0.5 on the training median instead of the range midpoint (which is
1.3–4.9× the median, and is why the inferred traits skew high), but it would
make the sigmoid carry a label-derived distributional prior. Left out
deliberately: the only thing labels touch here is the min/max bounds. The
consequence is that R² is penalised by a bias neither model can learn, so
**Pearson r is reported alongside it**.

`data/report.txt` prints where every observed extreme lands in z-space, and the
percentile each fixed value sits at.

## Changes to the main tree

All additive; defaults reproduce existing behaviour exactly, so the Wytham,
Austria and Mogi results are untouched.

| file | change |
|---|---|
| `datasets/spectrumGHS.py` | **new** `SpectrumGHS` dataset (npz; augmentation hook, off) |
| `data_loader/data_loaders.py` | **new** `SpectrumGHSDataLoader` |
| `model/model_phys_smpl.py` | **new** `Physics_PROSAIL` (PROSPECT-D + 4SAIL, optional `rtm_constants`); `physics: "PROSAIL"`; `hidlayers_aux_dec` made config-driven |
| `model/model_phys.py` | same, plus `hidlayers_aux2_expand` / `hidlayers_aux2_dec` / `aux2_map_hidden` made config-driven |
| `trainer/trainer_phys_smpl.py`, `trainer/trainer_phys.py` | **new** `rec_loss_scale` (default `1.0` = unchanged) |
| `train_pila.py`, `train_hvae.py` | optional `seed` config key; absent or `123` leaves the RNG untouched |

### Three things worth knowing

**`rec_loss_scale`.** The reconstruction loss is summed over the feature axis,
so 1721 bands make it ~156× larger than on 11-band S2 data, silently demoting
every weight measured against it — the orthogonality penalty for PILA and, more
importantly, the KL term for HVAE, which is weighted against `rec_loss` directly
as `kl * exp(x_lnvar)`. Rather than re-tuning each weight per dataset,
`rec_loss_scale = 11/1721` renormalizes the reconstruction loss to the S2 band
count the weights were tuned at, and **every existing loss weight carries over
untouched**. Note this is deliberately *not* a plain per-band mean (`1/1721`),
which was measured to leave HVAE's KL term ~100× stronger relative to `rec_loss`
than in the S2 runs rather than where it was. Both trainers read the same key;
default `1.0` leaves prior configs unchanged. (The HVAE trainer computes
`rec_loss` inline rather than via `self.criterion`, and its Mogi configs use a
different criterion entirely, so a shared config key was the only safe place to
put this.)

**A degenerate PROSPECT state.** Setting the lower bounds to exactly zero looked
clean but made HVAE diverge to NaN inside two epochs. `cw = 0` **and** `cbc = 0`
together leave no absorber anywhere in the SWIR and PROSPECT's plate model
degenerates; either alone is fine. PILA's sigmoid never quite reaches the corner,
but HVAE's `draw()` clamps sampled latents to exactly `[0, 1]` and lands on it.
Hence the small positive floors on `cw`, `cbc` and `cp` in the range table.

**Batch-size agnosticism.** `RTM.para_reset` broadcasts the frozen parameters to
the batch size of the *first* call only, which then caps every later batch.
Training never trips this (train 256 comes before validation 64), but it makes
mixed batch sizes order-dependent. `Physics_PROSAIL` re-arms the flag each call
— a few `torch.full`s against a 0.2 s RTM step, no numerical change.

## Sizing and cost (measured, CPU-only)

A PILA step costs ~1.4 ms/sample, flat in batch size *and* thread count: the
bottleneck is serial Python inside the RTM (the `para_grid` double loop alone is
28% of the forward), not BLAS. PROSPECT-PRO+4SAIL is ~2.6× *cheaper* per sample
than INFORM, which calls 4SAIL three times.

| config | s/epoch | 150 epochs |
|---|---|---|
| **PILA C** | 23 | **1.0 h** |
| **HVAE C** | 28 | **1.2 h** |
| PILA/HVAE B | 22–23 | ~1.0 h |
| PILA/HVAE A | 1–2 | ~0.1 h |

Optimizer, batch size (256), learning rate (3e-4) and epochs (150) follow the
existing PILA RTM configs unchanged. Note that 18 456 rows at batch 256 gives
~65 steps/epoch, so 150 epochs is ~9.8k optimizer steps against ~29.9k in the
Wytham runs; if the loss is still descending at the end, raising `EPOCHS` in
`make_configs.py` is the knob.

`residual_rank = 4` and `dim_z_aux = 4` for variant C (up from 2 on the 11-band
S2 data). Every loss weight is unchanged from the S2 configs; the band-count
difference is absorbed by `rec_loss_scale` instead.

An optional ~18% speedup was left on the table: replacing the `para_grid` double
loop in `physics/rtm/Resources/PROSAIL/call_model.py` with `torch.stack` is 65×
faster and numerically identical, but it sits in physics code the existing
results depend on. Worth doing only if scaling to all 20 unlabeled shards.
