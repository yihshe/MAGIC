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
| `make_configs.py` | generates the configs so PILA/HVAE stay consistent |
| `configs/` | `{PILA,HVAE}_PROSAIL_{A,B,C}_ghs.json` (C is the one that matters) |
| `smoke_test.py` | builds every config, runs real steps, reports timing + loss balance |
| `run_greenhyperspectra.sh` | launches the two C runs concurrently |
| `eval_greenhyperspectra.py` | per-trait R²/RMSE/nRMSE + 1:1 density plots |
| `compare_greenhyperspectra.py` | the headline PILA-vs-HVAE table |
| `figures/`, `logs/` | outputs |

## Commands

```bash
# 1. build the dataset (one-off; ~5 min, dominated by reading two 208 MB shards)
python rebuttal/greenhyperspectra/preprocess_greenhyperspectra.py \
    --ghs_root /maps/ys611/HyspectraSSL --n_splits 2

# 2. configs
python rebuttal/greenhyperspectra/make_configs.py

# 3. sanity check (~2 min)
python rebuttal/greenhyperspectra/smoke_test.py

# 4. train PILA-C and HVAE-C (~1.2 h wall-clock, both in parallel)
./rebuttal/greenhyperspectra/run_greenhyperspectra.sh

# 5. the headline table (evaluates both checkpoints and diffs them)
python rebuttal/greenhyperspectra/compare_greenhyperspectra.py

# ...or one checkpoint on its own, with the 1:1 density plots
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

PROSPECT-PRO + 4SAIL (`sensor='default'`, no SRF resampling) instead of INFORM.
PILA's `physics/rtm` already supports this, so `Physics_PROSAIL` differs from
`Physics_RTM` only in `model_init` and the band slice — no new RTM code, and
nothing is imported from HyspectraSSL.

Geometry is fixed (`tts=30, tto=0, psi=0`); GHS carries no per-sample angles.

### Latent parameterization

Seven free latents, `[cab, cw, LAI, cp, cbc, car, anth]`, each a sigmoid
rescaled to its range. **`cm` is derived as `cp + cbc`, not inferred**, because:

* PROSPECT-PRO is parameterised by `(cp, cbc)` and never takes `cm`;
* the constraint holds *exactly* in the labels (`max |cm − (cp+cbc)| = 1e-7`
  over the 2203 rows carrying all three).

This removes the failure mode in GHS's own implementation, where `cbc = cm − cp`
can go negative and drive PROSAIL to NaN, and it removes their dead latent
dimension (their index 5 is both overwritten by the constraint and excluded from
the supervised loss). Here both constituents are positive by construction, and
`cm` still comes out for free as an evaluated trait.

**`N` (leaf structure) is frozen at 1.5**, matching GHS: their `AE_RTM` learns
exactly 8 latents (`cab, cw, cm, LAI, cp, cbc, car, anth`) and never touches `N`,
which keeps the RTM's `para_init` default. Verified here by inspecting
`para_dict['N']` after a forward pass (constant 1.5, `requires_grad=False`).

So this parameterization has 7 free latents against GHS's 8 — but theirs includes
`cm`, which is immediately overwritten by their `cm - cp` constraint *and*
excluded from their supervised loss. Both carry the same **7 physical degrees of
freedom**; this one simply has no dead dimension.

### Ranges

Base values are the ones hardcoded in GHS
(`src/AE_RTM/AE_RTM_architectures.py`), which are the pooled train+test min/max.
Verified against the data, with these corrections:

| trait | change | why |
|---|---|---|
| `cbc` | `[0.00135, 0.0462]` → `[1.05e-5, 0.0636]` | declared range excludes **159** pooled samples below its min and 8 above its max |
| `cab` | max `229.5` → `120` | the max is a single *train* outlier; q99.9 = 98.3, test max = 78.1 |
| `car` | max `40.4` → `30` | same, one train outlier; q99.9 = 20.5, test max = 19.6 |
| `anth` | `[0.561, 2.98]` → `[0.05, 4.0]` | declared range comes from 289 rows of essentially one source dataset; binding the sigmoid that tightly is an unjustified prior |

Both capped outliers live in `train`, not `test`, so evaluation is unaffected.
Each upper bound is then padded 5% above the largest observed label and rounded
to clean decimals, so the data does not sit on the sigmoid's upper end-stop.
Lower bounds are set to zero where that is physically meaningful and verified
NaN-free in PROSAIL, with three exceptions carrying small positive floors:

| latent | floor | why |
|---|---|---|
| `LAI` | `0.01` | 4SAIL divides by `ks*LAI`; `LAI=0` is NaN forward and backward (PILA uses the same floor in `configs/rtm_paras.json`) |
| `cw` | `1e-5` | `cw=0` *and* `cbc=0` together leave no SWIR absorber → PROSPECT degenerates to NaN |
| `cbc` | `1e-5` | as above |
| `cp` | `1e-6` | same grounds, and keeps the derived `cm = cp + cbc` strictly positive |

`data/report.txt` prints where every observed extreme lands in z-space.

## Changes to the main tree

All additive; defaults reproduce existing behaviour exactly, so the Wytham,
Austria and Mogi results are untouched.

| file | change |
|---|---|
| `datasets/spectrumGHS.py` | **new** `SpectrumGHS` dataset (npz; augmentation hook, off) |
| `data_loader/data_loaders.py` | **new** `SpectrumGHSDataLoader` |
| `model/model_phys_smpl.py` | **new** `Physics_PROSAIL`; `physics: "PROSAIL"`; `hidlayers_aux_dec` made config-driven |
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
