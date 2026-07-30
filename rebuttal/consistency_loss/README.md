# Physical consistency loss added to PILA (rebuttal)

> **Reviewer.** "Could PILA potentially be improved through addition of a physical
> consistency loss as in HVAE?"

Run on **GreenHyperSpectra**, so the answer is a per-trait R² against measured
traits rather than an assertion. Data and evaluator are shared with
`rebuttal/greenhyperspectra/`, which is **read-only** here.

## Design

HVAE's physical-consistency term is the synthetic-pair inversion loss: sample
`z ~ U(0,1)`, push it through the physics model, require the encoder to invert it
back, and compare the **inferred z against the sampled z**. PILA gets exactly that
comparison (`PhysVAETrainerSMPL._synthetic_consistency_loss_z`); the only
difference from HVAE is where the squashing sits — HVAE's encoder head ends in
Softplus and emits z directly, PILA's emits u, so `draw`'s sigmoid is applied
before comparing.

PILA's pre-existing `_synthetic_data_loss` (u against `logit(z)`) is the Stage-A
bootstrap; it is left untouched and is inactive at `epochs_pretrain: 0`.

**No baseline is trained here** — the weight-0 reference is the existing
`PILA_PROSAIL_C_GHS` run under `saved/greenhyperspectra/`.

| run | weight |
|---|---|
| `S1_w0.1` | 0.1 |
| `S2_w1.0` | 1.0 |
| `S3_w10` | 10.0 |

## On the weights (and the 1721 bands)

The band count does **not** need compensating for: this loss never touches the
spectral axis, and `rec_loss` is already normalised per band via
`trainer.rec_loss_scale`.

The weights were set from measurement, not guesswork. In z-space the term is a sum
of squared errors over 4 latents in [0,1], so it is small — **0.32 at the end of
epoch 1, 0.19 by epoch 2**, against `rec_loss` 6.4 → 1.9. (The u-space Stage-A
version is ~30x larger, 9.87 → 5.46 in the same test, because logit targets have
`var = π²/3`. Weights calibrated for that term would be ~30x too small here.) So
0.1 / 1.0 / 10.0 spans roughly 1% of `rec_loss` (negligible), 5–10% — the order of
magnitude at which HVAE keeps its own regularisers — and 50–100% (dominant).

**The spread is itself the finding.** Whatever the R² comes out at, the added term
brings a weight that must be tuned per dataset — the multi-term balancing burden
PILA was designed to avoid. Worth stating alongside the numbers.

**Caveat for the write-up.** Matching synthetic pairs is not supervision. Where
measured labels exist, semi-supervised training on the traits themselves is the
better use of them; this experiment answers the question only as the reviewer
posed it.

## Commands

`run_consistency_loss.sh` is a menu, not a driver: every command in it is
commented out. Open it and uncomment the step you want.

```bash
python rebuttal/consistency_loss/make_configs.py
python train_pila.py --config rebuttal/consistency_loss/configs/PILA_PROSAIL_C_GHS_S1_w0.1.json
# ... S2_w1.0, S3_w10 likewise — ~1.7 h each, independent
python rebuttal/greenhyperspectra/eval_greenhyperspectra.py \
    --resume saved/rebuttal/consistency_loss/<name>/<run>/models/model_best.pth \
    --out_dir rebuttal/consistency_loss/figures
python rebuttal/compare_ghs_variants.py \
    --dir rebuttal/consistency_loss/figures --baseline PILA_PROSAIL_C_GHS
```

Requires the GHS dataset built by
`rebuttal/greenhyperspectra/preprocess_greenhyperspectra.py`. Each run is scored
with `rebuttal/greenhyperspectra/eval_greenhyperspectra.py` into `figures/`.
A 2-epoch smoke config (`configs/_debug_*.json`) runs the path in ~2 min. Long
runs die with their parent shell — see the `setsid` note at the top of the script.

## Code changes

One knob in `trainer/trainer_phys_smpl.py`, marked `[REBUTTAL ONLY]`:

- `balance_data_aug_cotrain` (default `0.0`) — enables
  `_synthetic_consistency_loss_z` during the main phase. At 0.0 the term is never
  computed, so every config under `configs/` keeps the published code path.

Verified: with the published Austria config, PILA reproduces its pre-change
training losses **bit-identically** over 2 epochs; the loss arithmetic of the new
term was checked by hand against the logs.
