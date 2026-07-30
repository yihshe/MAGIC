# End-stop loss inside HVAE: architecture or prior? (rebuttal)

> **Reviewer.** "The authors show that using a different prior in PILA reduces the
> quality of the output. What about using the PILA end-stop loss in the HVAE method
> — would that improve it? Basically, the question here is whether the benefits of
> PILA come from the architecture or the prior."

Run on **GreenHyperSpectra**, not Austria: GHS carries measured traits, so the
answer is a per-trait R² rather than a plausibility argument. Data, latent ranges
and the evaluator are shared with `rebuttal/greenhyperspectra/`, which is
**read-only** here — no file in that folder is modified.

## Design

HVAE's architecture is left exactly as published — Softplus mean, Gaussian draw,
clamp to [0,1], full-rank correction decoder. Only the loss changes.

One run: `HVAE_PROSAIL_C_GHS_endstop` — the prior on `z_phy` switched off
(`balance_kld_z_phy: 0`) and PILA's end-stop barrier added in its place, i.e.
**PILA's loss on HVAE's architecture**. `z_aux2` keeps its KL weight, so the only
latents whose treatment changes are the physical ones.

**No baseline is trained here** — the comparison is against the existing
`HVAE_PROSAIL_C_GHS` run under `saved/greenhyperspectra/` (identical config and
data), with PILA-C from the same folder as the target to reach.

**Reading it.** If the end-stop run approaches PILA-C, the benefit is the loss; if
it does not, the benefit is the architecture. Note this moves two things at once
(prior off, end-stop on), which is deliberate — PILA's loss *is* that combination —
but it does mean a positive result cannot be attributed to the end-stop alone
without the KL-off-only control.

**The end-stop weight is dynamic.** Rather than fixing an absolute value, the term
is rescaled each step by the same `_loss_weight()` helper HVAE already uses for its
unmix / synthetic-data / least-action terms, keeping it an order of magnitude below
`rec_loss`. Measured in the smoke test: weight 0.1 → 0.074 → 0.062 as `rec_loss`
falls, with the barrier converging to ~1.8 and staying there.

## Two things worth knowing before quoting the result

**HVAE's prior is already nearly inert.** The KL is weighted by
`x_var = exp(x_lnvar) = e^-9`, so at `balance_kld = 1.0` its effective weight is
~1.2e-4. What actually keeps `z_phy` in range is the clamp. The trainer now logs
`z_phy_clamp_frac` (fraction of draws the clamp had to cut) so this can be
measured rather than assumed — in the smoke test it was **0.61 at initialisation**,
falling to ~0 within 20 batches once the end-stop penalty is switched on.

**The end-stop penalty is applied to the pre-clamp draw.** On the clamped value the
barrier is constant exactly at the end-stops, i.e. zero gradient for precisely the
samples it is meant to push back inside — with 61% of draws out of range at init,
that would have rigged the experiment against the reviewer's hypothesis. The
barrier is therefore continued linearly past the end-stops
(`model.loss.end_stop_penalty`), which reduces to PILA's exact form for any value
in `[eps, 1-eps]`. `edge_penalty_target: "clamped"` reproduces the literal
alternative if wanted.

## Commands

`run_prior_vs_endstop.sh` is a menu, not a driver: every command in it is
commented out. Open it and uncomment the step you want.

```bash
python rebuttal/prior_vs_endstop/make_configs.py
python train_hvae.py --config rebuttal/prior_vs_endstop/configs/HVAE_PROSAIL_C_GHS_endstop.json
python rebuttal/greenhyperspectra/eval_greenhyperspectra.py \
    --resume saved/rebuttal/prior_vs_endstop/HVAE_PROSAIL_C_GHS_endstop/<run>/models/model_best.pth \
    --out_dir rebuttal/prior_vs_endstop/figures
python rebuttal/compare_ghs_variants.py \
    --dir rebuttal/prior_vs_endstop/figures --baseline HVAE_PROSAIL_C_GHS
```

Requires the GHS dataset built by
`rebuttal/greenhyperspectra/preprocess_greenhyperspectra.py`. Each run is scored
with `rebuttal/greenhyperspectra/eval_greenhyperspectra.py` into `figures/`
(`*_metrics.json`, `*_preds.csv`, `*_scatter.png`). A 2-epoch smoke config
(`configs/_debug_*.json`) runs the whole path in ~2 min. Long runs die with their
parent shell — see the `setsid` note at the top of the script.

## Code changes

All in `trainer/trainer_phys.py` + `model/model_phys.py`, marked `[REBUTTAL ONLY]`,
all default-off and keyed off config entries that no shipped config sets:

- `balance_kld_z_phy` / `balance_kld_z_aux` — per-latent KL weights. Absent, the
  original single combined weight is applied exactly as before.
- `edge_penalty_weight` (+ `_dynamic`, `_power`, `_target`, `_eps`) — the end-stop
  term, never computed at weight 0; `_dynamic` hands the weighting to
  `_loss_weight()`.
- `PHYS_VAE.draw(..., return_raw=False)` — opt-in pre-clamp draw; the decode path
  and every existing caller are untouched.

Verified: with the published Austria configs, PILA and HVAE reproduce their
pre-change training losses **bit-identically** over 2 epochs; the Mogi and `no_phy`
configs still train.
