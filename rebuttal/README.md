# Rebuttal experiments

One folder per reviewer question. Each is self-contained: config generator,
configs, a runner, and its own `figures/` and `logs/`.

| folder | reviewer question | dataset | code changes |
|---|---|---|---|
| `greenhyperspectra/` | results on Austria are hard to interpret without ground truth | GHS (measured traits) | — |
| `synthetic_control/` | does the low-rank residual recover a known bias? | synthetic Mogi | — |
| `seed_variance/` | Figure 8 could be anecdotal without multiple runs | Austria | none |
| `prior_vs_endstop/` | do PILA's benefits come from the architecture or the prior? | GHS | HVAE trainer, default-off |
| `consistency_loss/` | could PILA be improved by HVAE's physical consistency loss? | GHS | PILA trainer, default-off |

## Compatibility with the published experiments

Everything under `configs/` (the paper runs) must keep working unchanged, so every
source modification made for the rebuttal is:

- **config-gated and default-off** — no config under `configs/` sets any of the new
  keys, so they all take the original code path;
- **marked `[REBUTTAL ONLY]`** at every site, saying which reviewer question it
  serves and that it is not part of the published runs;
- **regression-tested** — with the published Austria configs, both PILA and HVAE
  reproduce their pre-change training losses bit-identically over 2 epochs; the
  Mogi and `no_phy` configs still train.

Grep the markers with:

```bash
grep -rn "REBUTTAL ONLY" model/ trainer/
```

| key | trainer | default | effect |
|---|---|---|---|
| `balance_kld_z_phy` / `balance_kld_z_aux` | HVAE | absent → combined weight, as before | weight the two KL terms separately |
| `edge_penalty_weight` (+ `_power`/`_target`/`_eps`) | HVAE | `0.0` → never computed | PILA's end-stop barrier on `z_phy` |
| `balance_data_aug_cotrain` | PILA | `0.0` → never computed | keep the synthetic-pair consistency loss on during the main phase |

`PHYS_VAE.draw(..., return_raw=False)` gained one opt-in keyword; the decode path
and all existing callers are unchanged.

## Reuse, not duplication

`prior_vs_endstop/` and `consistency_loss/` build their configs from
`greenhyperspectra/configs/*.json` and score checkpoints with
`greenhyperspectra/eval_greenhyperspectra.py`. That folder is **read-only** to
them — neither generator writes into it.
