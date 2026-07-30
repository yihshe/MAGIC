# Seed variance on the Austrian dataset (rebuttal)

> **Reviewer.** "The difference observed in Figure 8 seems relatively small and could
> be anecdotal without multiple runs. If this plot is included, it might be useful
> to include variance between runs."

**Claim under test.** Tree species stay *more separable* in PILA's inferred physical
variables than in HVAE's, whichever seed is used — the Figure-8 difference is a
property of the method, not of one lucky initialisation.

Three seeds x {PILA, HVAE} on Austria, everything else identical to the paper runs.

## Seeding

No new seeding machinery was needed. `train_pila.py` / `train_hvae.py` already set
`SEED = 123` (torch + numpy + `cudnn.deterministic`) *before* the model and the
data loaders are built, which covers parameter initialisation, batch shuffling,
worker seeds and the reparameterisation noise, and they already honour an optional
top-level `seed` key that overrides it. Verified: a 2-epoch Austria run reproduces
the October 2025 checkpoint's losses to the last digit.

Seed 123 is the published run and is included in the sweep rather than reused from
`saved/rtm/`, so all three runs come from one code state. (`base_data_loader.py`
also hardcodes `np.random.seed(0)` for the train/val split, but these configs use
`validation_split: 0.0`, so it never fires.)

## Layout

| path | what |
|---|---|
| `make_configs.py` | copies the two published Austria configs, changing only `name`, `seed`, `trainer.save_dir` |
| `configs/` | `{PILA,HVAE}_RTM_C_AUSTRIA_s{123,240,318}.json` |
| `run_seed_variance.sh` | the commands for every step, commented out — uncomment what you want |
| `jm_seeds.py` | JM distances per run -> mean +- std, paired PILA-vs-HVAE comparison |
| `figures/`, `logs/` | outputs |

## Commands

`run_seed_variance.sh` is a menu, not a driver: every command in it is commented
out. Open it and uncomment the step you want, or copy the line you need.

```bash
python rebuttal/seed_variance/make_configs.py

# one line per run — ~2 h each, six of them, independent
python train_pila.py --config rebuttal/seed_variance/configs/PILA_RTM_C_AUSTRIA_s240.json
python train_hvae.py --config rebuttal/seed_variance/configs/HVAE_RTM_C_AUSTRIA_s240.json
# ... s123, s318 likewise

# then the analyzer CSV per checkpoint (test_pila_rtm.py / test_hvae_rtm.py),
# then:
python rebuttal/seed_variance/jm_seeds.py --space jm   # as plotted in Figure 8
python rebuttal/seed_variance/jm_seeds.py --space b    # unsaturated
```

Long runs die with their parent shell — see the `setsid` note at the top of the
script if you need one to survive a closed terminal.

## What `jm_seeds.py` reports

The JM computation is lifted verbatim from `visuals/analysis_rtm.py` (latents
min-max normalised by their physical ranges, Bhattacharyya -> `JM = 2(1-exp(-B))`),
so the numbers stay comparable with the paper. Validated against the published
`jm_distance_statistics.txt` of the paper checkpoint: within-coniferous 1.2450,
within-deciduous 0.8552, between-groups 1.7195 — reproduced exactly.

Per run it reports within-coniferous, within-deciduous, within-all, between-groups,
and two separability measures:

- `gap_between_minus_within` — additive contrast,
- `separability_ratio` — the same contrast scale-free, so a model that merely
  spreads every species further apart does not score better for that alone.

Across runs it reports mean +- std per family, and then the **paired** PILA - HVAE
difference at each seed, how many seeds PILA wins, and whether the mean difference
clears two standard errors. Outputs (prefix `jm_` or `b_`, see below):
`<p>_summary.txt` (the numbers to quote), `<p>_group_stats.csv` (per seed),
`<p>_{PILA,HVAE}_{mean,std}.csv` (per pair), `<p>_seeds.png` (mean heatmaps + a bar
chart with across-seed error bars).

### The JM ceiling — read this before quoting a contrast

`JM = 2(1 - exp(-B))` saturates at 2, and the between-group distances sit close to
that ceiling while the within-group ones do not. The compression is therefore
one-sided, and it distorts any contrast built on JM:

|  | PILA | HVAE | paired diff |
|---|---|---|---|
| JM between-group | 1.719 ± 0.011 | 1.581 ± 0.036 | **+0.139 ± 0.041, 3/3 seeds** |
| JM `separability_ratio` | 1.748 ± 0.030 | 1.788 ± 0.056 | −0.039, PILA wins 1/3 |
| B between-group | 3.070 ± 0.111 | 1.983 ± 0.187 | **+1.087 ± 0.205, 3/3 seeds** |
| B `separability_ratio` | 3.907 ± 0.177 | 3.066 ± 0.244 | **+0.841 ± 0.347, 3/3 seeds** |

PILA's mean pairwise between-group distance is **55% larger** than HVAE's in B, but
that shrinks to 9% once JM squashes it against the ceiling — enough to invert the
JM ratio. JM is monotone in B, so every *pairwise* ordering is identical; only the
aggregates differ. Pass `--space b` to compute everything on the untransformed
Bhattacharyya distance:

```bash
python rebuttal/seed_variance/jm_seeds.py --space jm   # as plotted in Figure 8
python rebuttal/seed_variance/jm_seeds.py --space b    # unsaturated
```

## Code changes

None. This study needs no source modification — it is configs plus an analysis
script.
