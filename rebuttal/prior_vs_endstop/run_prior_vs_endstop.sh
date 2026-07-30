#!/bin/bash
# End-stop loss inside HVAE, on GreenHyperSpectra (reviewer: "do PILA's benefits
# come from the architecture or the prior?"). Same style as ../greenhyperspectra/
# run_greenhyperspectra.sh: every command is commented out -- uncomment the one
# you want to run.
#
# Run from the repo root:   ./rebuttal/prior_vs_endstop/run_prior_vs_endstop.sh
# Override the interpreter: PYTHON=python3 ./rebuttal/prior_vs_endstop/run_prior_vs_endstop.sh
#
# One training run: HVAE with the prior on z_phy switched off and PILA's end-stop
# barrier in its place, dynamically weighted. ~1.9 h (150 epochs, 1721 bands, CPU,
# 8 threads). Requires the GHS dataset built by
# rebuttal/greenhyperspectra/preprocess_greenhyperspectra.py.
#
# NOTE ON LONG RUNS: a plain background job dies when its parent shell exits. To
# survive a closed terminal, wrap the command:
#   setsid nohup $PYTHON train_hvae.py --config ... > train.log 2>&1 < /dev/null &
set -e
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON="${PYTHON:-python}"
CFG=rebuttal/prior_vs_endstop/configs
FIG=rebuttal/prior_vs_endstop/figures
SAVE=saved/rebuttal/prior_vs_endstop
EVAL=rebuttal/greenhyperspectra/eval_greenhyperspectra.py

# -------------------- 1. (re)generate the config --------------------
# Clones rebuttal/greenhyperspectra/configs/HVAE_PROSAIL_C_ghs.json (read-only)
# and adds: balance_kld_z_phy = 0, edge_penalty_weight = 1 + edge_penalty_dynamic.
# $PYTHON rebuttal/prior_vs_endstop/make_configs.py

# -------------------- 2. train --------------------
# $PYTHON train_hvae.py --config $CFG/HVAE_PROSAIL_C_GHS_endstop.json

# -------------------- 2b. quick debug run --------------------
# Two epochs, wandb off, separate experiment name, to check the pipeline.
# $PYTHON train_hvae.py --config $CFG/_debug_HVAE_PROSAIL_C_GHS_endstop.json

# -------------------- 3. evaluate against the measured traits --------------------
# Writes <name>_preds.csv, <name>_metrics.json and <name>_scatter.png into $FIG.
# Replace MMDD_HHMMSS with the run directory printed during training.
# $PYTHON $EVAL \
#   --resume $SAVE/HVAE_PROSAIL_C_GHS_endstop/MMDD_HHMMSS/models/model_best.pth \
#   --out_dir $FIG

# Newest run directory automatically:
# d=$(ls -td $SAVE/HVAE_PROSAIL_C_GHS_endstop/*/ | head -1)
# $PYTHON $EVAL --resume ${d}models/model_best.pth --out_dir $FIG

# -------------------- 4. the baseline it is compared against --------------------
# Not retrained here: the published HVAE-C run already exists. This only
# regenerates its predictions CSV in $FIG so step 5 can diff against it.
# $PYTHON $EVAL \
#   --resume saved/greenhyperspectra/HVAE_PROSAIL_C_GHS/0728_224154/models/model_best.pth \
#   --out_dir $FIG

# -------------------- 5. the comparison table --------------------
# nMAE per trait (truth and prediction scaled to [0,1] by physical range), which
# is the metric that matters here -- these models are unsupervised, so their
# predictions carry a scale/offset bias that dominates R2.
# $PYTHON rebuttal/compare_ghs_variants.py --dir $FIG --baseline HVAE_PROSAIL_C_GHS

# For reference, PILA-C's own numbers come from the same script pointed at the
# other study's figures folder:
# $PYTHON rebuttal/compare_ghs_variants.py \
#   --dir rebuttal/consistency_loss/figures --baseline PILA_PROSAIL_C_GHS

echo "Nothing ran: every command in $(basename "$0") is commented out by design."
echo "Open it and uncomment the step you want."
