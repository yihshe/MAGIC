#!/bin/bash
# GreenHyperSpectra rebuttal experiment. Same style as ../../run.sh:
# every command is commented out -- uncomment the one you want to run.
#
# Run from the repo root:   ./rebuttal/greenhyperspectra/train_greenhyperspectra.sh
# Override the interpreter: PYTHON=python3 ./rebuttal/greenhyperspectra/train_greenhyperspectra.sh
#
# Commands are foreground and one-at-a-time so they can be stepped through in a
# debugger. There is no GPU on this box and the RTM step is serial Python, so
# extra threads do not speed up a single job -- 8 is the sweet spot, and the two
# variant-C runs can simply be launched from two terminals.
set -e
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON="${PYTHON:-python}"
CFG=rebuttal/greenhyperspectra/configs

# -------------------- 1. build the dataset (one-off) --------------------
# Merges labeled_splits/train.csv with two unlabeled shards, standardizes, and
# writes rebuttal/greenhyperspectra/data/. Nothing depends on HyspectraSSL after
# this step. ~5 min, dominated by reading two 208 MB CSVs.
# $PYTHON -m pdb rebuttal/greenhyperspectra/preprocess_greenhyperspectra.py \
#   --ghs_root /maps/ys611/HyspectraSSL --n_splits 2

# -------------------- 2. (re)generate the configs --------------------
# $PYTHON rebuttal/greenhyperspectra/make_configs.py

# -------------------- 3. train (the headline comparison) --------------------
# Variant C = physics + correction, for both models. One run per setting.
# ~1.0 h (PILA) and ~1.2 h (HVAE) for 150 epochs.

# PILA (our method)
# $PYTHON train_pila.py --config $CFG/PILA_PROSAIL_C_ghs.json

# HVAE (baseline)
$PYTHON train_hvae.py --config $CFG/HVAE_PROSAIL_C_ghs.json

# -------------------- 3b. quick debug runs --------------------
# Two epochs each, wandb disabled, to check the pipeline end to end.
# $PYTHON train_pila.py --config $CFG/_debug_PILA_PROSAIL_C_ghs.json
# $PYTHON train_hvae.py --config $CFG/_debug_HVAE_PROSAIL_C_ghs.json

# -------------------- 3c. ablation (reference only, not in the claim) --------
# A = no physics, B = physics without correction.
# $PYTHON train_pila.py --config $CFG/PILA_PROSAIL_A_ghs.json
# $PYTHON train_pila.py --config $CFG/PILA_PROSAIL_B_ghs.json
# $PYTHON train_hvae.py --config $CFG/HVAE_PROSAIL_A_ghs.json
# $PYTHON train_hvae.py --config $CFG/HVAE_PROSAIL_B_ghs.json

# -------------------- 4. evaluate one checkpoint --------------------
# Works for both PILA and HVAE: the family is read from the saved config.json.
# Prints per-trait R2/Pearson/RMSE/nRMSE and writes the 1:1 density plots.
# $PYTHON rebuttal/greenhyperspectra/eval_greenhyperspectra.py \
#   --resume saved/greenhyperspectra/PILA_PROSAIL_C_GHS/MMDD_HHMMSS/models/model_best.pth

# -------------------- 5. the headline PILA-vs-HVAE table --------------------
# Finds the newest PILA_PROSAIL_C_GHS and HVAE_PROSAIL_C_GHS runs, evaluates
# both (cached unless --recompute), and prints the per-trait difference.
# $PYTHON rebuttal/greenhyperspectra/compare_greenhyperspectra.py

echo "Nothing ran: every command in $(basename "$0") is commented out by design."
echo "Open it and uncomment the step you want."
