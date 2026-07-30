#!/bin/bash
# Physical-consistency loss added to PILA, on GreenHyperSpectra (reviewer: "could
# PILA be improved through addition of a physical consistency loss as in HVAE?").
# Same style as ../greenhyperspectra/run_greenhyperspectra.sh: every command is
# commented out -- uncomment the one you want to run.
#
# Run from the repo root:   ./rebuttal/consistency_loss/run_consistency_loss.sh
# Override the interpreter: PYTHON=python3 ./rebuttal/consistency_loss/run_consistency_loss.sh
#
# Three training runs, one per weight of the co-trained consistency term
# (sampled z vs drawn z, as HVAE compares them). ~1.7 h each (150 epochs, 1721
# bands, CPU, 8 threads); they are independent, so run them from separate
# terminals or append ' &'. Requires the GHS dataset built by
# rebuttal/greenhyperspectra/preprocess_greenhyperspectra.py.
#
# NOTE ON LONG RUNS: a plain background job dies when its parent shell exits. To
# survive a closed terminal, wrap the command:
#   setsid nohup $PYTHON train_pila.py --config ... > train.log 2>&1 < /dev/null &
set -e
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON="${PYTHON:-python}"
CFG=rebuttal/consistency_loss/configs
FIG=rebuttal/consistency_loss/figures
SAVE=saved/rebuttal/consistency_loss
EVAL=rebuttal/greenhyperspectra/eval_greenhyperspectra.py

# -------------------- 1. (re)generate the configs --------------------
# Clones rebuttal/greenhyperspectra/configs/PILA_PROSAIL_C_ghs.json (read-only)
# and sets trainer.phys_vae.balance_data_aug_cotrain to each weight below.
# $PYTHON rebuttal/consistency_loss/make_configs.py

# -------------------- 2. train --------------------
# The weights span ~1% of rec_loss (negligible), 5-10% (where HVAE keeps its own
# regularisers) and 50-100% (dominant). No weight-0 baseline is trained: the
# reference is the existing PILA_PROSAIL_C_GHS run, see step 4.

# $PYTHON train_pila.py --config $CFG/PILA_PROSAIL_C_GHS_S1_w0.1.json
# $PYTHON train_pila.py --config $CFG/PILA_PROSAIL_C_GHS_S2_w1.0.json
# $PYTHON train_pila.py --config $CFG/PILA_PROSAIL_C_GHS_S3_w10.json

# -------------------- 2b. quick debug run --------------------
# Two epochs, wandb off, separate experiment name, to check the pipeline.
# $PYTHON train_pila.py --config $CFG/_debug_PILA_PROSAIL_C_GHS_S2_w1.0.json

# -------------------- 3. evaluate against the measured traits --------------------
# Writes <name>_preds.csv, <name>_metrics.json and <name>_scatter.png into $FIG.
# Replace MMDD_HHMMSS with the run directory printed during training.
# $PYTHON $EVAL \
#   --resume $SAVE/PILA_PROSAIL_C_GHS_S1_w0.1/MMDD_HHMMSS/models/model_best.pth \
#   --out_dir $FIG

# All three, newest run directory each:
# for name in PILA_PROSAIL_C_GHS_S1_w0.1 PILA_PROSAIL_C_GHS_S2_w1.0 PILA_PROSAIL_C_GHS_S3_w10; do
#   d=$(ls -td $SAVE/$name/*/ | head -1)
#   $PYTHON $EVAL --resume ${d}models/model_best.pth --out_dir $FIG
# done

# -------------------- 4. the baseline it is compared against --------------------
# Not retrained here: the weight-0 reference is the completed PILA-C run. This
# only regenerates its predictions CSV in $FIG so step 5 can diff against it.
# $PYTHON $EVAL \
#   --resume saved/greenhyperspectra/PILA_PROSAIL_C_GHS/0728_214108/models/model_best.pth \
#   --out_dir $FIG

# -------------------- 5. the comparison table --------------------
# nMAE per trait (truth and prediction scaled to [0,1] by physical range), which
# is the metric that matters here -- these models are unsupervised, so their
# predictions carry a scale/offset bias that dominates R2.
# $PYTHON rebuttal/compare_ghs_variants.py --dir $FIG --baseline PILA_PROSAIL_C_GHS

echo "Nothing ran: every command in $(basename "$0") is commented out by design."
echo "Open it and uncomment the step you want."
