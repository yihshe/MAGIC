#!/bin/bash
# Synthetic-control rank sweep (reviewer: does the low-rank residual recover a
# KNOWN bias?). Same style as ../greenhyperspectra/run_greenhyperspectra.sh:
# every command is commented out -- uncomment the one you want to run.
#
# Run from the repo root:   ./rebuttal/synthetic_control/run_sweep.sh
# Override the interpreter: PYTHON=python3 ./rebuttal/synthetic_control/run_sweep.sh
#
# Mogi data is generated with an intrinsic bias of known rank K, then PILA is
# trained with residual rank k. 3 datasets (K = 1, 2, 4) x 6 model ranks
# (k = 0, 1, 2, 4, 6, 8) = 18 runs. k = 0 means dim_z_aux = 0: pure physics, no
# residual, i.e. the bias is left uncorrected.
#
# The 18 runs are independent: run a block from its own terminal, or append ' &'
# to launch a few at once.
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
CFG=rebuttal/synthetic_control/configs
SAVE=saved/mogi_synth

# -------------------- 1. generate the datasets (one-off) --------------------
# Writes data/processed/mogi_synth/K{1,2,4}/. Clean (no observational noise):
# $PYTHON rebuttal/synthetic_control/gen_synth_mogi.py

# A noisier variant, written to K{1,2,4}_realistic/ (pink + white noise):
# $PYTHON rebuttal/synthetic_control/gen_synth_mogi.py --noise realistic

# Only some ranks:
# $PYTHON rebuttal/synthetic_control/gen_synth_mogi.py --ranks 1 4

# -------------------- 1b. sanity-check the datasets --------------------
# Format parity against the real data, bias/physics orthogonality, and the
# intrinsic rank actually realised in raw and standardized space.
# $PYTHON rebuttal/synthetic_control/inspect_synth.py
# $PYTHON rebuttal/synthetic_control/inspect_synth.py --suffix _realistic

# -------------------- 2. (re)generate the configs --------------------
# Clones the established Mogi rank-ablation config, swapping only name,
# save_dir, data paths, dim_z_aux and residual_rank.
# $PYTHON rebuttal/synthetic_control/make_configs.py

# -------------------- 3. train --------------------
# K = 1 (dataset bias is rank 1)
# $PYTHON train_pila.py --config $CFG/SYN_K1_r0.json
# $PYTHON train_pila.py --config $CFG/SYN_K1_r1.json
# $PYTHON train_pila.py --config $CFG/SYN_K1_r2.json
# $PYTHON train_pila.py --config $CFG/SYN_K1_r4.json
# $PYTHON train_pila.py --config $CFG/SYN_K1_r6.json
# $PYTHON train_pila.py --config $CFG/SYN_K1_r8.json

# K = 2
# $PYTHON train_pila.py --config $CFG/SYN_K2_r0.json
# $PYTHON train_pila.py --config $CFG/SYN_K2_r1.json
# $PYTHON train_pila.py --config $CFG/SYN_K2_r2.json
# $PYTHON train_pila.py --config $CFG/SYN_K2_r4.json
# $PYTHON train_pila.py --config $CFG/SYN_K2_r6.json
# $PYTHON train_pila.py --config $CFG/SYN_K2_r8.json

# K = 4
# $PYTHON train_pila.py --config $CFG/SYN_K4_r0.json
# $PYTHON train_pila.py --config $CFG/SYN_K4_r1.json
# $PYTHON train_pila.py --config $CFG/SYN_K4_r2.json
# $PYTHON train_pila.py --config $CFG/SYN_K4_r4.json
# $PYTHON train_pila.py --config $CFG/SYN_K4_r6.json
# $PYTHON train_pila.py --config $CFG/SYN_K4_r8.json

# One whole K block at a time:
# for k in 0 1 2 4 6 8; do $PYTHON train_pila.py --config $CFG/SYN_K1_r$k.json; done

# -------------------- 4. test / analyzer CSV --------------------
# Writes model_best_testset_analyzer.csv next to the checkpoint; eval_sweep.py
# and plot_params.py read exactly these files, so this step is required.
# Replace MMDD_HHMMSS with the run directory printed during training.
# $PYTHON test_pila_mogi.py \
#   --config $SAVE/SYN_K1_r4/MMDD_HHMMSS/models/config.json \
#   --resume $SAVE/SYN_K1_r4/MMDD_HHMMSS/models/model_best.pth

# Every run that has trained, newest run directory each, skipping finished ones:
# for name in $(ls $CFG | sed 's/\.json$//'); do
#   d=$(ls -td $SAVE/$name/*/ 2>/dev/null | head -1)
#   [ -n "$d" ] || continue
#   [ -f "${d}models/model_best_testset_analyzer.csv" ] && continue
#   $PYTHON test_pila_mogi.py --config ${d}models/config.json --resume ${d}models/model_best.pth
# done

# -------------------- 5. analysis --------------------
# Metrics and figures against the known ground truth -> figures/sweep_metrics.csv,
# figures/sweep_results.png.
$PYTHON -m pdb rebuttal/synthetic_control/eval_sweep.py

# Per-variable recovery of the Mogi parameters, per K_true and model rank k ->
# figures/param_accuracy.csv, figures/param_recovery.png.
# $PYTHON rebuttal/synthetic_control/plot_params.py

echo "Nothing ran: every command in $(basename "$0") is commented out by design."
echo "Open it and uncomment the step you want."
