#!/bin/bash
# Seed-variance experiment on the Austrian dataset (reviewer: "Figure 8 could be
# anecdotal without multiple runs"). Same style as ../greenhyperspectra/
# run_greenhyperspectra.sh: every command is commented out -- uncomment the one
# you want to run.
#
# Run from the repo root:   ./rebuttal/seed_variance/run_seed_variance.sh
# Override the interpreter: PYTHON=python3 ./rebuttal/seed_variance/run_seed_variance.sh
#
# PILA and HVAE at three seeds each, everything else identical to the paper runs.
# ~2 h per run (150 epochs, 11 S2 bands, CPU). The six are independent: run them
# from separate terminals, or append ' &' to launch several at once. 8 threads is
# the sweet spot per run.
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
CFG=rebuttal/seed_variance/configs
SAVE=saved/rebuttal/seed_variance

# -------------------- 1. (re)generate the configs --------------------
# Copies configs/phys_smpl/PILA_RTM_C_austria.json and configs/phys/
# HVAE_RTM_C_austria.json, changing only name, seed and trainer.save_dir.
# $PYTHON rebuttal/seed_variance/make_configs.py

# -------------------- 2. train --------------------
# Seed 123 is the seed hardcoded in the train scripts, i.e. the published run;
# it is included so all three seeds come from one code state. Skip it if you
# would rather reuse the existing checkpoint under saved/rtm/.

# PILA (our method)
# $PYTHON train_pila.py --config $CFG/PILA_RTM_C_AUSTRIA_s123.json
# $PYTHON train_pila.py --config $CFG/PILA_RTM_C_AUSTRIA_s240.json
# $PYTHON train_pila.py --config $CFG/PILA_RTM_C_AUSTRIA_s318.json

# HVAE (baseline)
# $PYTHON train_hvae.py --config $CFG/HVAE_RTM_C_AUSTRIA_s123.json
# $PYTHON train_hvae.py --config $CFG/HVAE_RTM_C_AUSTRIA_s240.json
# $PYTHON train_hvae.py --config $CFG/HVAE_RTM_C_AUSTRIA_s318.json

# -------------------- 3. test / analyzer CSV --------------------
# Writes model_best_testset_analyzer.csv next to the checkpoint. jm_seeds.py
# reads exactly these six files, so this step is required before step 4.
# Replace MMDD_HHMMSS with the run directory printed during training.

# $PYTHON test_pila_rtm.py \
#   --config $SAVE/PILA_RTM_C_AUSTRIA_s123/MMDD_HHMMSS/models/config.json \
#   --resume $SAVE/PILA_RTM_C_AUSTRIA_s123/MMDD_HHMMSS/models/model_best.pth

# $PYTHON test_hvae_rtm.py \
#   --config $SAVE/HVAE_RTM_C_AUSTRIA_s123/MMDD_HHMMSS/models/config.json \
#   --resume $SAVE/HVAE_RTM_C_AUSTRIA_s123/MMDD_HHMMSS/models/model_best.pth

# Same for s240 and s318. To fill in the newest run directory automatically:
# for name in PILA_RTM_C_AUSTRIA_s123 PILA_RTM_C_AUSTRIA_s240 PILA_RTM_C_AUSTRIA_s318; do
#   d=$(ls -td $SAVE/$name/*/ | head -1)
#   $PYTHON test_pila_rtm.py --config ${d}models/config.json --resume ${d}models/model_best.pth
# done
# for name in HVAE_RTM_C_AUSTRIA_s123 HVAE_RTM_C_AUSTRIA_s240 HVAE_RTM_C_AUSTRIA_s318; do
#   d=$(ls -td $SAVE/$name/*/ | head -1)
#   $PYTHON test_hvae_rtm.py --config ${d}models/config.json --resume ${d}models/model_best.pth
# done

# -------------------- 4. aggregate across seeds --------------------
# Per-seed JM matrices -> mean +- std, plus the paired PILA-vs-HVAE comparison.
# Writes figures/: <p>_summary.txt, <p>_group_stats.csv, <p>_seeds.png,
# <p>_{PILA,HVAE}_{mean,std}.csv.

# As plotted in Figure 8 (JM saturates at 2):
# $PYTHON rebuttal/seed_variance/jm_seeds.py --space jm

# On the underlying Bhattacharyya distance (unsaturated; the between-group
# differences sit near the JM ceiling, which compresses them):
# $PYTHON rebuttal/seed_variance/jm_seeds.py --space b

# A subset of seeds, or one family only:
# $PYTHON rebuttal/seed_variance/jm_seeds.py --seeds 240 318 --families PILA

echo "Nothing ran: every command in $(basename "$0") is commented out by design."
echo "Open it and uncomment the step you want."
