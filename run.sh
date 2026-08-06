#!/bin/bash
set -e
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Example commands used in the paper. Uncomment the ones you want.
# You can override the python executable with: PYTHON=python3 ./run.sh
PYTHON="${PYTHON:-python}"

# -------------------- PILA (our method) --------------------
# Mogi inversion (PILA)
# $PYTHON train_pila.py --config configs/phys_smpl/PILA_Mogi_C.json

# RTM inversion (PILA) - Austria and Wytham datasets
# $PYTHON train_pila.py --config configs/phys_smpl/PILA_RTM_C_austria.json
# $PYTHON train_pila.py --config configs/phys_smpl/PILA_RTM_C_wytham.json

# -------------------- HVAE (baseline) --------------------
# Mogi inversion (HVAE)
# $PYTHON train_hvae.py --config configs/phys/HVAE_Mogi_C.json

# RTM inversion (HVAE) - Austria and Wytham datasets
# $PYTHON train_hvae.py --config configs/phys/HVAE_RTM_C_austria.json
# $PYTHON train_hvae.py --config configs/phys/HVAE_RTM_C_wytham.json

# -------------------- Evaluation examples --------------------
# PILA evaluation (Mogi)
# $PYTHON test_pila_mogi.py \
#   --config saved/mogi/EXPERIMENT_NAME/MMDD_HHMMSS/models/config.json \
#   --resume saved/mogi/EXPERIMENT_NAME/MMDD_HHMMSS/models/model_best.pth

# PILA evaluation (RTM)
# $PYTHON test_pila_rtm.py \
#   --config saved/mogi/EXPERIMENT_NAME/MMDD_HHMMSS/models/config.json \
#   --resume saved/mogi/EXPERIMENT_NAME/MMDD_HHMMSS/models/model_best.pth
#   --insitu

$PYTHON test_pila_rtm.py \
  --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/0802_145450_klp0_edge0/models/config.json \
  --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/0802_145450_klp0_edge0/models/model_best.pth

# HVAE evaluation (Mogi)
# $PYTHON test_hvae_mogi.py \
#   --config saved/mogi/EXPERIMENT_NAME/MMDD_HHMMSS/models/config.json \
#   --resume saved/mogi/EXPERIMENT_NAME/MMDD_HHMMSS/models/model_best.pth

# HVAE evaluation (RTM)
# $PYTHON test_hvae_rtm.py \
#   --config saved/rtm/EXPERIMENT_NAME/MMDD_HHMMSS/models/config.json \
#   --resume saved/rtm/EXPERIMENT_NAME/MMDD_HHMMSS/models/model_best.pth