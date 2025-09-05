#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Use the conda environment's Python directly
PYTHON_CMD="/maps-priv/maps/ys611/miniconda3/envs/mres/bin/python"
echo "Using Python: $PYTHON_CMD"

#---------------WYTHAM DATA-----------------
# Simplified PhysVAE Framework
# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
# $PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_wytham.json

$PYTHON_CMD -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_austria.json

# Alternative: Test the trained model
# $PYTHON_CMD -m test_phys_rtm_smpl \
#         --config saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/0831_211338/models/config.json \
#         --resume saved/rtm/PHYS_VAE_RTM_C_AUSTRIA_SMPL/0831_211338/models/model_best.pth \
#         # --insitu

# #---------------ALL EXPERIMENTS-----------------
# # Run Wytham RTM experiments
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_A_wytham.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_B_wytham.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_C_wytham.json

# # Run Austria RTM experiments
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_A_austria.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_B_austria.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_C_austria.json

# # Run Mogi experiments
# python train_phys_smpl.py --config configs/phys_smpl/AE_Mogi_A.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_Mogi_B.json
# python train_phys_smpl.py --config configs/phys_smpl/AE_Mogi_C.json