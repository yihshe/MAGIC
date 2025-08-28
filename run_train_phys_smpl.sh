#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

#---------------WYTHAM DATA-----------------
# Simplified PhysVAE Framework
# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
python3 -m train_phys_smpl --config configs/phys_smpl/AE_RTM_C_wytham.json

# Alternative: Test the trained model
# python3 -m test_phys_rtm_smpl \
#         --config saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_SMPL/config.json \
#         --resume saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_SMPL/checkpoint-epochX.pth \
#         --insitu