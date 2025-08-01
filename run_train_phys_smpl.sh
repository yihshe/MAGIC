#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
#---------------WYTHAM DATA-----------------
# PhysVAE
# Train AE_RTM_A (VAE)
# python3 -m train_phys --config configs/phys_wytham/AE_RTM_A.json

# Train AE_RTM_B (encoder being replaced with RTM)
# python3 -m train_phys --config configs/phys_wytham/AE_RTM_B.json

# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
# python3 -m train_phys_smpl --config configs/phys_wytham_smpl/AE_RTM_C.json 
python3 -m train_phys --config configs/phys_wytham/AE_RTM_C.json 

# python3 -m pdb test_phys_rtm.py \
#         --config saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_SMPL_COMPARE/0617_154805_std0.1_kl0.1_otherweights0.1/config.json \
#         --resume saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_SMPL_COMPARE/0617_154805_std0.1_kl0.1_otherweights0.1/checkpoint-epoch40.pth \
        # --insitu