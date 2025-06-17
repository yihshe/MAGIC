#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
#---------------WYTHAM DATA-----------------
# PhysVAE
# Train AE_RTM_A (VAE)
# python3 -m train_phys --config configs/phys_wytham/AE_RTM_A.json

# Train AE_RTM_B (encoder being replaced with RTM)
# python3 -m train_phys --config configs/phys_wytham/AE_RTM_B.json

# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
# python3 -m pdb train_phys_smpl.py --config configs/phys_wytham_smpl/AE_RTM_C.json --resume /maps/ys611/MAGIC/saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_SMPL/0612_155633/checkpoint-epoch100.pth

python3 -m pdb test_phys_rtm_smpl.py \
        --config saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_SMPL/0611_173302/config.json \
        --resume saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_SMPL/0611_173302/checkpoint-epoch100.pth \
        # --insitu