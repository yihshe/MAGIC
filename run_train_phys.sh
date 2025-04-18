#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
#---------------WYTHAM DATA-----------------
# PhysVAE
# Train AE_RTM_A (VAE)
# python3 -m train_phys --config configs/phys_wytham/AE_RTM_A.json
# python3 -m pdb test_phys_rtm.py \
#         --config saved/rtm/models/PHYS_VAE_RTM_A_WYTHAM/0323_204514/config.json \
#         --resume saved/rtm/models/PHYS_VAE_RTM_A_WYTHAM/0323_204514/model_best.pth \

# Train AE_RTM_B (encoder being replaced with RTM)
# python3 -m train_phys --config configs/phys_wytham/AE_RTM_B.json
# python3 -m test_phys_rtm \
#         --config saved/rtm/models/PHYS_VAE_RTM_B_WYTHAM/0323_222945/config.json \
#         --resume saved/rtm/models/PHYS_VAE_RTM_B_WYTHAM/0323_222945/model_best.pth \
#         --insitu


# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
# python3 -m train_phys --config configs/phys_wytham/AE_RTM_C.json

# NOTE changes made for the following mode: 1) KL weight, 2) prior term
# python3 -m test_phys_rtm \
#         --config saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_1/0406_094832/config.json \
#         --resume saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_1/0406_094832/model_best.pth \
#         --insitu

# python3 -m test_phys_rtm \
#         --config saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_3/0406_105747/config.json \
#         --resume saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_3/0406_105747/model_best.pth \
#         --insitu

# python3 -m test_phys_rtm \
#         --config saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_3_prior_std0.1/0406_114131/config.json \
#         --resume saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_3_prior_std0.1/0406_114131/model_best.pth \
#         --insitu

python3 -m test_phys_rtm \
        --config saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_fixed1.5_prior_std0.1/0407_102159/config.json \
        --resume saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_fixed1.5_prior_std0.1/0407_102159/model_best.pth \
        --insitu
    
python3 -m test_phys_rtm \
        --config saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_fixed1.5_prior_std0.2/0407_120629/config.json \
        --resume saved/rtm/models/PHYS_VAE_RTM_C_WYTHAM_KL_LAIu_fixed1.5_prior_std0.2/0407_120629/model_best.pth \
        --insitu

#---------------AUSTRIA DATA-----------------
# PhysVAE
# Train AE_RTM_A (VAE)
# python3 -m train_phys --config configs/phys/AE_RTM_A.json
# python -m ptvsd --host 127.0.0.1 --port 5678 --wait -m train_phys --config configs/phys/AE_RTM_A.json
# python3 -m test_phys_rtm \
#         --config saved/rtm/models/PHYS_VAE_RTM_A/1204_181529/config.json \
#         --resume saved/rtm/models/PHYS_VAE_RTM_A/1204_181529/model_best.pth

# Train AE_RTM_B (encoder being replaced with RTM)
# python3 -m train_phys --config configs/phys/AE_RTM_B.json
# python3 -m test_phys_rtm \
#         --config saved/rtm/models/PHYS_VAE_RTM_B/1204_195439/config.json \
#         --resume saved/rtm/models/PHYS_VAE_RTM_B/1204_195439/model_best.pth

# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
# python3 -m train_phys  --config configs/phys/AE_RTM_C.json 
# python -m ptvsd --host 127.0.0.1 --port 5678 --wait -m train_phys --config configs/phys/AE_RTM_C.json

# python3 -m ptvsd --host 127.0.0.1 --port 5678 --wait -m test_phys_rtm \
#         --config saved/rtm/models/PHYS_VAE_RTM_C/1205_060657/config.json \
#         --resume saved/rtm/models/PHYS_VAE_RTM_C/1205_060657/model_best.pth

# Train AE_Mogi_A
# python3 -m train_phys --config configs/phys/AE_Mogi_A.json
# python3 -m test_phys_mogi \
#         --config saved/mogi/models/PHYS_VAE_Mogi_A/1209_174306/config.json \
#         --resume saved/mogi/models/PHYS_VAE_Mogi_A/1209_174306/model_best.pth

# Train AE_Mogi_B
# python3 -m train_phys --config configs/phys/AE_Mogi_B.json
# python3 -m test_phys_mogi \
#         --config saved/mogi/models/PHYS_VAE_Mogi_B/1209_174339/config.json \
#         --resume saved/mogi/models/PHYS_VAE_Mogi_B/1209_174339/model_best.pth
# python3 -m test_phys_mogi \
#         --config saved/mogi/models/PHYS_VAE_Mogi_B_seq/1210_093812/config.json \
#         --resume saved/mogi/models/PHYS_VAE_Mogi_B_seq/1210_093812/model_best.pth

# Train AE_Mogi_C 
# python3 -m train_phys --config configs/phys/AE_Mogi_C.json
# python3 -m test_phys_mogi \
#         --config saved/mogi/models/PHYS_VAE_Mogi_C/1209_174455/config.json \
#         --resume saved/mogi/models/PHYS_VAE_Mogi_C/1209_174455/model_best.pth

# python3 -m test_phys_mogi \
#         --config saved/mogi/models/PHYS_VAE_Mogi_C_seq/1210_094000/config.json \
#         --resume saved/mogi/models/PHYS_VAE_Mogi_C_seq/1210_094000/model_best.pth
