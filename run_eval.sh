#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# # Evaluate AE_RTM_A (classical autoencoder)
python3 test_AE_RTM.py \
        --config pretrained/AE_RTM_A/config.json \
        --resume pretrained/AE_RTM_A/model_best.pth

# # Evaluate AE_RTM_B (decoder being replaced with RTM)
python3 test_AE_RTM.py \
        --config pretrained/AE_RTM_B/config.json \
        --resume pretrained/AE_RTM_B/model_best.pth

# Evaluate AE_RTM_C (encoder being replaced with RTM + correction layer)
python3 test_AE_RTM.py \
        --config pretrained/AE_RTM_C/config.json \
        --resume pretrained/AE_RTM_C/model_best.pth

# Evaluate NN_RTM_D (baseline: neural network regressor)
python3 test_NN_RTM.py \
        --config pretrained/NN_RTM_D/config_infer_real.json \
        --resume pretrained/NN_RTM_D/model_best.pth

# # Evaluate AE_Mogi_A (classical autoencoder)
python3 test_AE_Mogi.py \
        --config pretrained/AE_Mogi_A/config.json \
        --resume pretrained/AE_Mogi_A/model_best.pth

# # Evaluate AE_Mogi_B (decoder being replaced with Mogi)
python3 test_AE_Mogi.py \
        --config pretrained/AE_Mogi_B/config.json \
        --resume pretrained/AE_Mogi_B/model_best.pth

# # Evaluate AE_Mogi_C (encoder being replaced with Mogi + correction layer)
python3 test_AE_Mogi.py \
        --config pretrained/AE_Mogi_C/config.json \
        --resume pretrained/AE_Mogi_C/model_best.pth