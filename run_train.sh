#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Train AE_RTM_A (classical autoencoder)
# python3 train.py --config configs/AE_RTM_A.json

# Train AE_RTM_B (decoder being replaced with RTM)
# python3 train.py --config configs/AE_RTM_B.json

# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
# python3 train.py --config configs/AE_RTM_C.json

# Train NN_RTM_D (baseline: neural network regressor)
# python3 train.py --config configs/NN_RTM_D.json

# Train AE_Mogi_A (classical autoencoder)
# python3 train.py --config configs/AE_Mogi_A.json

# Train AE_Mogi_B (decoder being replaced with Mogi)
# python3 train.py --config configs/AE_Mogi_B.json

# Train AE_Mogi_C (encoder being replaced with Mogi + correction layer)
# python3 train.py --config configs/AE_Mogi_C.json



