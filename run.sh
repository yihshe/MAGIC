#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Train AE_RTM_A (classical autoencoder)
# python3 train.py --config configs/AE_RTM_A.json

# Train AE_RTM_B (decoder being replaced with RTM)
# python3 -m pdb train.py --config configs/AE_RTM_B.json

# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
python3 train.py --config configs/AE_RTM_C.json

