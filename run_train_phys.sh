#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# PhysVAE
# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
python3 -m pdb train_phys.py --config configs/phys/AE_RTM_C.json