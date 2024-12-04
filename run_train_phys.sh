#!/bin/sh
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# PhysVAE
# Train AE_RTM_A (VAE)
python3 -m train_phys --config configs/phys/AE_RTM_A.json
# python -m ptvsd --host 127.0.0.1 --port 5678 --wait -m train_phys --config configs/phys/AE_RTM_A.json

# Train AE_RTM_B (encoder being replaced with RTM)
python3 -m train_phys --config configs/phys/AE_RTM_B.json

# Train AE_RTM_C (encoder being replaced with RTM + correction layer)
python3 -m train_phys  --config configs/phys/AE_RTM_C.json 
# python -m ptvsd --host 127.0.0.1 --port 5678 --wait -m train_phys --config configs/phys/AE_RTM_C.json
