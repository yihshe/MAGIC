import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

from .model_phys import PHYS_VAE
from .model_phys_smpl import PHYS_VAE_SMPL