# Simplified PhysVAE Configuration Files

This directory contains configuration files for running experiments with the simplified PhysVAE framework across different datasets and model variants.

## Experiment Overview

The experiments are organized by:
- **Physics Model**: RTM (Radiative Transfer Model) or Mogi (deformation model)
- **Dataset**: Wytham (UK), Austria, or Seismic data
- **Model Variant**: A, B, or C

### Model Variants

- **A (Vanilla Autoencoder)**: `no_phy: true`, uses only auxiliary parameters
- **B (Physics Only)**: `no_phy: false`, `dim_z_aux: 0`, uses only physics parameters
- **C (Physics + Correction)**: `no_phy: false`, `dim_z_aux: 2`, uses physics parameters with auxiliary correction

## Configuration Files

### RTM Experiments (Radiative Transfer Model)

#### Wytham Data (UK)
- `AE_RTM_A_wytham.json` - Vanilla autoencoder for Wytham RTM data
- `AE_RTM_B_wytham.json` - Physics-only model for Wytham RTM data  
- `AE_RTM_C_wytham.json` - Physics + correction model for Wytham RTM data

#### Austria Data
- `AE_RTM_A_austria.json` - Vanilla autoencoder for Austria RTM data
- `AE_RTM_B_austria.json` - Physics-only model for Austria RTM data
- `AE_RTM_C_austria.json` - Physics + correction model for Austria RTM data

### Mogi Experiments (Deformation Model)

#### Seismic Data
- `AE_Mogi_A.json` - Vanilla autoencoder for seismic Mogi data
- `AE_Mogi_B.json` - Physics-only model for seismic Mogi data
- `AE_Mogi_C.json` - Physics + correction model for seismic Mogi data

## Key Configuration Parameters

### Architecture
- **Model Type**: `PHYS_VAE_SMPL` (simplified PhysVAE)
- **Input Dimensions**: 11 for RTM, 36 for Mogi
- **Hidden Dimensions**: 7 for RTM, 4 for Mogi
- **Activation**: ELU

### Training
- **Epochs**: 150
- **Learning Rate**: 0.0003
- **Scheduler**: CosineAnnealingLR with T_max=150
- **Batch Size**: 256 for RTM, 64 for Mogi
- **Gradient Clipping**: 1.0

### Physics Parameters
- **RTM**: 7 parameters (N, cab, cw, cm, LAI, LAIu, fc)
- **Mogi**: 4 parameters (xcen, ycen, d, dV)

### Auxiliary Parameters
- **A Variants**: 9 for RTM, 4 for Mogi
- **B Variants**: 0 (no auxiliary)
- **C Variants**: 2 (correction only)

## Data Paths

### RTM Data
- **Wytham**: `data/processed/rtm/wytham/insitu_period_subset/`
- **Austria**: `data/processed/rtm/real/`

### Mogi Data
- **Seismic**: `data/processed/mogi/`

## Usage

To run an experiment:

```bash
# Example: Run Wytham RTM experiment C
python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_C_wytham.json

# Example: Run Austria RTM experiment B
python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_B_austria.json

# Example: Run Mogi experiment A
python train_phys_smpl.py --config configs/phys_smpl/AE_Mogi_A.json
```

## Experiment Naming Convention

The experiment names follow the pattern:
`PHYS_VAE_{MODEL}_{VARIANT}_{DATASET}_SMPL`

Examples:
- `PHYS_VAE_RTM_A_WYTHAM_SMPL`
- `PHYS_VAE_RTM_C_AUSTRIA_SMPL`
- `PHYS_VAE_MOGI_B_SMPL`

## Monitoring

All experiments are configured with:
- **WandB Integration**: Enabled for experiment tracking
- **Gradient Clipping**: For training stability
- **Early Stopping**: 20 epochs patience
- **Save Period**: Every 10 epochs

## Ablation Study Support

The simplified framework supports systematic ablation studies:
- Compare A vs B vs C variants
- Compare different datasets
- Compare different physics models
- Analyze the effect of auxiliary parameters
