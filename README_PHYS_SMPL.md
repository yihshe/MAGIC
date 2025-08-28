# Simplified PhysVAE Framework

This directory contains a simplified and more interpretable implementation of the PhysVAE framework for model inversion tasks, specifically designed for the Wytham Wood dataset.

## Overview

The simplified PhysVAE framework provides a cleaner, more maintainable version of the original PhysVAE implementation with the following key improvements:

### Key Features

1. **Simplified Architecture**: Removed complex unmixing paths and components for better interpretability
2. **U-space Representation**: Physics parameters are represented in unbounded u-space and transformed to (0,1) z-space via sigmoid
3. **Additive Residual Correction**: Uses gated additive residual instead of complex multiplicative corrections
4. **Two-stage Training**: 
   - Stage A: Synthetic bootstrap for physics parameter learning
   - Stage B: Full VAE training with KL divergence
5. **Enhanced Monitoring**: Better logging, validation metrics, and training stability

### Key Changes from Original

- **KL divergence computed in u-space** for physics parameters
- **Simplified decoder** with explicit additive residual and gate
- **Removed unmixing path** for cleaner ablation studies
- **Better initialization** of gate parameters
- **Enhanced error handling** and validation
- **Gradient clipping** for training stability

## File Structure

```
├── configs/phys_smpl/
│   └── AE_RTM_C_wytham.json          # Configuration for Wytham Wood experiment
├── model/
│   └── model_phys_smpl.py            # Simplified PhysVAE model implementation
├── trainer/
│   └── trainer_phys_smpl.py          # Enhanced trainer with better monitoring
├── physics/                          # Forward physical models
│   ├── rtm/                          # PyTorch implementation of RTM model
│   ├── mogi/                         # PyTorch implementation of Mogi model
│   ├── rtm_numpy/                    # NumPy implementation of RTM model
│   └── dpm/                          # DPM model implementation
├── train_phys_smpl.py                # Training script
├── test_phys_rtm_smpl.py             # Testing script
└── run_train_phys_smpl.sh            # Training execution script
```

## Configuration

The main configuration file `configs/phys_smpl/AE_RTM_C_wytham.json` includes:

- **Model Architecture**: Simplified PhysVAE with RTM physics model
- **Training Parameters**: 150 epochs, cosine annealing LR scheduler
- **Physics Parameters**: 7 RTM parameters (N, cab, cw, cm, LAI, LAIu, fc)
- **Auxiliary Parameters**: 2 auxiliary dimensions for residual correction
- **Training Stages**: 20 epochs pretraining, 50 epochs KL warmup

## Usage

### Training

```bash
# Run training with the Wytham Wood configuration
python train_phys_smpl.py --config configs/phys_smpl/AE_RTM_C_wytham.json

# Or use the provided script
bash run_train_phys_smpl.sh
```

### Testing

```bash
# Test on validation set
python test_phys_rtm_smpl.py \
    --config saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/MMDD_HHMMSS/models/config.json \
    --resume saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/MMDD_HHMMSS/models/checkpoint-epochX.pth

# Test on in-situ data
python test_phys_rtm_smpl.py \
    --config saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/MMDD_HHMMSS/models/config.json \
    --resume saved/rtm/PHYS_VAE_RTM_C_WYTHAM_SMPL/MMDD_HHMMSS/models/checkpoint-epochX.pth \
    --insitu
```

## Model Architecture

### Encoder
- **Feature Extractor**: Shared backbone for feature extraction
- **Physics Encoder**: Maps features to u-space physics parameters
- **Auxiliary Encoder**: Maps features to auxiliary parameters

### Decoder
- **Physics Model**: RTM radiative transfer model
- **Residual Correction**: Gated additive residual for model imperfections
- **Context Network**: Processes physics + auxiliary context

### Training Stages

1. **Stage A (Pretraining)**: 
   - Uses synthetic data loss to bootstrap physics parameter learning
   - No KL divergence penalty
   - Duration: `epochs_pretrain` epochs

2. **Stage B (Full Training)**:
   - Full VAE training with reconstruction and KL losses
   - KL warmup over `kl_warmup_epochs` epochs
   - Gate penalty for controlling residual correction

## Monitoring

The framework provides comprehensive monitoring through:

- **WandB Integration**: Automatic logging of all metrics
- **Training Metrics**: Loss, reconstruction loss, KL loss, gate mean
- **Validation Metrics**: Reconstruction and KL losses
- **Physics Parameter Statistics**: Per-dimension u-space statistics
- **Gradient Monitoring**: Gradient clipping and stabilization

## Directory Structure

### New Saving Structure

The framework now uses a unified directory structure for experiments:

```
saved/
├── rtm/
│   └── EXPERIMENT_NAME/
│       └── MMDD_HHMMSS/
│           ├── models/           # Model checkpoints and config
│           │   ├── checkpoint-epochX.pth
│           │   ├── config.json
│           │   └── plots/
│           └── log/              # Training logs and tensorboard
│               ├── events.out.tfevents
│               └── train.log
└── mogi/
    └── EXPERIMENT_NAME/
        └── MMDD_HHMMSS/
            ├── models/
            └── log/
```

This structure makes it easier to find all files related to a specific experiment run.

### Physics Models

All forward physical models are now organized under the `physics/` directory:

- **`physics/rtm/`**: PyTorch implementation of RTM radiative transfer model
- **`physics/mogi/`**: PyTorch implementation of Mogi deformation model
- **`physics/rtm_numpy/`**: NumPy implementation of RTM model
- **`physics/dpm/`**: DPM model implementation

## Key Parameters

- `balance_gate`: Weight for gate penalty (default: 0.001)
- `balance_data_aug`: Weight for synthetic data loss (default: 1.0)
- `kl_warmup_epochs`: KL divergence warmup duration (default: 50)
- `epochs_pretrain`: Pretraining duration (default: 20)
- `grad_clip_norm`: Gradient clipping norm (default: 1.0)

## Ablation Studies

The simplified framework is designed to support ablation studies:

- **No Physics**: Set `no_phy: true` to disable physics model
- **No Auxiliary**: Set `dim_z_aux: 0` to disable auxiliary parameters
- **No Residual**: The gate mechanism can be controlled via `balance_gate`
- **Different Physics Models**: Extend to support Mogi and other models

## Troubleshooting

### Common Issues

1. **Training Instability**: 
   - Reduce learning rate
   - Increase gradient clipping norm
   - Check data normalization

2. **Poor Convergence**:
   - Increase pretraining epochs
   - Adjust KL warmup duration
   - Check physics parameter ranges

3. **Memory Issues**:
   - Reduce batch size
   - Reduce number of workers
   - Use gradient accumulation if needed

## Future Improvements

- [ ] Add uncertainty quantification
- [ ] Support for additional physics models
- [ ] Comprehensive ablation study configurations
- [ ] Model interpretability tools
- [ ] Performance optimization for large datasets
