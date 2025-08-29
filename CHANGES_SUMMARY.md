# Directory Structure Reorganization Summary

This document summarizes the changes made to reorganize the PhysVAE framework directory structure.

## Changes Made

### 1. Unified Experiment Directory Structure

**Problem**: Previously, logs and models were stored in separate top-level directories:
```
saved/
├── models/
│   └── EXPERIMENT_NAME/
│       └── MMDD_HHMMSS/
│           ├── checkpoint-epochX.pth
│           └── config.json
└── log/
    └── EXPERIMENT_NAME/
        └── MMDD_HHMMSS/
            └── train.log
```

**Solution**: Modified `parse_config.py` to create a unified structure:
```
saved/
└── EXPERIMENT_NAME/
    └── MMDD_HHMMSS/
        ├── models/           # Model checkpoints and config
        │   ├── checkpoint-epochX.pth
        │   ├── config.json
        │   └── plots/
        └── log/              # Training logs and tensorboard
            ├── events.out.tfevents
            └── train.log
```

**Files Modified**:
- `parse_config.py`: Updated save directory path construction

### 2. Physics Models Reorganization

**Problem**: Physics models were scattered across the root directory:
```
├── rtm_torch/
├── mogi/
├── rtm_numpy/
└── dpm/
```

**Solution**: Created a unified `physics/` directory:
```
physics/
├── rtm/              # Renamed from rtm_torch/
├── mogi/
├── rtm_numpy/
└── dpm/
```

**Files Modified**:
- All import statements updated to use `physics.rtm.rtm` instead of `rtm_torch.rtm`
- All import statements updated to use `physics.mogi.mogi` instead of `mogi.mogi`
- All import statements updated to use `physics.dpm.dpm` instead of `dpm.dpm`

**Files Updated**:
- `model/model_phys_smpl.py`
- `model/model_phys.py`
- `model/model.py`
- `test_phys_rtm_smpl.py`
- `test_phys_rtm.py`
- `test_phys_mogi.py`
- `test_NN_RTM.py`
- `test_AE_Mogi.py`
- `test_AE_RTM.py`
- `utils/rtm_unit_test.py`
- All internal imports within `physics/rtm/Resources/PROSAIL/` files
- `README.md`
- `README_PHYS_SMPL.md`

## Benefits

### 1. Better Organization
- All physics models are now logically grouped under `physics/`
- Clear separation between different types of models (RTM, Mogi, DPM)
- Easier to find and maintain physics model implementations

### 2. Improved Experiment Management
- All files related to an experiment are now in the same directory
- No need to navigate between separate `models/` and `log/` directories
- Easier to copy, move, or delete entire experiments
- Better organization for experiment comparison and analysis

### 3. Cleaner Root Directory
- Reduced clutter in the main project directory
- More intuitive project structure
- Easier for new users to understand the codebase

## Migration Notes

### For Existing Experiments
Existing experiments in the old structure will continue to work, but new experiments will use the new structure. To migrate existing experiments:

1. **Manual Migration**: Copy files from old structure to new structure
2. **Update Paths**: Update any hardcoded paths in analysis scripts
3. **Test**: Verify that all imports and file references work correctly

### For Development
- New experiments will automatically use the new structure
- All import statements have been updated to use the new paths
- The `README_PHYS_SMPL.md` has been updated with the new structure

## Verification

The new structure has been tested and verified:
- ✓ Directory structure created correctly
- ✓ All import paths updated
- ✓ Save directory structure modified
- ✓ Documentation updated

## Future Considerations

1. **Backward Compatibility**: Consider maintaining backward compatibility for existing scripts
2. **Migration Script**: Create a script to automatically migrate existing experiments
3. **Documentation**: Update any additional documentation that references the old structure
4. **CI/CD**: Update any CI/CD pipelines that might reference the old paths
