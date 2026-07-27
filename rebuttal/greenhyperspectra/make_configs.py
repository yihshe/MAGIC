"""
Generate the PILA and HVAE configs for the GreenHyperSpectra experiment.

Six configs, matching the A/B/C ablation used for Wytham and Austria:
  A  no physics          (encoder -> MLP decoder)
  B  physics only        (encoder -> PROSPECT-PRO+4SAIL, no correction)
  C  physics + residual  (PILA low-rank correction / HVAE full-rank correction)

Optimizer, batch size, schedule and epochs follow the existing PILA RTM configs
unchanged. Only the widths and penalty weights are re-sized for the 1721-band
input; see SCALE below.

Usage:  python rebuttal/greenhyperspectra/make_configs.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'configs')
DATA = 'rebuttal/greenhyperspectra/data'

INPUT_DIM = 1721      # retained GHS bands
DIM_Z_PHY = 7         # cab, cw, LAI, cp, cbc, car, anth  (cm derived = cp + cbc; N frozen at 1.5 as in GHS)
DIM_Z_AUX = 4         # auxiliary latents for variant C
RESIDUAL_RANK = 4     # rank of the PILA low-rank correction

# The reconstruction loss sums over the feature axis, so 1721 bands would make
# rec_loss ~156x larger than on 11-band S2 data and silently demote every other
# term. Rather than re-tuning each weight, the reconstruction loss itself is
# normalized per band (trainer.rec_loss_scale) and all existing weights carry
# over unchanged -- including the HVAE KL term, which is weighted against
# rec_loss directly.
# Normalized to the S2 band count the existing weights were tuned at, not simply
# to a per-band mean: HVAE weights its KL term as `kl * exp(x_lnvar)` against
# rec_loss directly, so a plain per-band mean would make that term ~100x
# *stronger* than in the S2 runs instead of leaving it where it was.
REFERENCE_BANDS = 11
REC_LOSS_SCALE = REFERENCE_BANDS / INPUT_DIM
ORTHO_W = 0.1   # unchanged from the S2 configs

EPOCHS = 150

# Extra seeds for a variant-C sweep. Empty by default: one run per setting, at
# the seed hardcoded in train_pila.py / train_hvae.py (123). Adding seeds here
# (e.g. [240, 318]) emits `*_ghs_s<seed>.json` configs, which the train scripts
# pick up via the optional `seed` config key; compare_greenhyperspectra.py then
# reports mean +- std with `--seeds`.
SEEDS = []


def data_loader(loader_type):
    return {
        'type': loader_type,
        'args': {
            'data_dir': f'{DATA}/train.npz',
            'batch_size': 256,
            'shuffle': True,
            'validation_split': 0.0,
            'num_workers': 2,
            'with_const': False,
            # Augmentation stays off so loading matches Wytham/Austria; it is a
            # documented follow-up, not part of this experiment.
            'augment': False,
        },
        'data_dir_valid': f'{DATA}/valid.npz',
        'data_dir_test': f'{DATA}/test.npz',
    }


def common_arch():
    return {
        'physics': 'PROSAIL',
        'input_dim': INPUT_DIM,
        'hidden_dim': DIM_Z_PHY,
        'rtm_paras': f'{DATA}/ghs_prosail_paras.json',
        'standardization': {
            'x_mean': f'{DATA}/train_x_mean.npy',
            'x_scale': f'{DATA}/train_x_scale.npy',
        },
    }


def optimizer_block():
    return {
        'optimizer': {
            'type': 'Adam',
            'args': {'lr': 0.0003, 'weight_decay': 0.0001, 'amsgrad': True},
        },
        'loss': 'mse_loss',
        'metrics': [],
        'lr_scheduler': {'type': 'CosineAnnealingLR', 'args': {'T_max': EPOCHS}},
    }


def pila_config(variant):
    no_phy = variant == 'A'
    dim_z_aux = {'A': DIM_Z_PHY + DIM_Z_AUX, 'B': 0, 'C': DIM_Z_AUX}[variant]
    phys_vae = {
        'no_phy': no_phy,
        'dim_z_aux': dim_z_aux,
        'dim_z_phy': 0 if no_phy else DIM_Z_PHY,
        'activation': 'elu',
        'num_units_feat': 128,
        'hidlayers_feat': [512, 256],
        'hidlayers_z_aux': [128],
        'hidlayers_z_phy': [128],
        'hidlayers_aux_dec': [128, 256, 512],
        'residual_rank': RESIDUAL_RANK,
        'tau_init': 3.0,
        'tau_final': 1.0,
        'tau_warmup_epochs': 20,
        'r_init': 0.0,
        'r_final': 1.0,
        'r_warmup_epochs': 30,
        'ortho_penalty_weight': ORTHO_W,
        'coeff_penalty_weight': 0.0,
        'delta_penalty_weight': 0.0,
        'detach_x_P_for_bias': True,
    }
    cfg = {
        'name': f'PILA_PROSAIL_{variant}_GHS',
        'n_gpu': 0,
        'arch': {'type': 'PHYS_VAE_SMPL', 'args': common_arch(), 'phys_vae': phys_vae},
        'data_loader': data_loader('SpectrumGHSDataLoader'),
    }
    cfg.update(optimizer_block())
    cfg['trainer'] = {
        'epochs': EPOCHS,
        'save_dir': 'saved/greenhyperspectra',
        'save_period': 10,
        'log_step': 20,
        'verbosity': 2,
        'monitor': 'min val_rec_loss',
        'early_stop': 100,
        'input_key': 'spectrum',
        'output_key': 'spectrum',
        'stablize_grad': True,
        'grad_clip_norm': 1.0,
        'rec_loss_scale': REC_LOSS_SCALE,
        'tensorboard': False,
        'wandb': True,
        'phys_vae': {
            'epochs_pretrain': 0,
            'kl_warmup_epochs': 50,
            'balance_gate': 0.001,
            'balance_data_aug': 1.0,
            'balance_residual': 0.001,
            'use_kl_term_z_phy': False,
            'beta_max_z_phy': 1.0,
            'use_kl_term_z_aux': False,
            'beta_max_z_aux': 1.0,
            'use_capacity_control': False,
            'C_max': float(DIM_Z_PHY),
            'C_gamma': 10.0,
            'beta_aux': 1.0,
            'edge_penalty_weight': 0.0,
            'edge_penalty_power': 1.0,
            'use_ema_prior': False,
            'ema_momentum': 0.999,
        },
    }
    return cfg


def hvae_config(variant):
    no_phy = variant == 'A'
    dim_z_aux2 = {'A': DIM_Z_PHY + DIM_Z_AUX, 'B': -1, 'C': DIM_Z_AUX}[variant]
    phys_vae = {
        'no_phy': no_phy,
        'dim_z_aux2': dim_z_aux2,
        'dim_z_phy': DIM_Z_PHY,
        'activation': 'elu',
        'hidlayers_feat': [512, 256],
        'num_units_feat': 128,
        'hidlayers_aux2_enc': [128],
        'hidlayers_z_phy': [128],
        'hidlayers_aux2_expand': [128, 256, 512],
        'hidlayers_aux2_dec': [128, 256, 512],
        # 4 * in_channels would be 6884 units (~35M params) purely because the
        # input got wider; capped so the correction MLP stays trainable on CPU.
        'aux2_map_hidden': 2048,
    }
    cfg = {
        'name': f'HVAE_PROSAIL_{variant}_GHS',
        'n_gpu': 0,
        'arch': {'type': 'PHYS_VAE', 'args': common_arch(), 'phys_vae': phys_vae},
        'data_loader': data_loader('SpectrumGHSDataLoader'),
    }
    cfg.update(optimizer_block())
    cfg['trainer'] = {
        'epochs': EPOCHS,
        'save_dir': 'saved/greenhyperspectra',
        'save_period': 10,
        'log_step': 20,
        'verbosity': 2,
        'monitor': 'min val_rec_loss',
        'early_stop': 100,
        'input_key': 'spectrum',
        'output_key': 'spectrum',
        'stablize_grad': True,
        'grad_clip_norm': 1.0,
        'rec_loss_scale': REC_LOSS_SCALE,
        'tensorboard': False,
        'wandb': True,
        'phys_vae': {
            'epochs_pretrain': 0,
            'x_lnvar': -9.0,
            'balance_kld': 1.0,
            'balance_lact_enc': 0.1,
            'balance_unmix': 0.1,
            'balance_data_aug': 0.1,
            'balance_lact_dec': 0.1,
        },
    }
    return cfg


def main():
    os.makedirs(OUT, exist_ok=True)

    # Variant C is the headline comparison (PILA vs HVAE, both with the residual
    # correction) and the only one the rebuttal claim rests on. A and B are
    # written for reference but are not run by default.
    for variant in ('A', 'B', 'C'):
        for fam, cfg in (('PILA', pila_config(variant)),
                         ('HVAE', hvae_config(variant))):
            path = os.path.join(OUT, f'{fam}_PROSAIL_{variant}_ghs.json')
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=4)
            print(f'wrote {path}')

    # Short debug variants of the headline runs: 2 epochs, wandb off, separate
    # experiment name so they never land in the same saved/ directory as a real run.
    for fam, builder in (('PILA', pila_config), ('HVAE', hvae_config)):
        cfg = builder('C')
        cfg['name'] = f'DEBUG_{fam}_PROSAIL_C_GHS'
        cfg['trainer']['epochs'] = 2
        cfg['trainer']['save_period'] = 1
        cfg['trainer']['log_step'] = 5
        cfg['trainer']['wandb'] = False
        cfg['lr_scheduler']['args']['T_max'] = 2
        path = os.path.join(OUT, f'_debug_{fam}_PROSAIL_C_ghs.json')
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=4)
        print(f'wrote {path}')

    for seed in SEEDS:
        for fam, builder in (('PILA', pila_config), ('HVAE', hvae_config)):
            cfg = builder('C')
            cfg['seed'] = seed
            cfg['name'] = f'{fam}_PROSAIL_C_GHS_s{seed}'
            path = os.path.join(OUT, f'{fam}_PROSAIL_C_ghs_s{seed}.json')
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=4)
            print(f'wrote {path}')

    print(f'\nrec_loss_scale = {REFERENCE_BANDS}/{INPUT_DIM} = {REC_LOSS_SCALE:.6g}; all loss weights unchanged')
    print('headline runs: PILA_PROSAIL_C_ghs.json vs HVAE_PROSAIL_C_ghs.json'
          + (f'  (+ seed sweep {SEEDS})' if SEEDS else '  (single seed)'))


if __name__ == '__main__':
    main()
