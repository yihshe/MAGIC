# Training script for PILA (our method)
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model import PHYS_VAE_SMPL  # PILA model
from parse_config import ConfigParser
from trainer import PhysVAETrainerSMPL  # PILA trainer
from utils import prepare_device
import wandb

# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # Setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    
    valid_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['data_dir_valid'],
        batch_size=64,
        shuffle=True,
        validation_split=0.0,
        num_workers=2,
        with_const=config['data_loader']['args']['with_const'] if 'with_const' in config['data_loader']['args'] else False
    )

    # Build model architecture and log 
    model = PHYS_VAE_SMPL(config)
    logger.info(model)

    # Prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Get function handles for loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # Build optimizer and learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # CHANGED: make scheduler optional
    lr_scheduler = None
    if 'lr_scheduler' in config.config:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Initialize Phys-VAE Trainer
    trainer = PhysVAETrainerSMPL(
        model, criterion, metrics, optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler
    )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PILA training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size'),
        # PhysVAE specific arguments TODO a more elegant way to do this, currently it is hardcoded in parse_config.py
        CustomArgs(['--use_kl_term_z_phy'], type=str,
                   target='trainer;phys_vae;use_kl_term_z_phy'),
        CustomArgs(['--beta_max_z_phy'], type=float,
                   target='trainer;phys_vae;beta_max_z_phy'),
        CustomArgs(['--use_kl_term_z_aux'], type=str,
                   target='trainer;phys_vae;use_kl_term_z_aux'),
        CustomArgs(['--beta_max_z_aux'], type=float,
                   target='trainer;phys_vae;beta_max_z_aux'),
        CustomArgs(['--kl_warmup_epochs'], type=int,
                   target='trainer;phys_vae;kl_warmup_epochs'),
        CustomArgs(['--epochs_pretrain'], type=int,
                   target='trainer;phys_vae;epochs_pretrain'),
        CustomArgs(['--tau_warmup_epochs'], type=int,
                   target='arch;phys_vae;tau_warmup_epochs'),
        CustomArgs(['--tau_init'], type=float,
                   target='arch;phys_vae;tau_init'),
        CustomArgs(['--r_warmup_epochs'], type=int,
                   target='arch;phys_vae;r_warmup_epochs'),
        CustomArgs(['--r_init'], type=float,
                   target='arch;phys_vae;r_init'),
        CustomArgs(['--dim_z_aux'], type=int,
                   target='arch;phys_vae;dim_z_aux'),
        CustomArgs(['--residual_rank'], type=int,
                   target='arch;phys_vae;residual_rank'),
        CustomArgs(['--detach_x_P_for_bias'], type=str,
                   target='arch;phys_vae;detach_x_P_for_bias'),
        # NEW: Capacity control arguments
        CustomArgs(['--use_capacity_control'], type=str,
                   target='trainer;phys_vae;use_capacity_control'),
        CustomArgs(['--C_max'], type=float,
                   target='trainer;phys_vae;C_max'),
        CustomArgs(['--C_gamma'], type=float,
                   target='trainer;phys_vae;C_gamma'),
        CustomArgs(['--beta_aux'], type=float,
                   target='trainer;phys_vae;beta_aux'),
        CustomArgs(['--coeff_penalty_weight'], type=float,
            target='trainer;phys_vae;coeff_penalty_weight'),
        CustomArgs(['--delta_penalty_weight'], type=float,
            target='trainer;phys_vae;delta_penalty_weight'),
        # NEW: Edge penalty arguments
        CustomArgs(['--edge_penalty_weight'], type=float,
                   target='trainer;phys_vae;edge_penalty_weight'),
        CustomArgs(['--edge_penalty_power'], type=float,
                   target='trainer;phys_vae;edge_penalty_power'),
        # NEW: EMA prior arguments
        CustomArgs(['--use_ema_prior'], type=str,
                   target='trainer;phys_vae;use_ema_prior'),
        CustomArgs(['--ema_momentum'], type=float,
                   target='trainer;phys_vae;ema_momentum'),
        # NEW: data_loader type
        CustomArgs(['--data_loader_type'], type=str,
                   target='data_loader;type'),
        # NEW: Time feature arguments
        CustomArgs(['--time_feat_dim'], type=int,
                   target='arch;args;time_feat_dim'),
        CustomArgs(['--use_time_in_residual'], type=str,
                   target='arch;args;use_time_in_residual'),
        CustomArgs(['--temporal_smoothness_weight'], type=float,
                   target='trainer;phys_vae;temporal_smoothness_weight'),
        CustomArgs(['--loss'], type=str,
                   target='loss'),
    ]
    config = ConfigParser.from_args(args, options)
    
    # Debug: Print updated config values
    print(f"Updated config values:")
    print(f"  use_kl_term_z_phy: {config.config['trainer']['phys_vae'].get('use_kl_term_z_phy', 'NOT_SET')}")
    print(f"  beta_max_z_phy: {config.config['trainer']['phys_vae'].get('beta_max_z_phy', 'NOT_SET')}")
    print(f"  use_kl_term_z_aux: {config.config['trainer']['phys_vae'].get('use_kl_term_z_aux', 'NOT_SET')}")
    print(f"  beta_max_z_aux: {config.config['trainer']['phys_vae'].get('beta_max_z_aux', 'NOT_SET')}")
    print(f"  kl_warmup_epochs: {config.config['trainer']['phys_vae'].get('kl_warmup_epochs', 'NOT_SET')}")
    print(f"  epochs_pretrain: {config.config['trainer']['phys_vae'].get('epochs_pretrain', 'NOT_SET')}")
    print(f"  use_ema_prior: {config.config['trainer']['phys_vae'].get('use_ema_prior', 'NOT_SET')}")
    print(f"  ema_momentum: {config.config['trainer']['phys_vae'].get('ema_momentum', 'NOT_SET')}")
    print(f"  edge_penalty_weight: {config.config['trainer']['phys_vae'].get('edge_penalty_weight', 'NOT_SET')}")
    print(f"  edge_penalty_power: {config.config['trainer']['phys_vae'].get('edge_penalty_power', 'NOT_SET')}")
    print(f"  detach_x_P_for_bias: {config.config['arch']['phys_vae'].get('detach_x_P_for_bias', 'NOT_SET')}")

    # Optional per-run seed (used by the GreenHyperSpectra seed sweep). Absent
    # or equal to SEED leaves the RNG state above untouched, so previously
    # trained configs reproduce exactly.
    run_seed = config.config.get('seed', SEED)
    if run_seed != SEED:
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        print(f"Re-seeded with seed={run_seed}")

    wandb.init(
        project="PILA",
        name=f"{config['name']}",
        config=config,
        mode="online" if config['trainer']['wandb'] else "disabled",
    )

    main(config)

    wandb.finish()

# NOTE: consider CosineAnnealingLR or ReduceLROnPlateau; see config.
