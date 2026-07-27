# Training script for HVAE (baseline)
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model import PHYS_VAE  # HVAE model
from parse_config import ConfigParser
from trainer import PhysVAETrainer  # HVAE trainer
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
        validation_split=0.0,  # Validation set is already separated
        num_workers=2,
        with_const=config['data_loader']['args']['with_const'] if 'with_const' in config['data_loader']['args'] else False
    )

    # Build model architecture and log 
    # model = config.init_obj('arch', module_arch)
    model = PHYS_VAE(config)
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
    trainer = PhysVAETrainer(
        model, criterion, metrics, optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler
    )

    trainer.train()


if __name__ == '__main__':
    # TODO add custom arguments
    args = argparse.ArgumentParser(description='HVAE training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # Custom options to modify configuration from default values in JSON file
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    # Optional per-run seed (used by the GreenHyperSpectra seed sweep). Absent
    # or equal to SEED leaves the RNG state above untouched, so previously
    # trained configs reproduce exactly.
    run_seed = config.config.get('seed', SEED)
    if run_seed != SEED:
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        print(f"Re-seeded with seed={run_seed}")

    # Initialize WandB
    wandb.init(
        project="HVAE",
        name=f"{config['name']}",
        config=config,
        mode="online" if config['trainer']['wandb'] else "disabled",
    )

    main(config)

    wandb.finish()
