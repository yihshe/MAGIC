# Adapted from the training script of Phys-VAE
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model import PHYS_VAE_SMPL  # Updated model for Phys-VAE
from parse_config import ConfigParser
from trainer import PhysVAETrainerSMPL  # Updated trainer for Phys-VAE
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
    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)

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
    # TODO add custom arguments
    args = argparse.ArgumentParser(description='PyTorch Template for Phys-VAE')
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

    # Initialize WandB
    wandb.init(
        project="PhysVAE_SMPL_RTM",
        entity="yihshe",
        name=f"{config['name']}",
        config=config,
        mode="online" if config['trainer']['wandb'] else "disabled",
    )

    main(config)

    wandb.finish()
