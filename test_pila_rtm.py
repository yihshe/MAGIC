import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model import PHYS_VAE_SMPL  # PILA model
from parse_config import ConfigParser
import pandas as pd
import numpy as np
# from physics.rtm.rtm import RTM

import os
import logging
from datetime import datetime
from pathlib import Path
from utils import read_json
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_test_logging(checkpoint_path):
    """
    Setup logging to save test logs in the checkpoint's directory
    """
    checkpoint_path = Path(checkpoint_path)
    # Get the log directory from the checkpoint path
    # checkpoint is in models/ subdirectory, so go up one level to get the experiment directory
    experiment_dir = checkpoint_path.parent.parent
    log_dir = experiment_dir / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file name
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    log_filename = f'info_test_{timestamp}.log'
    log_file_path = log_dir / log_filename
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger('test')
    logger.info(f"Test logging setup. Log file: {log_file_path}")
    return logger

def main(config, args: argparse.Namespace):
    # Setup custom logging for test
    logger = setup_test_logging(args.resume)

    if args.full_region:
        # Use full region mapping data
        data_dir_test = os.path.join(
            CURRENT_DIR,
            'data/processed/rtm/wytham/full_region_mapping/full_region_for_mapping.csv'
        )
        logger.info(f"Using full region mapping data: {data_dir_test}")
    elif args.insitu:
        data_dir_test = config.config['data_loader']['data_dir_test'].replace('test.csv', 'test_frm4veg.csv')
    else:
        data_dir_test = config.config['data_loader']['data_dir_test']
    # setup data_loader instances
    data_loader = getattr(module_data, config.config['data_loader']['type'])(
        data_dir = data_dir_test,
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        num_workers=0,
        with_const=config.config['data_loader']['args']['with_const'] if 'with_const' in config.config['data_loader']['args'] else False
    )

    # build model architecture
    # model = config.init_obj('arch', module_arch)
    # model = model_init(config)
    base_model = PHYS_VAE_SMPL(config.config)
    model: torch.nn.Module = base_model
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config.config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config.config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(os.path.join(CURRENT_DIR, config.resume))
    state_dict = checkpoint['state_dict']
    if config.config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    
    # Get config values first
    no_phy = config.config['arch']['phys_vae']['no_phy']
    dim_z_aux = config.config['arch']['phys_vae']['dim_z_aux']
    
    # Load tau/r values for inference if available
    tau_r_values = checkpoint.get('tau_r_values', None)
    if tau_r_values is not None and not no_phy:
        base_model.dec.set_tau_r_from_checkpoint(tau_r_values)
        logger.info(f"Loaded tau/r values from checkpoint: tau={tau_r_values['tau']:.3f}, r={tau_r_values['r']:.3f}")
    else:
        logger.info("Using default tau/r values for inference")

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    data_key = config.config['trainer']['input_key']
    target_key = config.config['trainer']['output_key']
    if 'input_const_keys' in config.config['trainer']:
        input_const_keys = config.config['trainer']['input_const_keys']
    else:
        input_const_keys = None
    no_phy = config.config['arch']['phys_vae']['no_phy']
    dim_z_aux = config.config['arch']['phys_vae']['dim_z_aux']

    if not no_phy:
        ATTRS = ['N', 'cab', 'cw', 'cm', 'LAI', 'LAIu', 'fc', 'cd', 'h']
    else:
        ATTRS = [str(i+1) for i in range(dim_z_aux)]

    # analyze the reconstruction loss per band
    S2_BANDS = ['B02_BLUE', 'B03_GREEN', 'B04_RED', 'B05_RE1', 'B06_RE2',
                'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B11_SWI1',
                'B12_SWI2']

    analyzer = {}


    with torch.no_grad():
        for batch_idx, data_dict in enumerate(data_loader):
            data = data_dict[data_key].to(device)
            target = data_dict[target_key].to(device)
            if input_const_keys is not None:
                input_const = {k: data_dict[k].to(device) for k in input_const_keys}
            else:
                input_const = None
            # forward pass 
            # Determine hard_z based on whether KL term was used during training
            # Check if model was trained with KL terms by looking at config
            # Backward compatibility: handle both old and new config formats
            if 'use_kl_term' in config.config['trainer']['phys_vae']:
                # Old config format: use_kl_term applies to both z_phy and z_aux
                use_kl_term_old = config.config['trainer']['phys_vae']['use_kl_term']
                use_kl_term_z_phy = use_kl_term_old
                use_kl_term_z_aux = use_kl_term_old
            else:
                # New config format: separate controls for z_phy and z_aux
                use_kl_term_z_phy = config.config['trainer']['phys_vae'].get('use_kl_term_z_phy', False)
                use_kl_term_z_aux = config.config['trainer']['phys_vae'].get('use_kl_term_z_aux', False)
            
            hard_z_phy = not use_kl_term_z_phy  # Use deterministic sampling when KL term was disabled
            hard_z_aux = not use_kl_term_z_aux  # Use deterministic sampling when KL term was disabled
            
            # Model returns: z_phy, z_aux, x_PB (physics+bias), x_P (raw physics)
            latent_phy, latent_aux, x_PB, x_P = model(data, inference=True, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux, const=input_const)

            if not no_phy:
                latent_phy = base_model.physics_model.rescale(latent_phy)
                latent = torch.stack([latent_phy[k] for k in latent_phy.keys()], dim=1)
                bias = None
                if dim_z_aux > 0:  # Only include residual components when auxiliary latent variables exist
                    bias = x_PB - x_P  # bias = corrected output - raw physics output
                    data_concat(analyzer, 'init_output', x_P)  # raw physics output
                    data_concat(analyzer, 'bias', bias)
                    data_concat(analyzer, 'latent_aux', latent_aux)
                else:
                    # When dim_z_aux = 0, no residual correction, so x_PB = x_P
                    data_concat(analyzer, 'init_output', x_P)  # raw physics output
                    data_concat(analyzer, 'bias', torch.zeros_like(x_P))  # zero bias
                    data_concat(analyzer, 'latent_aux', torch.zeros(x_P.shape[0], 0, device=x_P.device))  # empty latent_aux
            else:
                latent = latent_aux

            data_concat(analyzer, 'output', x_PB)  # corrected output (physics + bias)
            data_concat(analyzer, 'target', target)
            data_concat(analyzer, 'latent', latent)
            data_concat(analyzer, 'sample_id', data_dict['sample_id'])
            data_concat(analyzer, 'class', data_dict['class'])
            data_concat(analyzer, 'date', data_dict['date'])
            # Save spatial coordinates if available (for full region mapping)
            if 'x' in data_dict:
                data_concat(analyzer, 'x', data_dict['x'].cpu().numpy() if torch.is_tensor(data_dict['x']) else data_dict['x'])
            if 'y' in data_dict:
                data_concat(analyzer, 'y', data_dict['y'].cpu().numpy() if torch.is_tensor(data_dict['y']) else data_dict['y'])
            # NOTE this MSE loss given by PyTorch is element-wise, but for Phys-VAE, it is sample-wise torch.sum((output-target)**2, dim=1).mean()
            # NOTE the way how the loss is computed in MAGIC also need to be double-checked
            # computing loss, metrics on test set
            loss = loss_fn(x_PB, target)  # Use corrected output for loss calculation 
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(x_PB, target) * batch_size  # Use corrected output for metrics

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples
        for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    # save the analyzer to csv using pandas
    columns = []

    columns += ['output'+'_'+b for b in S2_BANDS]
    columns += ['target'+'_'+b for b in S2_BANDS]
    columns += ['latent'+'_'+b for b in ATTRS]

    data = torch.hstack((
        analyzer['output'],
        analyzer['target'],
        analyzer['latent']
    ))

    if not no_phy:
        # Always include init_output and bias columns for consistency
        columns += ['init_output_'+b for b in S2_BANDS]
        columns += ['bias_'+b for b in S2_BANDS]
        data = torch.hstack((
            data,
            analyzer['init_output'],
            analyzer['bias']
        ))
        
        # Only include latent_aux columns when dim_z_aux > 0
        if dim_z_aux > 0:
            columns += ['latent_aux_'+str(b+1) for b in range(dim_z_aux)]
            data = torch.hstack((
                data,
                analyzer['latent_aux']
            ))

    # Create a pandas dataframe
    data = data.cpu().numpy()
    df = pd.DataFrame(columns=columns, data=data)
    # Add sample_id, class, and date to the dataframe
    df['sample_id'] = analyzer['sample_id']
    df['class'] = analyzer['class']
    df['date'] = analyzer['date']
    # Add spatial coordinates if available (for full region mapping)
    if 'x' in analyzer:
        df['x'] = analyzer['x']
    if 'y' in analyzer:
        df['y'] = analyzer['y']

    if args.full_region:
        suffix = '_full_region'
    elif args.insitu:
        suffix = '_frm4veg'
    else:
        suffix = ''

    df.to_csv(
        os.path.join(CURRENT_DIR, str(config.resume).split('.pth')[0]+f'_testset_analyzer{suffix}.csv'),
              index=False)
    logger.info('Analyzer saved to {}'.format(
        os.path.join(CURRENT_DIR, str(config.resume).split('.pth')[0]+f'_testset_analyzer{suffix}.csv')
    ))


def data_concat(analyzer: dict, key: str, data):
    if key not in analyzer:
        analyzer[key] = data
    elif type(data) == torch.Tensor:
        analyzer[key] = torch.cat((analyzer[key], data), dim=0)
    elif type(data) == list:
        analyzer[key] = analyzer[key] + data
    elif type(data) == np.ndarray:
        analyzer[key] = np.concatenate((analyzer[key], data), axis=0)


class TestConfigParser:
    """
    Custom config parser for testing that doesn't create new experiment directories
    """
    def __init__(self, config_dict, resume_path):
        self._config = config_dict
        self.resume = resume_path
        # Don't create new directories for testing
    
    @classmethod
    def from_args(cls, args):
        """Initialize from command line arguments"""
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Resume path need to be specified for testing."
            assert args.resume is not None, msg_no_cfg
        
        config = read_json(cfg_fname)
        return cls(config, args.resume)
    
    @property
    def config(self):
        return self._config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-i', '--insitu', action='store_true', 
                        help='use insitu data (default: False)')
    parser.add_argument('-f', '--full_region', action='store_true',
                        help='use full region mapping data (default: False)')

    args = parser.parse_args()
    
    # Use custom config parser for testing
    config = TestConfigParser.from_args(args)
    main(config, args)
