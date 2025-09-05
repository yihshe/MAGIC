import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model import PHYS_VAE_SMPL  # TODO modify for Phys-VAE
from parse_config import ConfigParser
import pandas as pd
import numpy as np
# from physics.rtm.rtm import RTM

import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(config, args: argparse.Namespace):
    logger = config.get_logger('test')

    if args.insitu:
        data_dir_test = config['data_loader']['data_dir_test'].replace('test.csv', 'test_frm4veg.csv')
    else:
        data_dir_test = config['data_loader']['data_dir_test']
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        data_dir = data_dir_test,
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        num_workers=0,
        with_const=config['data_loader']['args']['with_const'] if 'with_const' in config['data_loader']['args'] else False
    )

    # build model architecture
    # model = config.init_obj('arch', module_arch)
    # model = model_init(config)
    model = PHYS_VAE_SMPL(config)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(os.path.join(CURRENT_DIR, config.resume))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    data_key = config['trainer']['input_key']
    target_key = config['trainer']['output_key']
    if 'input_const_keys' in config['trainer']:
        input_const_keys = config['trainer']['input_const_keys']
    else:
        input_const_keys = None
    no_phy = config['arch']['phys_vae']['no_phy']
    dim_z_aux = config['arch']['phys_vae']['dim_z_aux']

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
            # forward pass NOTE hard_z is True is KL term is not used
            latent_phy, latent_aux, output, init_output = model(data, inference=True, hard_z=False, const=input_const)

            if not no_phy:
                latent_phy = model.physics_model.rescale(latent_phy)
                latent = torch.stack([latent_phy[k] for k in latent_phy.keys()], dim=1)
                bias = None
                if dim_z_aux >= 0:#TODO in ablation, it is possible that dim_z_aux=-1 but the model still has the bias correction
                    bias = output - init_output
                    data_concat(analyzer, 'init_output', init_output)
                    data_concat(analyzer, 'bias', bias)
                    data_concat(analyzer, 'latent_aux', latent_aux)
            else:
                latent = latent_aux

            data_concat(analyzer, 'output', output)
            data_concat(analyzer, 'target', target)
            data_concat(analyzer, 'latent', latent)
            data_concat(analyzer, 'sample_id', data_dict['sample_id'])
            data_concat(analyzer, 'class', data_dict['class'])
            data_concat(analyzer, 'date', data_dict['date'])
            # NOTE this MSE loss given by PyTorch is element-wise, but for Phys-VAE, it is sample-wise torch.sum((output-target)**2, dim=1).mean()
            # NOTE the way how the loss is computed in MAGIC also need to be double-checked
            # computing loss, metrics on test set
            loss = loss_fn(output, target) 
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

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
        if dim_z_aux >=0:
            columns += ['init_output_'+b for b in S2_BANDS]
            columns += ['bias_'+b for b in S2_BANDS]
            columns += ['latent_aux_'+str(b+1) for b in range(dim_z_aux)]
            data = torch.hstack((
                data,
                analyzer['init_output'],
                analyzer['bias'],
                analyzer['latent_aux']
            ))

    # Create a pandas dataframe
    data = data.cpu().numpy()
    df = pd.DataFrame(columns=columns, data=data)
    # Add sample_id, class, and date to the dataframe
    df['sample_id'] = analyzer['sample_id']
    df['class'] = analyzer['class']
    df['date'] = analyzer['date']

    insitu = '_frm4veg' if args.insitu else ''

    df.to_csv(
        os.path.join(CURRENT_DIR, str(config.resume).split('.pth')[0]+f'_testset_analyzer{insitu}.csv'),
              index=False)
    logger.info('Analyzer saved to {}'.format(
        os.path.join(CURRENT_DIR, str(config.resume).split('.pth')[0]+f'_testset_analyzer{insitu}.csv')
    ))


def data_concat(analyzer: dict, key: str, data):
    if key not in analyzer:
        analyzer[key] = data
    elif type(data) == torch.Tensor:
        analyzer[key] = torch.cat((analyzer[key], data), dim=0)
    elif type(data) == list:
        analyzer[key] = analyzer[key] + data


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

    # args.add_argument('-a', '--analyze', default=False, type=bool,
    #                   help='analyze and saved the test results (default: False)')
    config = ConfigParser.from_args(parser)
    args = parser.parse_args()
    main(config, args)
