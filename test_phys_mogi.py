import argparse
import json
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model import PHYS_VAE  # TODO modify for Phys-VAE
from parse_config import ConfigParser
import pandas as pd
import numpy as np
# from physics.rtm.rtm import RTM

import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type_test'])(
        config['data_loader']['data_dir_test'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        num_workers=0,
        with_const=config['data_loader']['args']['with_const'] if 'with_const' in config['data_loader']['args'] else False
    )

    # build model architecture
    # model = config.init_obj('arch', module_arch)
    # model = model_init(config)
    model = PHYS_VAE(config)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss_test'])
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
    no_phy = config['arch']['phys_vae']['no_phy']
    dim_z_aux2 = config['arch']['phys_vae']['dim_z_aux2']

    if not no_phy:
        ATTRS = ['xcen', 'ycen', 'd', 'dV']
    else:
        ATTRS = [str(i+1) for i in range(dim_z_aux2)]

    analyzer = {}
    station_info = json.load(open(os.path.join(CURRENT_DIR,
                                               'configs/station_info.json')))
    GPS = []
    for direction in ['ux', 'uy', 'uz']:
        for station in station_info.keys():
            GPS.append(f'{direction}_{station}')

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(data_loader):
            data = data_dict[data_key].to(device)
            target = data_dict[target_key].to(device)

            latent_phy, latent_aux, output, init_output = model(data, inference=True)

            if not no_phy:
                latent_phy = model.physics_model.rescale(latent_phy)
                latent = torch.stack([latent_phy[k] for k in latent_phy.keys()], dim=1)
                bias = None
                if dim_z_aux2 >= 0:
                    bias = output - init_output
                    data_concat(analyzer, 'init_output', init_output)
                    data_concat(analyzer, 'bias', bias)
                    data_concat(analyzer, 'latent_aux', latent_aux)
            else:
                latent = latent_aux

            data_concat(analyzer, 'output', output)
            data_concat(analyzer, 'target', target)
            data_concat(analyzer, 'latent', latent)
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

    columns += ['output'+'_'+b for b in GPS]
    columns += ['target'+'_'+b for b in GPS]
    columns += ['latent'+'_'+b for b in ATTRS]

    data = torch.hstack((
        analyzer['output'],
        analyzer['target'],
        analyzer['latent']
    ))

    if not no_phy:
        if dim_z_aux2 >=0:
            columns += ['init_output_'+b for b in GPS]
            columns += ['bias_'+b for b in GPS]
            columns += ['latent_aux_'+str(b+1) for b in range(dim_z_aux2)]
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
    df['date'] = analyzer['date']
    df.to_csv(
        os.path.join(CURRENT_DIR, str(config.resume).split('.pth')[0]+'_testset_analyzer.csv'),
              index=False)
    logger.info('Analyzer saved to {}'.format(
        os.path.join(CURRENT_DIR, str(config.resume).split('.pth')[0]+'_testset_analyzer.csv')
    ))


def data_concat(analyzer: dict, key: str, data):
    if key not in analyzer:
        analyzer[key] = data
    elif type(data) == torch.Tensor:
        analyzer[key] = torch.cat((analyzer[key], data), dim=0)
    elif type(data) == list:
        analyzer[key] = analyzer[key] + data


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # args.add_argument('-a', '--analyze', default=False, type=bool,
    #                   help='analyze and saved the test results (default: False)')

    config = ConfigParser.from_args(args)
    main(config)
