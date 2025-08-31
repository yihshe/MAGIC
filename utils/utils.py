import json
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / \
            self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def kldiv_normal_normal(mean1:torch.Tensor, lnvar1:torch.Tensor, mean2:torch.Tensor, lnvar2:torch.Tensor):
    """
    KL divergence between normal distributions, KL( N(mean1, diag(exp(lnvar1))) || N(mean2, diag(exp(lnvar2))) )
    """
    # NEW: Clamp lnvar to prevent numerical instability (consistent with draw_normal)
    lnvar1_clamped = lnvar1.clamp(-9.0, 5.0)
    lnvar2_clamped = lnvar2.clamp(-9.0, 5.0)
    
    if lnvar1_clamped.ndim==2 and lnvar2_clamped.ndim==2:
        return 0.5 * torch.sum((lnvar1_clamped-lnvar2_clamped).exp() - 1.0 + lnvar2_clamped - lnvar1_clamped + (mean2-mean1).pow(2)/lnvar2_clamped.exp(), dim=1)
    elif lnvar1_clamped.ndim==1 and lnvar2_clamped.ndim==1:
        d = mean1.shape[1]
        return 0.5 * (d*((lnvar1_clamped-lnvar2_clamped).exp() - 1.0 + lnvar2_clamped - lnvar1_clamped) + torch.sum((mean2-mean1).pow(2), dim=1)/lnvar2_clamped.exp())
    else:
        raise ValueError()
    
def actmodule(activation:str):
    if activation == 'softplus':
        return nn.Softplus()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError('unknown activation function specified')

def draw_normal(mean:torch.Tensor, lnvar:torch.Tensor):
    # NEW: Clamp lnvar to prevent numerical instability
    lnvar_clamped = lnvar.clamp(-9.0, 5.0)
    std = torch.exp(0.5*lnvar_clamped) #TODO lnvar and reparameterization in VAE
    eps = torch.randn_like(std) # reparametrization trick
    return mean + eps*std