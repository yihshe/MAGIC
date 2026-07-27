from torchvision import datasets, transforms
from base import BaseDataLoader
from datasets.spectrumS2 import SpectrumS2, SyntheticS2
from datasets.spectrumGHS import SpectrumGHS
from datasets.displacementGPS import DisplacementGPS, DisplacementGPSSeq
from datasets.timeseriesEVI import TimeSeriesEVI
import numpy as np
import torch

class SpectrumS2DataLoader(BaseDataLoader):
    """
    SpectrumS2 data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = SpectrumS2(self.data_dir, with_const=with_const)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class SyntheticS2DataLoader(BaseDataLoader):
    """
    SyntheticS2 data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = SyntheticS2(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class SpectrumGHSDataLoader(BaseDataLoader):
    """
    GreenHyperSpectra data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False,
                 augment=False, aug_prob=0.5, noise_std=0.01):
        self.data_dir = data_dir
        self.dataset = SpectrumGHS(self.data_dir, with_const=with_const, augment=augment,
                                   aug_prob=aug_prob, noise_std=noise_std)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class GPSDataLoader(BaseDataLoader):
    """
    GPS data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = DisplacementGPS(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class GPSSeqDataLoader(BaseDataLoader):
    """
    GPS data loading demo using BaseDataLoader with temporal smoothness support
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False, 
                 seq_len=5, mode='train'):
        self.data_dir = data_dir
        self.dataset = DisplacementGPSSeq(self.data_dir, seq_len=seq_len, mode=mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class EVIDataLoader(BaseDataLoader):
    """
    GPS data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = TimeSeriesEVI(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)