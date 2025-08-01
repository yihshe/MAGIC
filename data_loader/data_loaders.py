from torchvision import datasets, transforms
from base import BaseDataLoader
from datasets.spectrumS2 import SpectrumS2, SyntheticS2
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
    GPS data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = DisplacementGPSSeq(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class EVIDataLoader(BaseDataLoader):
    """
    GPS data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = TimeSeriesEVI(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)