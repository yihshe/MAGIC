import os

import numpy as np
import torch
import torch.utils.data as data

from datasets import PARENT_DIR


class SpectrumGHS(data.Dataset):
    """
    GreenHyperSpectra hyperspectral reflectance, prepared by
    `rebuttal/ghs/preprocess_ghs.py`.

    The .npz holds spectra already standardized with the training-set per-band
    StandardScaler, so `__getitem__` mirrors SpectrumS2: it just hands over the
    spectrum. Traits are carried alongside for evaluation only -- PILA and HVAE
    never see them during training (they are NaN for every unlabeled row).
    """

    def __init__(self, npz_path, transform=None, with_const=False,
                 augment=False, aug_prob=0.5, noise_std=0.01):
        super(SpectrumGHS, self).__init__()
        arr = np.load(os.path.join(PARENT_DIR, npz_path), allow_pickle=False)
        self.spectra = arr['X'].astype(np.float32)
        self.traits = arr['Y'].astype(np.float32)
        self.source = arr['source'].astype(np.int64)
        self.dataset_id = arr['dataset'].astype(np.int64)
        self.bands = [str(b) for b in arr['bands']]
        self.trait_names = [str(t) for t in arr['traits']]
        self.transform = transform
        # `with_const` exists for signature parity with the S2 loaders; GHS has
        # no per-sample acquisition geometry, so PROSAIL runs at fixed angles.
        self.with_const = with_const

        # Augmentation is off by default so that loading matches the Wytham and
        # Austria pipelines. Kept here as a hook for the follow-up experiment.
        self.augment = augment
        self.aug_prob = aug_prob
        self.noise_std = noise_std

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, index):
        spectrum = torch.from_numpy(self.spectra[index])

        if self.augment and torch.rand(1).item() < self.aug_prob:
            spectrum = spectrum + torch.randn_like(spectrum) * self.noise_std

        if self.transform is not None:
            spectrum = self.transform(spectrum)

        return {
            'spectrum': spectrum,
            'traits': torch.from_numpy(self.traits[index]),
            'source': int(self.source[index]),
            'dataset_id': int(self.dataset_id[index]),
            'sample_id': int(index),
        }
