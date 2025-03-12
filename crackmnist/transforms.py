import numpy as np
import torch


class InputNormalization:
    def __init__(self, means=None, stds=None):
        self.means = np.asarray(means).reshape((-1, 1, 1)) if means is not None else None
        self.stds = np.asarray(stds).reshape((-1, 1, 1)) if stds is not None else None

    def __call__(self, sample):
        # Normalize input
        img = np.asarray(sample)
        means = img.mean(axis=(1, 2), keepdims=True) if self.means is None else self.means
        stds = img.std(axis=(1, 2), keepdims=True) if self.stds is None else self.stds
        img = (img - means) / stds
        sample = torch.tensor(img, dtype=torch.float32)

        return sample

class NumpyToTensor:
    def __call__(self, sample):
        # Assuming sample is a NumPy array or image with shape (2, 28, 28)
        return torch.tensor(sample)