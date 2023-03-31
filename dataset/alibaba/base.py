from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

import numpy as np


class AlibabaDataset(Dataset, metaclass=ABCMeta):
    TRAIN_RATIO = 0.8

    @abstractmethod
    def input_size(self) -> int:
        raise NotImplementedError

    def _process_nan(self, arr, fill=0):
        # change to previous row
        mask = np.isnan(arr[0])
        arr[0][mask] = fill
        for i in range(1, len(arr)):
            mask = np.isnan(arr[i])
            arr[i][mask] = arr[i - 1][mask]
        return arr

    @abstractmethod
    def n_experiences(self) -> int:
        raise NotImplementedError


__all__ = ["AlibabaDataset"]
