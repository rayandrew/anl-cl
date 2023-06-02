from abc import ABCMeta, abstractmethod
from typing import Literal

from torch.utils.data import Dataset


TDatasetSubset = Literal["training", "testing", "all"]


def assert_dataset_subset(subset: TDatasetSubset):
    assert subset in [
        "training",
        "testing",
        "all",
    ], "subset must be one of 'training', 'testing', 'all'"


class BaseDataset(Dataset, metaclass=ABCMeta):
    TRAIN_RATIO = 0.8

    @property
    @abstractmethod
    def input_size(self) -> int:
        raise NotImplementedError

    # @property
    # @abstractmethod
    # def n_experiences(self) -> int:
    #     raise NotImplementedError


__all__ = ["BaseDataset", "TDatasetSubset", "assert_dataset_subset"]


# non-sequence
# y_t = f(X_t)
