from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

from torch.utils.data import Dataset

from avalanche.benchmarks.utils.data import AvalancheDataset

import numpy as np
import pandas as pd

from src.transforms import BaseTransform, apply_transforms
from src.utils.general import read_dataframe

TDatasetSubset = Literal["training", "testing", "all"]
TDataset = TypeVar("TDataset", bound="BaseDataset")
# TProcessor = TypeVar("TProcessor", bound="BaseDatasetPreprocessor")
TAvalancheDataset = TypeVar(
    "TAvalancheDataset", bound=AvalancheDataset
)


def assert_dataset_subset(subset: TDatasetSubset):
    assert subset in [
        "training",
        "testing",
        "all",
    ], "subset must be one of 'training', 'testing', 'all'"


class _BaseDataset(Dataset, metaclass=ABCMeta):
    @property
    @abstractmethod
    def input_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class BaseDataset(_BaseDataset):
    TRAIN_RATIO = 0.8

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        tasks: pd.Series | None = None,
    ):
        self.features = features
        self.targets = targets.values
        self.tasks = tasks.values if tasks is not None else None

    @property
    def input_size(self) -> int:
        if self.features is None:
            raise ValueError("Dataset not loaded yet")
        if len(self.features) == 0:
            raise ValueError("Dataset is empty")
        return self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: int):
        if self.tasks is None:
            return (
                self.features.iloc[index].astype(np.float32).values,
                self.targets[index],
            )

        return (
            self.features.iloc[index].astype(np.float32).values,
            self.targets[index],
            self.tasks[index],
        )


# class BaseDatasetWithDrift(BaseDataset):
#     def __init__(
#         self,
#         features: pd.DataFrame,
#         targets: pd.Series,
#         drifts: pd.Series,
#     ):
#         super().__init__(features, targets)
#         self.drifts = drifts.values

#     def __getitem__(self, index: int):
#         return (
#             self.features.iloc[index].astype(np.float32).values,
#             self.targets[index],
#             self.drifts[index],
#         )


@dataclass
class BaseDatasetAccessor(Generic[TDataset]):
    train: TDataset
    test: TDataset
    # train_dataset: TAvalancheDataset
    # test_dataset: TAvalancheDataset


TGeneratorReturn = TypeVar("TGeneratorReturn")


class BaseDatasetGenerator(
    Generic[TGeneratorReturn], metaclass=ABCMeta
):
    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str,
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: BaseTransform | list[BaseTransform] | None = None,
    ):
        self.data: pd.DataFrame = read_dataframe(file)
        self.train_ratio = train_ratio
        self._target = target
        self._transform = transform

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return apply_transforms(df, self._transform)

    @property
    def target(self):
        return self._target

    @abstractmethod
    def __call__(self, shuffle: bool = False) -> TGeneratorReturn:
        pass

    def generate(self, shuffle: bool = False) -> TGeneratorReturn:
        return self.__call__(shuffle=shuffle)


# class BaseDatasetGenerator(Generic[TAccessor], metaclass=ABCMeta):
#     @abstractmethod
#     def __call__(self) -> TAccessor:
#         raise NotImplementedError

#     def generate(self) -> TAccessor:
#         return self.__call__()


__all__ = [
    "_BaseDataset",
    "BaseDataset",
    # "BaseDatasetWithDrift",
    "TDatasetSubset",
    "TDataset",
    "assert_dataset_subset",
    "BaseDatasetAccessor",
    "BaseDatasetGenerator",
]
