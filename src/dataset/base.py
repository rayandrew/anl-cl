from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from torch.utils.data import Dataset

from avalanche.benchmarks.utils.data import AvalancheDataset

import numpy as np
import pandas as pd

from src.transforms import BaseFeatureEngineering

TDatasetSubset = Literal["training", "testing", "all"]
# TProcessor = TypeVar("TProcessor", bound="BaseDatasetPreprocessor")
TAvalancheDataset = TypeVar("TAvalancheDataset", bound=AvalancheDataset)


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


TDataset = TypeVar("TDataset", bound=BaseDataset)


@dataclass
class BaseDatasetAccessor(Generic[TDataset]):
    train: TDataset
    test: TDataset


TAccessor = TypeVar("TAccessor", bound=BaseDatasetAccessor)
TGeneratorReturn = TypeVar("TGeneratorReturn")


class BaseDatasetPrototype(Generic[TDataset, TAccessor], metaclass=ABCMeta):
    def create_accessor(self, train: Any, test: Any, **kwargs) -> TAccessor:
        raise NotImplementedError

    def create_dataset(self, X: Any, y: Any, **kwargs) -> TDataset:
        raise NotImplementedError


TDatasetPrototype = TypeVar("TDatasetPrototype", bound=BaseDatasetPrototype)

# TFeatureEngineering = TypeVar(
#     "TFeatureEngineering", bound=BaseFeatureEngineering
# )


class BaseDatasetGenerator(
    Generic[TDataset, TAccessor],
    metaclass=ABCMeta,
):
    def __init__(
        self,
        prototype: BaseDatasetPrototype[TDataset, TAccessor],
        target: str,
        feature_engineering: BaseFeatureEngineering,
        train_ratio: float,
    ):
        self.target = target
        self.feature_engineering = feature_engineering
        self._prototype = prototype
        self.train_ratio = train_ratio

    @property
    def prototype(self) -> BaseDatasetPrototype[TDataset, TAccessor]:
        return self._prototype

    @abstractmethod
    def __call__(self, shuffle: bool = False) -> Any:
        pass


__all__ = [
    "_BaseDataset",
    "BaseDataset",
    "TDatasetSubset",
    "TDataset",
    "TAccessor",
    "TGeneratorReturn",
    "TDatasetPrototype",
    "assert_dataset_subset",
    "BaseDatasetAccessor",
    "BaseDatasetGenerator",
    "BaseDatasetPrototype",
]
