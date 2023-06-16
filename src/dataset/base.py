from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Callable,
    Generic,
    Literal,
    Sequence,
    Tuple,
    TypeVar,
)

from torch.utils.data import Dataset

from avalanche.benchmarks.utils.data import TAvalancheDataset

import pandas as pd

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


TDataset = TypeVar("TDataset", bound=BaseDataset)


@dataclass
class BaseDatasetAccessor(
    Generic[TDataset, TAvalancheDataset], metaclass=ABCMeta
):
    original_train_dataset: TDataset
    original_test_dataset: TDataset
    train_dataset: TAvalancheDataset
    test_dataset: TAvalancheDataset


# TTransformInput = TypeVar("TTransformInput")
# TTransformOutput = TypeVar("TTransformOutput")


class BaseTransform(metaclass=ABCMeta):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def transform(self) -> pd.DataFrame:
        return self.__call__(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return self.__repr__()


TTransform = Callable[[pd.DataFrame], pd.DataFrame] | BaseTransform


class BaseDatasetPreprocessor(Generic[TDataset], metaclass=ABCMeta):
    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: TTransform | None = None,
    ):
        self.data: pd.DataFrame = data
        self.train_ratio = train_ratio
        self.y_var = y
        self._transform = transform

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._transform is None:
            return df
        return self._transform(df)

    @property
    def output_column(self):
        return self.y_var

    @abstractmethod
    @cached_property
    def preprocessed_data(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self, shuffle: bool = False
    ) -> Tuple[TDataset, TDataset]:
        pass


TAccessor = TypeVar(
    "TAccessor",
    bound=BaseDatasetAccessor | Sequence[BaseDatasetAccessor],
)
TProcessor = TypeVar("TProcessor", bound=BaseDatasetPreprocessor)


class BaseDatasetGenerator(
    Generic[TAccessor, TProcessor], metaclass=ABCMeta
):
    accessor_cls: type[TAccessor]
    preprocessor_cls: type[TProcessor]

    @abstractmethod
    def __call__(self) -> TAccessor:
        raise NotImplementedError

    def generate(self) -> TAccessor:
        return self.__call__()


__all__ = [
    "BaseDataset",
    "TDatasetSubset",
    "TDataset",
    "TAccessor",
    "TTransform",
    "assert_dataset_subset",
    "BaseDatasetAccessor",
    "BaseDatasetPreprocessor",
    "BaseDatasetGenerator",
]
