from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Sequence, TypeVar, cast

import pandas as pd

from src.dataset.base import (
    BaseDataset,
    BaseDatasetAccessor,
    BaseDatasetGenerator,
)
from src.transforms import BaseTransform
from src.utils.general import read_dataframe
from src.utils.general import split_dataset as split_dataset_fn

TAzureVMDataset = TypeVar("TAzureVMDataset", bound="AzureVMDataset")
TAccessor = TypeVar("TAccessor", bound=BaseDatasetAccessor)
TAccessorReturn = TypeVar(
    "TAccessorReturn",
    bound="AzureVMDataAccessor" | Sequence["AzureVMDataAccessor"],
)


class AzureVMDataset(BaseDataset):
    pass


@dataclass
class AzureVMDataAccessor(BaseDatasetAccessor[AzureVMDataset]):
    pass


class BaseAzureVMDatasetGenerator(
    Generic[TAccessorReturn, TAccessor, TAzureVMDataset],
    BaseDatasetGenerator[TAccessorReturn],
):
    dataset_cls: type[TAzureVMDataset]
    accessor_cls: type[TAccessor]

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str = "cpu_avg",
        n_split: int = 4,
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: BaseTransform | list[BaseTransform] | None = None,
    ):
        super(BaseAzureVMDatasetGenerator, self).__init__(
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self._file = file
        self.n_split = n_split

    @property
    def data(self) -> pd.DataFrame:
        # TODO: REMOVE THIS LIMIT
        return read_dataframe(self._file)[0:500_000]

    def __base_call__(self, data: pd.DataFrame, shuffle: bool) -> TAccessor:
        data = self.transform(data)

        X_train, X_test, y_train, y_test = split_dataset_fn(
            data.drop(columns=[self.target]),
            data[self.target],
            test_size=1 - self.train_ratio,
            shuffle=shuffle,
        )
        return self.accessor_cls(
            train=self.dataset_cls(X_train, y_train),  # type: ignore
            test=self.dataset_cls(X_test, y_test),  # type: ignore
        )

    def __call__(self, shuffle: bool = False) -> TAccessorReturn:
        return cast(TAccessorReturn, self.__base_call__(self.data, shuffle))


class AzureVMDatasetGenerator(
    BaseAzureVMDatasetGenerator[
        AzureVMDataAccessor,
        AzureVMDataAccessor,
        AzureVMDataset,
    ]
):
    dataset_cls = AzureVMDataset
    accessor_cls = AzureVMDataAccessor


class AzureVMDatasetChunkGenerator(
    BaseAzureVMDatasetGenerator[
        list[AzureVMDataAccessor],
        AzureVMDataAccessor,
        AzureVMDataset,
    ]
):
    dataset_cls = AzureVMDataset
    accessor_cls = AzureVMDataAccessor

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str = "cpu_avg",
        n_split: int = 4,
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: BaseTransform | list[BaseTransform] | None = None,
    ):
        super(AzureVMDatasetChunkGenerator, self).__init__(
            file=file,
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self.n_split = n_split

    def __call__(self, shuffle: bool = False) -> list[AzureVMDataAccessor]:
        # NOTE: for debugging
        data = self.data.iloc[0:100_000]
        size = len(data)
        split_size = size // self.n_split
        subsets: list[AzureVMDataAccessor] = []

        for i in range(self.n_split):
            if i == self.n_split - 1:
                data = data.iloc[i * split_size :]
            else:
                data = data.iloc[i * split_size : (i + 1) * split_size]
            subsets.append(
                self.__base_call__(data.reset_index(drop=True), shuffle)
            )
        return subsets


class AzureVMDatasetDistChunkGenerator(
    BaseAzureVMDatasetGenerator[
        list[AzureVMDataAccessor],
        AzureVMDataAccessor,
        AzureVMDataset,
    ]
):
    dataset_cls = AzureVMDataset
    accessor_cls = AzureVMDataAccessor

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str,
        dist_col: str = "dist_id",
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: BaseTransform | list[BaseTransform] | None = None,
    ):
        super(AzureVMDatasetDistChunkGenerator, self).__init__(
            file=file,
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self.dist_col = dist_col

    def __call__(self, shuffle: bool = False) -> list[AzureVMDataAccessor]:
        subsets: list[AzureVMDataAccessor] = []
        grouped = self.data.groupby(self.dist_col)

        for _, data in grouped:
            subsets.append(
                self.__base_call__(data.reset_index(drop=True), shuffle)
            )

        return subsets


__all__ = [
    "AzureVMDataset",
    "AzureVMDataAccessor",
    "AzureVMDatasetGenerator",
    "AzureVMDatasetChunkGenerator",
    "AzureVMDatasetDistChunkGenerator",
]
