from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Sequence, TypeVar, cast

import pandas as pd

from src.dataset.base import (
    BaseDataset,
    BaseDatasetAccessor,
    BaseDatasetGenerator,
)
from src.transforms import TAcceptableTransform
from src.utils.general import read_dataframe
from src.utils.general import split_dataset as split_dataset_fn

TAlibabaContainerDataset = TypeVar(
    "TAlibabaContainerDataset", bound="AlibabaContainerDataset"
)
TAccessor = TypeVar("TAccessor", bound=BaseDatasetAccessor)
TAccessorReturn = TypeVar(
    "TAccessorReturn",
    bound="AlibabaContainerDataAccessor"
    | Sequence["AlibabaContainerDataAccessor"],
)


class AlibabaContainerDataset(BaseDataset):
    pass


@dataclass
class AlibabaContainerDataAccessor(
    BaseDatasetAccessor[AlibabaContainerDataset]
):
    pass


class BaseAlibabaContainerDatasetGenerator(
    Generic[TAccessorReturn, TAccessor, TAlibabaContainerDataset],
    BaseDatasetGenerator[TAccessorReturn],
):
    dataset_cls: type[TAlibabaContainerDataset]
    accessor_cls: type[TAccessor]

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str = "cpu_avg",
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: TAcceptableTransform
        | list[TAcceptableTransform]
        | None = None,
    ):
        super(BaseAlibabaContainerDatasetGenerator, self).__init__(
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self._file = file

    @property
    def data(self) -> pd.DataFrame:
        return read_dataframe(self._file)

    def __base_call__(self, data: pd.DataFrame, shuffle: bool) -> TAccessor:
        data = self.transform(data)

        X_train, X_test, y_train, y_test = split_dataset_fn(
            data.drop(columns=[self.target]),
            data[self.target],
            test_size=1 - self.train_ratio,
            shuffle=shuffle,
        )
        return self.accessor_cls(
            train=self.dataset_cls(X_train, y_train),
            test=self.dataset_cls(X_test, y_test),
        )

    def __call__(self, shuffle: bool = False) -> TAccessorReturn:
        return cast(TAccessorReturn, self.__base_call__(self.data, shuffle))


class AlibabaContainerDatasetGenerator(
    BaseAlibabaContainerDatasetGenerator[
        AlibabaContainerDataAccessor,
        AlibabaContainerDataAccessor,
        AlibabaContainerDataset,
    ]
):
    dataset_cls = AlibabaContainerDataset
    accessor_cls = AlibabaContainerDataAccessor


class AlibabaContainerDatasetChunkGenerator(
    BaseAlibabaContainerDatasetGenerator[
        list[AlibabaContainerDataAccessor],
        AlibabaContainerDataAccessor,
        AlibabaContainerDataset,
    ]
):
    dataset_cls = AlibabaContainerDataset
    accessor_cls = AlibabaContainerDataAccessor

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str = "cpu_avg",
        n_split: int = 4,
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: TAcceptableTransform
        | list[TAcceptableTransform]
        | None = None,
    ):
        super(AlibabaContainerDatasetChunkGenerator, self).__init__(
            file=file,
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self.n_split = n_split

    def __call__(
        self, shuffle: bool = False
    ) -> list[AlibabaContainerDataAccessor]:
        size = len(self.data)
        split_size = size // self.n_split
        subsets: list[AlibabaContainerDataAccessor] = []

        for i in range(self.n_split):
            if i == self.n_split - 1:
                data = self.data.iloc[i * split_size :]
            else:
                data = self.data.iloc[i * split_size : (i + 1) * split_size]
            subsets.append(
                self.__base_call__(data.reset_index(drop=True), shuffle)
            )
        return subsets


class AlibabaContainerDatasetDistChunkGenerator(
    BaseAlibabaContainerDatasetGenerator[
        list[AlibabaContainerDataAccessor],
        AlibabaContainerDataAccessor,
        AlibabaContainerDataset,
    ]
):
    dataset_cls = AlibabaContainerDataset
    accessor_cls = AlibabaContainerDataAccessor

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str,
        dist_col: str = "dist_id",
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: TAcceptableTransform
        | list[TAcceptableTransform]
        | None = None,
    ):
        super(AlibabaContainerDatasetDistChunkGenerator, self).__init__(
            file=file,
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self.dist_col = dist_col

    def __call__(
        self, shuffle: bool = False
    ) -> list[AlibabaContainerDataAccessor]:
        subsets: list[AlibabaContainerDataAccessor] = []
        grouped = self.data.groupby(self.dist_col)

        for _, data in grouped:
            subsets.append(
                self.__base_call__(data.reset_index(drop=True), shuffle)
            )

        return subsets


__all__ = [
    "AlibabaContainerDataset",
    "AlibabaContainerDataAccessor",
    "AlibabaContainerDatasetGenerator",
    "AlibabaContainerDatasetChunkGenerator",
    "AlibabaContainerDatasetDistChunkGenerator",
]
