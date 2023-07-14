from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Sequence, TypeVar, cast
from functools import cached_property

import pandas as pd

from src.dataset.base import (
    BaseDataset,
    BaseDatasetAccessor,
    BaseDatasetGenerator,
)
from src.transforms import TAcceptableTransform
from src.utils.general import read_dataframe
from src.utils.general import split_dataset as split_dataset_fn

TGoogleSchedulerDataset = TypeVar(
    "TGoogleSchedulerDataset", bound="GoogleSchedulerDataset"
)
TAccessor = TypeVar("TAccessor", bound=BaseDatasetAccessor)
TAccessorReturn = TypeVar(
    "TAccessorReturn",
    bound="GoogleSchedulerDataAccessor"
    | Sequence["GoogleSchedulerDataAccessor"],
)


class GoogleSchedulerDataset(BaseDataset):
    pass


@dataclass
class GoogleSchedulerDataAccessor(BaseDatasetAccessor[GoogleSchedulerDataset]):
    pass


class BaseGoogleSchedulerDatasetGenerator(
    Generic[TAccessorReturn, TAccessor, TGoogleSchedulerDataset],
    BaseDatasetGenerator[TAccessorReturn],
):
    dataset_cls: type[TGoogleSchedulerDataset]
    accessor_cls: type[TAccessor]

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str = "util_cpu",
        n_labels: int = 4,
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: TAcceptableTransform
        | list[TAcceptableTransform]
        | None = None,
    ):
        super(BaseGoogleSchedulerDatasetGenerator, self).__init__(
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self._file = file
        self.n_labels = n_labels

    @cached_property
    def data(self) -> pd.DataFrame:
        # Cache so no repeated reads / transforms
        data = read_dataframe(self._file)
        data = data.sort_values(by=["start_time"])
        data = data.head(200000)
        data = data.reset_index(drop=True)
        # Move transform here to get better history
        data = self.transform(data)
        data = data.fillna(-1)
        return data

    def __base_call__(self, data: pd.DataFrame, shuffle: bool) -> TAccessor:
        # Print the frequency of values
        print(data.columns)
        print("Bucket distribution: ")
        frequency = data[self.target].value_counts()
        print(frequency)

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

    def __call__(self, shuffle: bool = True) -> TAccessorReturn:
        return cast(TAccessorReturn, self.__base_call__(self.data, shuffle))


class GoogleSchedulerDatasetGenerator(
    BaseGoogleSchedulerDatasetGenerator[
        GoogleSchedulerDataAccessor,
        GoogleSchedulerDataAccessor,
        GoogleSchedulerDataset,
    ]
):
    dataset_cls = GoogleSchedulerDataset
    accessor_cls = GoogleSchedulerDataAccessor


class GoogleSchedulerDatasetChunkGenerator(
    BaseGoogleSchedulerDatasetGenerator[
        list[GoogleSchedulerDataAccessor],
        GoogleSchedulerDataAccessor,
        GoogleSchedulerDataset,
    ]
):
    dataset_cls = GoogleSchedulerDataset
    accessor_cls = GoogleSchedulerDataAccessor

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str = "util_cpu",
        n_labels: int = 4,
        n_split: int = 4,
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: TAcceptableTransform
        | list[TAcceptableTransform]
        | None = None,
    ):
        super(GoogleSchedulerDatasetChunkGenerator, self).__init__(
            file=file,
            n_labels=n_labels,
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self.n_split = n_split

    def __call__(
        self, shuffle: bool = True
    ) -> list[GoogleSchedulerDataAccessor]:
        size = len(self.data)
        split_size = size // self.n_split
        subsets: list[GoogleSchedulerDataAccessor] = []

        for i in range(self.n_split):
            if i == self.n_split - 1:
                data = self.data.iloc[i * split_size :]
            else:
                data = self.data.iloc[i * split_size : (i + 1) * split_size]
            subsets.append(
                self.__base_call__(data.reset_index(drop=True), shuffle)
            )
        return subsets


class GoogleSchedulerDatasetDistChunkGenerator(
    BaseGoogleSchedulerDatasetGenerator[
        list[GoogleSchedulerDataAccessor],
        GoogleSchedulerDataAccessor,
        GoogleSchedulerDataset,
    ]
):
    dataset_cls = GoogleSchedulerDataset
    accessor_cls = GoogleSchedulerDataAccessor

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str,
        n_labels: int = 4,
        dist_col: str = "dist_id",
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: TAcceptableTransform
        | list[TAcceptableTransform]
        | None = None,
    ):
        super(GoogleSchedulerDatasetDistChunkGenerator, self).__init__(
            file=file,
            n_labels=n_labels,
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self.dist_col = dist_col

    def __call__(
        self, shuffle: bool = True
    ) -> list[GoogleSchedulerDataAccessor]:
        subsets: list[GoogleSchedulerDataAccessor] = []
        grouped = self.data.groupby(self.dist_col)

        for _, data in grouped:
            subsets.append(
                self.__base_call__(data.reset_index(drop=True), shuffle)
            )

        return subsets


__all__ = [
    "GoogleSchedulerDataset",
    "GoogleSchedulerDataAccessor",
    "GoogleSchedulerDatasetGenerator",
    "GoogleSchedulerDatasetChunkGenerator",
    "GoogleSchedulerDatasetDistChunkGenerator",
]
