from abc import ABCMeta
from pathlib import Path
from typing import Any, Generic, Literal

import pandas as pd

from src.transforms import BaseFeatureEngineering
from src.transforms.general import add_transform_to_feature_engineering
from src.utils.general import read_dataframe
from src.utils.general import split_dataset as split_dataset_fn
from src.utils.logging import logging

from .base import (
    BaseDatasetGenerator,
    BaseDatasetPrototype,
    TAccessor,
    TDataset,
)

log = logging.getLogger(__name__)


def split_dataset(
    data: pd.DataFrame,
    target: str,
    train_ratio: float,
    shuffle: bool = True,
):
    return split_dataset_fn(
        data.drop(columns=[target], errors="ignore"),
        data[target],
        test_size=1 - train_ratio,
        shuffle=shuffle,
    )


class DataFrameGenerator(metaclass=ABCMeta):
    def __init__(self, data: pd.DataFrame | str | Path) -> None:
        self.data = read_dataframe(data)


class SplitChunkGenerator(
    Generic[TDataset, TAccessor],
    DataFrameGenerator,
    BaseDatasetGenerator[TDataset, TAccessor],
):
    def __init__(
        self,
        prototype: BaseDatasetPrototype[TDataset, TAccessor],
        data: pd.DataFrame | str | Path,
        target: str,
        n_split: int,
        train_ratio: float,
        feature_engineering: BaseFeatureEngineering,
    ) -> None:
        DataFrameGenerator.__init__(self, data)
        BaseDatasetGenerator.__init__(
            self=self,
            prototype=prototype,
            target=target,
            feature_engineering=feature_engineering,
            train_ratio=train_ratio,
        )
        self.n_split = n_split

    def __call__(self, shuffle: bool = True) -> list[TAccessor]:
        data = self.feature_engineering.apply_preprocess_transform(self.data)
        size = len(data)
        split_size = size // self.n_split
        subsets: list[Any] = []

        for i in range(self.n_split):
            if i == self.n_split - 1:
                chunk_data = data.iloc[i * split_size :]
            else:
                chunk_data = data.iloc[i * split_size : (i + 1) * split_size]
            chunk_data = chunk_data.reset_index(drop=True)
            chunk_data = self.feature_engineering.apply_chunk_transform(
                chunk_data
            )
            X_train, X_test, y_train, y_test = split_dataset(
                chunk_data, self.target, self.train_ratio, shuffle
            )
            subsets.append(
                self.prototype.create_accessor(
                    self.prototype.create_dataset(X_train, y_train),
                    self.prototype.create_dataset(X_test, y_test),
                )
            )
        return subsets


class DistributionColumnBasedGenerator(
    Generic[TDataset, TAccessor],
    DataFrameGenerator,
    BaseDatasetGenerator[TDataset, TAccessor],
):
    def __init__(
        self,
        prototype: BaseDatasetPrototype[TDataset, TAccessor],
        data: pd.DataFrame | str | Path,
        train_ratio: float,
        target: str,
        feature_engineering: BaseFeatureEngineering,
        dist_col: str = "dist_id",
        max_split: int | Literal["all"] = "all",
    ) -> None:
        DataFrameGenerator.__init__(self, data)
        BaseDatasetGenerator.__init__(
            self=self,
            prototype=prototype,
            target=target,
            feature_engineering=feature_engineering,
            train_ratio=train_ratio,
        )
        self.dist_col = dist_col
        self.feature_engineering = add_transform_to_feature_engineering(
            self.feature_engineering,
            transform=lambda data: data.drop(
                columns=[self.dist_col]
            ).reset_index(drop=True),
            pos=0,
            sections=["chunk"],
        )
        self.max_split = max_split

    def __call__(self, shuffle: bool = True) -> list[TAccessor]:
        subsets: list[TAccessor] = []
        data = self.feature_engineering.apply_preprocess_transform(self.data)
        grouped = data.groupby(self.dist_col)

        log.info(f"Number of groups: {len(grouped)}")

        count = 0
        for _, group_data in grouped:
            if self.max_split != "all" and count >= self.max_split:
                break
            chunk_data = group_data.copy()
            chunk_data = self.feature_engineering.apply_chunk_transform(
                chunk_data
            )
            X_train, X_test, y_train, y_test = split_dataset(
                chunk_data, self.target, self.train_ratio, shuffle
            )
            subsets.append(
                self.prototype.create_accessor(
                    self.prototype.create_dataset(X_train, y_train),
                    self.prototype.create_dataset(X_test, y_test),
                )
            )
            count += 1

        return subsets


__all__ = [
    "DataFrameGenerator",
    "SplitChunkGenerator",
    "DistributionColumnBasedGenerator",
]
