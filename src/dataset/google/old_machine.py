from collections.abc import Collection
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
)

import numpy as np
import pandas as pd

from src.utils.general import split_evenly_by_classes

from .base import GoogleDataset

TGoogleOutput = Literal["cpu"]


def assert_google_output(output: TGoogleOutput):
    assert output in ["cpu"], "output must be 'cpu'"


class GoogleMachineDataset(GoogleDataset):
    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        timestamps: pd.DataFrame,
    ):
        super().__init__()  # type: ignore # noqa # NOTE: @william fix this
        self.features = features
        self.targets = targets.values
        self.timestamps = timestamps

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
        # print(index, self.features.shape, self.targets.shape)
        # print(index)
        # print(index, self.features[index], self.targets[index])
        return (
            self.features.iloc[index].astype(np.float32).values,
            self.targets[index],
        )


class GoogleMachineDatasetGenerator:
    def __init__(
        self,
        filename: str | Path | None = None,
        n_labels: int = 10,
        train_ratio: float = GoogleDataset.TRAIN_RATIO,
        dataframe: pd.DataFrame | None = None,
        y: TGoogleOutput = "cpu",
    ):
        assert_google_output(y)

        if dataframe is not None and filename is not None:
            raise ValueError(
                "Only one of dataframe or filename should be specified"
            )

        self.dataframe = dataframe
        self.filename = filename
        self.train_ratio = train_ratio
        self.n_labels = n_labels
        self.y_var = y

    def _clean_data(self, data: pd.DataFrame):
        data = data.fillna(0)
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)

        # remove invalid values (negative or > 100) because this is a percentage
        data = data[(data[self.y_var] >= 0) & (data[self.y_var] <= 100)]

        return data

    @cached_property
    def raw_data(self):
        if self.dataframe is not None:
            return self.dataframe
        data = pd.read_csv(self.filename)
        return data

    @cached_property
    def cleaned_data(self):
        data = self._clean_data(self.raw_data)
        # change y to be discrete
        data[self.y_var] = pd.qcut(
            data[self.y_var], self.n_labels, labels=False
        )
        return data

    def __call__(self):
        (
            data_train,
            data_test,
            y_train,
            y_test,
        ) = split_evenly_by_classes(
            self.cleaned_data.drop(columns=[self.y_var]),
            self.cleaned_data[self.y_var],
            train_ratio=self.train_ratio,
            shuffle=True,
        )

        ts_train = data_train["start_time"]
        X_train = data_train.drop(columns=["start_time", "end_time"])

        ts_test = data_test["start_time"]
        X_test = data_test.drop(columns=["start_time", "end_time"])

        self.train_features = X_train
        self.train_outputs = y_train
        self.train_timestamps = ts_train

        self.test_features = X_test
        self.test_outputs = y_test
        self.test_timestamps = ts_test
        return GoogleMachineDataset(
            self.train_features,
            self.train_outputs,
            self.train_timestamps,
        ), GoogleMachineDataset(
            self.test_features,
            self.test_outputs,
            self.test_timestamps,
        )


@dataclass
class ClassificationGoogleMachineDataAccessor:
    original_train_dataset: GoogleMachineDataset
    original_test_dataset: GoogleMachineDataset
    train_dataset: ClassificationDataset
    test_dataset: ClassificationDataset


def get_classification_google_machine_dataset(
    filename: str,
    n_labels: int = 10,
    y: TGoogleOutput = "cpu",
):
    assert_google_output(y)

    generator = GoogleMachineDatasetGenerator(
        filename=filename,
        n_labels=n_labels,
        y=y,
    )
    train_dataset, test_dataset = generator()
    avalanche_train_dataset = make_classification_dataset(train_dataset)
    avalanche_test_dataset = make_classification_dataset(test_dataset)
    return ClassificationGoogleMachineDataAccessor(
        original_train_dataset=train_dataset,
        original_test_dataset=test_dataset,
        train_dataset=avalanche_train_dataset,
        test_dataset=avalanche_test_dataset,
    )


def get_classification_google_machine_dataset_splitted(
    filename: str,
    n_labels: int = 10,
    y: TGoogleOutput = "cpu",
    num_split: int = 4,
) -> Collection[ClassificationGoogleMachineDataAccessor]:
    assert_google_output(y)

    raw_data = pd.read_csv(filename)
    size = len(raw_data)
    split_size = size // num_split

    subsets = []
    for i in range(num_split):
        if i == num_split - 1:
            data = raw_data.iloc[i * split_size :]
        else:
            data = raw_data.iloc[i * split_size : (i + 1) * split_size]

        generator = GoogleMachineDatasetGenerator(
            dataframe=data,
            n_labels=n_labels,
            y=y,
        )
        train_dataset, test_dataset = generator()
        avalanche_train_dataset = make_classification_dataset(train_dataset)
        avalanche_test_dataset = make_classification_dataset(test_dataset)
        subsets.append(
            ClassificationGoogleMachineDataAccessor(
                original_train_dataset=train_dataset,
                original_test_dataset=test_dataset,
                train_dataset=avalanche_train_dataset,
                test_dataset=avalanche_test_dataset,
            )
        )
    return subsets


__all__ = [
    "GoogleMachineDataset",
    "GoogleMachineDatasetGenerator",
    "ClassificationGoogleMachineDataAccessor",
    "get_classification_google_machine_dataset",
    "get_classification_google_machine_dataset_splitted",
]
