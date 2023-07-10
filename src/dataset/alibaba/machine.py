from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import torch

from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
)

import numpy as np
import pandas as pd

from src.dataset.base import TDatasetSubset, assert_dataset_subset
from src.utils.general import split_evenly_by_classes

from .base import AlibabaDataset

TAlibabaMachineOutput = Literal[
    "cpu_util_percent", "mem_util_percent", "disk_io_percent"
]


def assert_alibaba_machine_output(output: TAlibabaMachineOutput):
    assert output in [
        "cpu_util_percent",
        "mem_util_percent",
        "disk_io_percent",
    ], "output must be one of 'cpu_util_percent', 'mem_util_percent', 'disk_io_percent'"


class AlibabaMachineDataset(AlibabaDataset):
    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        timestamps: pd.DataFrame,
    ):
        super().__init__()
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


class AlibabaMachineDatasetGenerator:
    def __init__(
        self,
        filename: Union[str, Path] = None,
        n_labels: int = 4,
        train_ratio: float = AlibabaDataset.TRAIN_RATIO,
        dataframe: Optional[pd.DataFrame] = None,
        y: TAlibabaMachineOutput = "cpu_util_percent",
    ):
        """Dataset for Alibaba Machine dataset

        Args:
            filename (Union[str, Path]): Path to the dataset file
            n_labels (int): Number of labels to use
            train_ratio (float, optional): Ratio of training data. Defaults to AlibabaDataset.TRAIN_RATIO.
            y (Literal["cpu_util_percent", "mem_util_percent", "disk_io_percent"], optional): Variable to predict. Defaults to "cpu".
        """
        assert_alibaba_machine_output(y)

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
        data = data[
            (data[self.y_var] >= 0) & (data[self.y_var] <= 100)
        ]

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

        ts_train = data_train["time_stamp"]
        X_train = data_train.drop(
            columns=["time_stamp", "machine_id"]
        )

        ts_test = data_test["time_stamp"]
        X_test = data_test.drop(columns=["time_stamp", "machine_id"])

        self.train_features = X_train
        self.train_outputs = y_train
        self.train_timestamps = ts_train

        self.test_features = X_test
        self.test_outputs = y_test
        self.test_timestamps = ts_test
        return AlibabaMachineDataset(
            self.train_features,
            self.train_outputs,
            self.train_timestamps,
        ), AlibabaMachineDataset(
            self.test_features,
            self.test_outputs,
            self.test_timestamps,
        )


@dataclass
class ClassificationAlibabaMachineDataAccessor:
    original_train_dataset: AlibabaMachineDataset
    original_test_dataset: AlibabaMachineDataset
    train_dataset: ClassificationDataset
    test_dataset: ClassificationDataset


def get_classification_alibaba_machine_dataset(
    filename: str,
    n_labels: int = 10,
    y: TAlibabaMachineOutput = "cpu_util_percent",
):
    assert_alibaba_machine_output(y)

    generator = AlibabaMachineDatasetGenerator(
        filename=filename,
        n_labels=n_labels,
        y=y,
    )
    train_dataset, test_dataset = generator()
    avalanche_train_dataset = make_classification_dataset(
        train_dataset
    )
    avalanche_test_dataset = make_classification_dataset(test_dataset)
    return ClassificationAlibabaMachineDataAccessor(
        original_train_dataset=train_dataset,
        original_test_dataset=test_dataset,
        train_dataset=avalanche_train_dataset,
        test_dataset=avalanche_test_dataset,
    )


def get_classification_alibaba_machine_dataset_splitted(
    filename: str,
    n_labels: int = 10,
    y: TAlibabaMachineOutput = "cpu_util_percent",
    num_split: int = 4,
) -> Sequence[ClassificationAlibabaMachineDataAccessor]:
    assert_alibaba_machine_output(y)

    raw_data = pd.read_csv(filename)
    size = len(raw_data)
    split_size = size // num_split

    subsets = []
    for i in range(num_split):
        if i == num_split - 1:
            data = raw_data.iloc[i * split_size :]
        else:
            data = raw_data.iloc[
                i * split_size : (i + 1) * split_size
            ]

        generator = AlibabaMachineDatasetGenerator(
            dataframe=data,
            n_labels=n_labels,
            y=y,
        )
        train_dataset, test_dataset = generator()
        avalanche_train_dataset = make_classification_dataset(
            train_dataset
        )
        avalanche_test_dataset = make_classification_dataset(
            test_dataset
        )
        subsets.append(
            ClassificationAlibabaMachineDataAccessor(
                original_train_dataset=train_dataset,
                original_test_dataset=test_dataset,
                train_dataset=avalanche_train_dataset,
                test_dataset=avalanche_test_dataset,
            )
        )
    return subsets


__all__ = [
    "AlibabaMachineDataset",
    "AlibabaMachineDatasetGenerator",
    "ClassificationAlibabaMachineDataAccessor",
    "get_classification_alibaba_machine_dataset",
    "get_classification_alibaba_machine_dataset_splitted",
]

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # parser.add_argument("-d", "--data", type=str, default="data/mu_dist/m_25.csv")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="out_preprocess/m_25/m_25.csv",
    )
    parser.add_argument("-n", "--n_labels", type=int, default=10)
    parser.add_argument(
        "-m",
        "--subset",
        type=str,
        choices=["training", "testing", "all"],
        default="training",
    )
    parser.add_argument(
        "-y",
        type=str,
        choices=[
            "cpu_util_percent",
            "mem_util_percent",
            "disk_io_percent",
        ],
        default="cpu",
    )
    parser.add_argument(
        "-s",
        "--seq",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--seq_len",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--univariate",
        action="store_true",
    )
    args = parser.parse_args()

    if args.seq:
        dataset, raw_dataset = AlibabaMachineSequenceDataset(
            filename=args.data,
            n_labels=args.n_labels,
            y=args.y,
            subset=args.subset,
            seq_len=args.seq_len,
            univariate=args.univariate,
        )
    else:
        dataset, raw_dataset = AlibabaMachineDataset(
            filename=args.data,
            n_labels=args.n_labels,
            y=args.y,
            subset=args.subset,
        )
    print("INPUT SIZE", raw_dataset.input_size)
    print("N EXPERIENCES", raw_dataset.n_experiences)
    # print("TARGETS", np.unique(dataset.targets))
    # print("OUTPUTS", np.unique(dataset.outputs))
    print("LENGTH", len(dataset))
    for d in dataset:
        print(d)
        break

    # print(dataset[3911])
    # print(dataset[3912])
    # print(dataset[3913])
    # print(dataset[3914])
    # print(dataset[3915])
    # print(dataset[3916])
    # print(dataset[3917])