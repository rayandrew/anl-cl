from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union, Optional, Sequence, Any
from functools import cached_property

import torch

from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer

from src.utils.general import split_evenly_by_classes, split_dataset

from .base import AlibabaDataset


TAlibabaSchedulerOutput = Literal[
    "mem_avg", "mem_max", "cpu_avg", "cpu_max", "duration"
]


def assert_alibaba_scheduler_output(output: TAlibabaSchedulerOutput):
    assert output in [
        "mem_avg",
        "mem_max",
        "cpu_avg",
        "cpu_max",
        # "duration",
    ], "output must be one of 'mem_avg', 'mem_max', 'cpu_avg', 'cpu_max', 'duration'"


def discretize_column(series: pd.Series, n_bins: int = 4):
    # return pd.cut(series, bins=n_bins, labels=list(range(n_bins)))
    return pd.cut(series, bins=n_bins, labels=False)


def append_prev_feature(df, num, colname):
    for i in range(1, num + 1):
        df["prev_" + colname + "_" + str(i)] = (
            df[colname].shift(i).values
        )


class AlibabaSchedulerDataset(AlibabaDataset):
    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
    ):
        super().__init__()
        self.features = features
        self.targets = targets.values

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
        return (
            self.features.iloc[index].astype(np.float32).values,
            self.targets[index],
        )


class AlibabaSchedulerDatasetGenerator:
    def __init__(
        self,
        filename: Union[str, Path] = None,
        n_labels: int = 4,
        train_ratio: float = AlibabaDataset.TRAIN_RATIO,
        dataframe: Optional[pd.DataFrame] = None,
        y: TAlibabaSchedulerOutput = "cpu_avg",
        n_historical: int = 4,
    ):
        """Dataset for Alibaba Scheduler dataset

        Args:
            filename (Union[str, Path]): Path to the dataset file
            n_labels (int): Number of labels to use
            train_ratio (float, optional): Ratio of training data. Defaults to AlibabaDataset.TRAIN_RATIO.
            y: Variable to predict. Defaults to "cpu_avg".
            n_historical (int): Number of historical values to use as features. Defaults to 4.
        """
        assert_alibaba_scheduler_output(y)

        if dataframe is not None and filename is not None:
            raise ValueError(
                "Only one of dataframe or filename should be specified"
            )

        self.dataframe = dataframe
        self.filename = filename
        self.train_ratio = train_ratio
        self.n_labels = n_labels
        self.y_var = y
        self.n_historical = n_historical

    # def _clean_data(self, data: pd.DataFrame):
    #     data = data.fillna(0)
    #     data = data.drop_duplicates()
    #     data = data.reset_index(drop=True)

    #     # remove invalid values (negative or > 100) because this is a percentage
    #     data = data[
    #         (data[self.y_var] >= 0) & (data[self.y_var] <= 100)
    #     ]
    #     data = data[(data.plan_cpu > 0) & (data.plan_mem > 0)]
    #     data = data.sort_values(by=["instance_start_time"])
    #     data = data.reset_index(drop=True)

    #     return data
    @property
    def output_column(self):
        return f"{self.y_var}_pred"

    @cached_property
    def raw_data(self):
        if self.dataframe is not None:
            return self.dataframe
        data = pd.read_parquet(self.filename, engine="fastparquet")
        return data

    @cached_property
    def preprocessed_data(self):
        """Cleaned data and preprocessed data

        Returns:
            pd.DataFrame: Preprocessed data
        """
        data = self.raw_data
        data = data.fillna(0)
        data = data[(data.plan_cpu > 0) & (data.plan_mem > 0)]
        data = data.sort_values(by=["start_time"])
        append_prev_feature(data, self.n_historical, "plan_cpu")
        append_prev_feature(data, self.n_historical, "plan_mem")
        append_prev_feature(data, self.n_historical, "instance_num")
        data[self.output_column] = discretize_column(
            data.cpu_avg, n_bins=self.n_labels
        )
        data = data.dropna()
        data = data.reset_index(drop=True)

        non_feature_columns = [
            "name",
            # "task_type",
            "status",
            "start_time",
            "end_time",
            # "instance_num",
            # "plan_cpu",
            # "plan_mem",
            "instance_name",
            # "instance_name.1",
            "instance_start_time",
            "instance_end_time",
            "machine_id",
            "seq_no",
            "total_seq_no",
            # "instance_name",
            "cpu_avg",
            "cpu_max",
            "mem_avg",
            "mem_max",
        ]
        feature_columns = [
            col
            for col in data.columns
            if col not in non_feature_columns + [self.output_column]
        ]
        data = data.drop(columns=non_feature_columns)

        scaler = Normalizer()
        data[feature_columns] = scaler.fit_transform(
            data[feature_columns]
        )

        return data

    def __call__(self):
        (
            data_train,
            data_test,
            y_train,
            y_test,
        ) = split_dataset(
            self.preprocessed_data.drop(columns=[self.output_column]),
            self.preprocessed_data[self.output_column],
            test_size=1 - self.train_ratio,
            # train_ratio=self.train_ratio,
            shuffle=True,
        )

        X_train = data_train
        X_test = data_test

        self.train_features = X_train
        self.train_outputs = y_train

        self.test_features = X_test
        self.test_outputs = y_test
        return AlibabaSchedulerDataset(
            self.train_features,
            self.train_outputs,
        ), AlibabaSchedulerDataset(
            self.test_features,
            self.test_outputs,
        )


@dataclass
class ClassificationAlibabaSchedulerDataAccessor:
    original_train_dataset: AlibabaSchedulerDataset
    original_test_dataset: AlibabaSchedulerDataset
    train_dataset: ClassificationDataset
    test_dataset: ClassificationDataset


def get_classification_alibaba_scheduler_dataset(
    filename: str,
    n_labels: int = 4,
    y: TAlibabaSchedulerOutput = "cpu_util_percent",
    n_historical: int = 4,
):
    assert_alibaba_scheduler_output(y)

    generator = AlibabaSchedulerDatasetGenerator(
        filename=filename,
        n_labels=n_labels,
        y=y,
        n_historical=n_historical,
    )
    train_dataset, test_dataset = generator()
    avalanche_train_dataset = make_classification_dataset(
        train_dataset
    )
    avalanche_test_dataset = make_classification_dataset(test_dataset)
    return ClassificationAlibabaSchedulerDataAccessor(
        original_train_dataset=train_dataset,
        original_test_dataset=test_dataset,
        train_dataset=avalanche_train_dataset,
        test_dataset=avalanche_test_dataset,
    )


def get_classification_alibaba_scheduler_dataset_splitted(
    filename: str,
    n_labels: int = 4,
    y: TAlibabaSchedulerOutput = "cpu_util_percent",
    num_split: int = 4,
    n_historical: int = 4,
) -> Sequence[ClassificationAlibabaSchedulerDataAccessor]:
    assert_alibaba_scheduler_output(y)

    raw_data = pd.read_parquet(filename, engine="fastparquet")
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

        generator = AlibabaSchedulerDatasetGenerator(
            dataframe=data,
            n_labels=n_labels,
            y=y,
            n_historical=n_historical,
        )
        train_dataset, test_dataset = generator()
        avalanche_train_dataset = make_classification_dataset(
            train_dataset
        )
        avalanche_test_dataset = make_classification_dataset(
            test_dataset
        )
        subsets.append(
            ClassificationAlibabaSchedulerDataAccessor(
                original_train_dataset=train_dataset,
                original_test_dataset=test_dataset,
                train_dataset=avalanche_train_dataset,
                test_dataset=avalanche_test_dataset,
            )
        )
    return subsets


__all__ = [
    "AlibabaSchedulerDataset",
    "AlibabaSchedulerDatasetGenerator",
    "ClassificationAlibabaSchedulerDataAccessor",
    "get_classification_alibaba_scheduler_dataset",
    "get_classification_alibaba_scheduler_dataset_splitted",
]

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # parser.add_argument("-d", "--data", type=str, default="data/mu_dist/m_25.csv")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="raw_data/alibaba/chunk-0.parquet",
    )
    parser.add_argument("-n", "--n_labels", type=int, default=4)
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
            "mem_avg",
            "mem_max",
            "cpu_avg",
            "cpu_max",
            "duration",
        ],
        default="cpu_avg",
    )
    parser.add_argument(
        "-w",
        "--n_historical",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    data = get_classification_alibaba_scheduler_dataset(
        filename=args.data,
        n_labels=args.n_labels,
        y=args.y,
        n_historical=args.n_historical,
    )
    print("INPUT SIZE", data.original_test_dataset.input_size)
    print("FEATURES", data.original_test_dataset.features.columns)
    print("OUTPUT", data.original_test_dataset.targets.unique())
    # print("OUTPUT SIZE", data.original_test_dataset.features.columns)
