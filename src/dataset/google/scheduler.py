from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import torch

from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
)

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    StandardScaler,
    minmax_scale,
)

from src.utils.general import split_dataset

from .base import GoogleDataset

TGoogleSchedulerOutput = Literal["cpu_95"]


def assert_google_scheduler_output(output: TGoogleSchedulerOutput):
    assert output in ["cpu_95"], "output must cpu_95"


def discretize_column(series: pd.Series, n_bins: int = 4):
    # return pd.cut(series, bins=n_bins, labels=list(range(n_bins)))
    return pd.cut(series, bins=n_bins, labels=False)


def append_prev_feature(df, num, colname):
    for i in range(1, num + 1):
        df["prev_" + colname + "_" + str(i)] = (
            df[colname].shift(i).values
        )


class GoogleSchedulerDataset(GoogleDataset):
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


def _default_transform(df: pd.DataFrame):
    return df


class GoogleSchedulerDatasetGenerator:
    def __init__(
        self,
        filename: Union[str, Path] = None,
        n_labels: int = 4,
        train_ratio: float = GoogleDataset.TRAIN_RATIO,
        dataframe: Optional[pd.DataFrame] = None,
        y: TGoogleSchedulerOutput = "cpu_95",
        n_historical: int = 4,
        transform: Callable[
            [pd.DataFrame], pd.DataFrame
        ] = _default_transform,
    ):
        """Dataset for Google Scheduler dataset

        Args:
            filename (Union[str, Path]): Path to the dataset file
            n_labels (int): Number of labels to use
            train_ratio (float, optional): Ratio of training data. Defaults to GoogleDataset.TRAIN_RATIO.
            y: Variable to predict.
            n_historical (int): Number of historical values to use as features. Defaults to 4.
        """
        assert_google_scheduler_output(y)

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
        self.transform = transform

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
        print("1 mil")
        data = pd.read_parquet(
            self.filename, engine="fastparquet", nrows=1000000
        )
        return data

    @cached_property
    def preprocessed_data(self):
        """Cleaned data and preprocessed data

        Returns:
            pd.DataFrame: Preprocessed data
        """
        data = self.raw_data
        data = data.fillna(-1)
        data = data.sort_values(by=["start_time"])
        # append_prev_feature(data, self.n_historical, "req_cpu")
        # append_prev_feature(data, self.n_historical, "req_mem")

        data[self.y_var] = minmax_scale(data[self.y_var])
        data[self.output_column] = discretize_column(
            data[self.y_var], n_bins=self.n_labels
        )
        frequency = data[self.output_column].value_counts()
        # Print the frequency of values
        print(frequency)

        data = data.dropna()
        data = data.reset_index(drop=True)

        feature_columns = [
            "sched_class",
            "priority",
            "req_cpu",
            "req_mem",
            "constraint_mapped",
            "collection_logical_name_mapped",
            "collection_max_per_machine",
            "collection_max_per_switch",
            "collection_vertical_scaling",
            "collection_scheduler",
        ]

        non_feature_columns = [
            col
            for col in data.columns
            if col not in feature_columns + [self.output_column]
        ]
        data = data.drop(columns=non_feature_columns)

        # scaler = Normalizer()
        scaler = StandardScaler()
        print(data.columns)
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
        return GoogleSchedulerDataset(
            self.train_features,
            self.train_outputs,
        ), GoogleSchedulerDataset(
            self.test_features,
            self.test_outputs,
        )


@dataclass
class ClassificationGoogleSchedulerDataAccessor:
    original_train_dataset: GoogleSchedulerDataset
    original_test_dataset: GoogleSchedulerDataset
    train_dataset: ClassificationDataset
    test_dataset: ClassificationDataset


def get_classification_google_scheduler_dataset(
    filename: str,
    n_labels: int = 4,
    y: TGoogleSchedulerOutput = "cpu_95",
    n_historical: int = 4,
):
    assert_google_scheduler_output(y)

    generator = GoogleSchedulerDatasetGenerator(
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
    return ClassificationGoogleSchedulerDataAccessor(
        original_train_dataset=train_dataset,
        original_test_dataset=test_dataset,
        train_dataset=avalanche_train_dataset,
        test_dataset=avalanche_test_dataset,
    )


def get_classification_google_scheduler_dataset_splitted(
    filename: str,
    n_labels: int = 4,
    y: TGoogleSchedulerOutput = "cpu_95",
    num_split: int = 4,
    n_historical: int = 4,
) -> Sequence[ClassificationGoogleSchedulerDataAccessor]:
    assert_google_scheduler_output(y)

    raw_data = pd.read_parquet(filename, engine="fastparquet")
    raw_data = raw_data.head(1000000)
    size = len(raw_data)
    print("SAIZE " + str(size))
    split_size = size // num_split

    subsets = []
    for i in range(num_split):
        if i == num_split - 1:
            data = raw_data.iloc[i * split_size :]
        else:
            data = raw_data.iloc[
                i * split_size : (i + 1) * split_size
            ]

        generator = GoogleSchedulerDatasetGenerator(
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
            ClassificationGoogleSchedulerDataAccessor(
                original_train_dataset=train_dataset,
                original_test_dataset=test_dataset,
                train_dataset=avalanche_train_dataset,
                test_dataset=avalanche_test_dataset,
            )
        )
    return subsets


__all__ = [
    "GoogleSchedulerDataset",
    "GoogleSchedulerDatasetGenerator",
    "ClassificationGoogleSchedulerDataAccessor",
    "get_classification_google_scheduler_dataset",
    "get_classification_google_scheduler_dataset_splitted",
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

    data = get_classification_google_scheduler_dataset(
        filename=args.data,
        n_labels=args.n_labels,
        y=args.y,
        n_historical=args.n_historical,
    )
    print("INPUT SIZE", data.original_test_dataset.input_size)
    print("FEATURES", data.original_test_dataset.features.columns)
    print("OUTPUT", data.original_test_dataset.targets)
    # print("OUTPUT SIZE", data.original_test_dataset.features.columns)
