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

TGoogleSchedulerOutput = Literal["util_cpu"]


def assert_google_scheduler_output(output: TGoogleSchedulerOutput):
    assert output in [
        # "cpu_95",
        "util_cpu",
    ], "output must cpu_95 or util_cpu"


def discretize_column(series: pd.Series, n_bins: int = 4):
    # return pd.cut(series, bins=n_bins, labels=list(range(n_bins)))
    bin_edges = [
        -float("inf"),
        1,
        float("inf"),
    ]  # Specify the custom bin edges
    return pd.cut(series, bins=bin_edges, labels=False)


def append_prev_feature(df, num, colname):
    feature_name = []
    for i in range(1, num + 1):
        df["prev_" + colname + "_" + str(i)] = (
            df[colname].shift(i).values
        )
        feature_name.append("prev_" + colname + "_" + str(i))
    return feature_name


def append_history_time(df, colname):
    """colname = column to group the history on
    This function will map jobs with duration >= ~6 minutes as 1, otherwise 0.
    Used to add duration information to a job.
    """
    long_duration_name = str(colname) + "_history_duration_long"
    short_duration_name = str(colname) + "_history_duration_short"

    df[long_duration_name] = 0
    df[short_duration_name] = 0

    histogram_map = {}

    for index, row in df.iterrows():
        group_name = row[colname]
        duration_classification = (
            1 if row["duration"] >= 412000000 else 0
        )

        group_hist = histogram_map.get(
            group_name, {"long": 0, "short": 0}
        )

        total_rows = max(
            group_hist["long"] + group_hist["short"],
            1,
        )

        df.at[index, long_duration_name] = (
            group_hist["long"] / total_rows
        )
        df.at[index, short_duration_name] = (
            group_hist["short"] / total_rows
        )

        if duration_classification == 1:
            group_hist["long"] += 1
        else:
            group_hist["short"] += 1

        # Update the dictionary with the modified histogram for the collection_logical_name
        histogram_map[group_name] = group_hist


def append_history(df, colname):
    """colname = column name to group on
    Will make a history of throttled/non-throttled jobs based on colname
    """
    throttle_name = str(colname) + "_history_throttle"
    non_throttle_name = str(colname) + "_history_non_throttle"

    df[throttle_name] = 0
    df[non_throttle_name] = 0

    histogram_map = {}

    for index, row in df.iterrows():
        collection_name = row[colname]
        cpu_classification = 1 if row["util_cpu"] > 1 else 0

        collection_hist = histogram_map.get(
            collection_name, {"throttle": 0, "non_throttle": 0}
        )

        total_rows = max(
            collection_hist["throttle"]
            + collection_hist["non_throttle"],
            1,
        )
        df.at[index, throttle_name] = (
            collection_hist["throttle"] / total_rows
        )
        df.at[index, non_throttle_name] = (
            collection_hist["non_throttle"] / total_rows
        )

        if cpu_classification == 1:
            collection_hist["throttle"] += 1
        else:
            collection_hist["non_throttle"] += 1

        # Update the dictionary with the modified histogram for the collection_logical_name
        histogram_map[collection_name] = collection_hist


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

        # data[self.y_var] = minmax_scale(data[self.y_var])
        data[self.output_column] = discretize_column(
            data[self.y_var], n_bins=self.n_labels
        )

        # Print the frequency of values
        print("Bucket distribution: ")
        frequency = data[self.output_column].value_counts()
        print(frequency)

        data = data.fillna(-1)
        data = data.reset_index(drop=True)

        # Additional features after mapping
        additional_features = [
            "constraint_mapped_history_throttle",
            "collection_logical_name_mapped_history_throttle",
            "constraint_mapped_history_non_throttle",
            "collection_logical_name_mapped_history_non_throttle",
            "constraint_mapped_history_duration_long",
            "collection_logical_name_mapped_history_duration_long",
            "constraint_mapped_history_duration_short",
            "collection_logical_name_mapped_history_duration_short",
        ]

        feature_columns = [
            "sched_class",
            # "duration",
            # "collection_max_per_machine",
            # "collection_max_per_switch",
            "collection_vertical_scaling",
            "collection_scheduler",
            "priority",
            "req_cpu",
            "req_mem",
            "constraint_mapped",
            "collection_logical_name_mapped",
        ]
        feature_columns_mix = feature_columns + additional_features

        non_feature_columns = [
            col
            for col in data.columns
            if col not in feature_columns_mix + [self.output_column]
        ]
        data = data.drop(columns=non_feature_columns)

        # scaler = Normalizer()
        scaler = StandardScaler()
        data[feature_columns] = scaler.fit_transform(
            data[feature_columns]
        )
        print(data.head(5))
        print(data.dtypes)
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
    raw_data.sort_values(by="start_time", inplace=True)
    # Cut the data here
    raw_data = raw_data.head(200000)
    size = len(raw_data)

    print("SIZE of dataset: " + str(size))
    split_size = size // num_split

    raw_data["duration"] = (
        raw_data["end_time"] - raw_data["start_time"]
    )

    # Append history based on grouping columns on duration (Whether they exceed 6 minutes or not)
    append_history_time(raw_data, "constraint_mapped")
    append_history_time(raw_data, "collection_logical_name_mapped")

    # Append history based on grouping columns on previous throttle/non_throttle
    append_history(raw_data, "collection_logical_name_mapped")
    append_history(raw_data, "constraint_mapped")

    # one_hot = [
    #     "sched_class",
    #     "collection_scheduler",
    #     "collection_vertical_scaling",
    # ]
    # raw_data = pd.get_dummies(raw_data, columns=one_hot)

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
