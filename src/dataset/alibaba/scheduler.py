from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Tuple

from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.classification_dataset import (
    ClassificationDataset,
)

import numpy as np
import pandas as pd

from src.dataset.base import (
    BaseDataset,
    BaseDatasetAccessor,
    BaseDatasetGenerator,
    BaseDatasetPreprocessor,
    TTransform,
)
from src.utils.general import read_dataframe
from src.utils.general import split_dataset as split_dataset_fn

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


class AlibabaSchedulerDataset(BaseDataset):
    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
    ):
        super(AlibabaSchedulerDataset, self).__init__()
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


@dataclass
class AlibabaSchedulerDataAccessor(
    BaseDatasetAccessor[
        AlibabaSchedulerDataset, ClassificationDataset
    ]
):
    pass


class AlibabaSchedulerDatasetPreprocessor(
    BaseDatasetPreprocessor[AlibabaSchedulerDataset]
):
    def __init__(
        self,
        data: pd.DataFrame,
        y: TAlibabaSchedulerOutput = "cpu_avg",
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        n_labels: int = 4,
        transform: TTransform | None = None,
    ):
        assert_alibaba_scheduler_output(y)

        super(AlibabaSchedulerDatasetPreprocessor, self).__init__(
            data=data,
            y=y,
            train_ratio=train_ratio,
            transform=transform,
        )

        self.n_labels = n_labels

    @property
    def non_feature_columns(self):
        return [
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

    @property
    def output_column(self):
        return f"{self.y_var}"

    @cached_property
    def preprocessed_data(self) -> pd.DataFrame:
        """Cleaned data and preprocessed data

        Returns:
            pd.DataFrame: Preprocessed data
        """
        data = self.data
        data = data.fillna(0)
        data = data[(data.plan_cpu > 0) & (data.plan_mem > 0)]
        data = data.sort_values(by=["start_time"])
        data = self.transform(data)
        return data

    def __call__(
        self, shuffle: bool = True
    ) -> Tuple[AlibabaSchedulerDataset, AlibabaSchedulerDataset]:
        X_train, X_test, y_train, y_test = split_dataset_fn(
            self.preprocessed_data.drop(columns=[self.output_column]),
            self.preprocessed_data[self.output_column],
            test_size=1 - self.train_ratio,
            shuffle=shuffle,
        )

        return AlibabaSchedulerDataset(
            X_train, y_train
        ), AlibabaSchedulerDataset(X_test, y_test)


class AlibabaSchedulerDatasetGenerator(
    BaseDatasetGenerator[
        AlibabaSchedulerDataAccessor,
        AlibabaSchedulerDatasetPreprocessor,
    ]
):
    accessor_cls = AlibabaSchedulerDataAccessor
    preprocessor_cls = AlibabaSchedulerDatasetPreprocessor

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        n_labels: int = 4,
        y: TAlibabaSchedulerOutput = "cpu_avg",
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: TTransform | None = None,
    ):
        assert_alibaba_scheduler_output(y)
        self.data = read_dataframe(file)
        self.n_labels = n_labels
        self.y = y
        self.train_ratio = train_ratio
        self.transform = transform

    def _base_call__(self, data: pd.DataFrame):
        preprocessor = self.preprocessor_cls(
            data=data,
            n_labels=self.n_labels,
            y=self.y,
            train_ratio=self.train_ratio,
            transform=self.transform,
        )
        train_dataset, test_dataset = preprocessor()
        avalanche_train_dataset = make_classification_dataset(
            train_dataset
        )
        avalanche_test_dataset = make_classification_dataset(
            test_dataset
        )
        return self.accessor_cls(
            original_train_dataset=train_dataset,
            original_test_dataset=test_dataset,
            train_dataset=avalanche_train_dataset,
            test_dataset=avalanche_test_dataset,
        )

    def __call__(self):
        return self._base_call__(self.data)


class AlibabaSchedulerDatasetChunkGenerator(
    AlibabaSchedulerDatasetGenerator
):
    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        n_labels: int = 4,
        n_split: int = 4,
        y: TAlibabaSchedulerOutput = "cpu_avg",
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: TTransform | None = None,
    ):
        super(AlibabaSchedulerDatasetChunkGenerator, self).__init__(
            file=file,
            n_labels=n_labels,
            y=y,
            train_ratio=train_ratio,
            transform=transform,
        )
        self.n_split = n_split

    def __call__(self):
        size = len(self.data)
        split_size = size // self.n_split
        subsets: list[AlibabaSchedulerDataAccessor] = []

        for i in range(self.n_split):
            if i == self.n_split - 1:
                data = self.data.iloc[i * split_size :]
            else:
                data = self.data.iloc[
                    i * split_size : (i + 1) * split_size
                ]
            subsets.append(self._base_call__(data))
        return subsets


__all__ = [
    "AlibabaSchedulerDataset",
    "AlibabaSchedulerDataAccessor",
    "AlibabaSchedulerDatasetPreprocessor",
    "AlibabaSchedulerDatasetGenerator",
    "AlibabaSchedulerDatasetChunkGenerator",
]

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
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
    args = parser.parse_args()

    generator = AlibabaSchedulerDatasetChunkGenerator(
        file=args.data,
        n_labels=args.n_labels,
        y=args.y,
    )
    data = generator()
    original_dataset = data[0].original_train_dataset
    print("INPUT SIZE", original_dataset.input_size)
    print("FEATURES", original_dataset.features.columns)
    print("OUTPUT", original_dataset.targets)
    # print("OUTPUT SIZE", data.original_dataset.features.columns)
