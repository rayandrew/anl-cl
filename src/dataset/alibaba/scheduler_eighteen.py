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
from src.utils.ds import StrEnum
from src.utils.general import read_dataframe
from src.utils.general import split_dataset as split_dataset_fn


class TAlibabaSchedulerTarget(StrEnum):
    MEM_AVG = "mem_avg"
    MEM_MAX = "mem_max"
    CPU_AVG = "cpu_avg"
    CPU_MAX = "cpu_max"
    DURATION = "duration"


TAlibabaSchedulerDataset = TypeVar(
    "TAlibabaSchedulerDataset", bound="AlibabaSchedulerDataset"
)
TAccessor = TypeVar("TAccessor", bound=BaseDatasetAccessor)
TAccessorReturn = TypeVar(
    "TAccessorReturn",
    bound="AlibabaSchedulerDataAccessor"
    | Sequence["AlibabaSchedulerDataAccessor"],
)
# TAlibabaAccessor = TypeVar("TAccessor", bound="TAccessor")

NON_FEATURE_COLUMNS = [
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


def assert_alibaba_scheduler_target(
    output: str | TAlibabaSchedulerTarget,
):
    assert output in TAlibabaSchedulerTarget, (
        "output must be one of 'mem_avg',"
        " 'mem_max', 'cpu_avg', 'cpu_max', 'duration'"
    )


class AlibabaSchedulerDataset(BaseDataset):
    pass


@dataclass
class AlibabaSchedulerDataAccessor(
    BaseDatasetAccessor[AlibabaSchedulerDataset]
):
    pass


class BaseAlibabaSchedulerDatasetGenerator(
    Generic[TAccessorReturn, TAccessor, TAlibabaSchedulerDataset],
    BaseDatasetGenerator[TAccessorReturn],
):
    dataset_cls: type[TAlibabaSchedulerDataset]
    accessor_cls: type[TAccessor]

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str = "cpu_avg",
        n_labels: int = 4,
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: BaseTransform | list[BaseTransform] | None = None,
    ):
        assert_alibaba_scheduler_target(target)
        super(BaseAlibabaSchedulerDatasetGenerator, self).__init__(
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self._file = file
        self.n_labels = n_labels

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


class AlibabaSchedulerDatasetGenerator(
    BaseAlibabaSchedulerDatasetGenerator[
        AlibabaSchedulerDataAccessor,
        AlibabaSchedulerDataAccessor,
        AlibabaSchedulerDataset,
    ]
):
    dataset_cls = AlibabaSchedulerDataset
    accessor_cls = AlibabaSchedulerDataAccessor


class AlibabaSchedulerDatasetChunkGenerator(
    BaseAlibabaSchedulerDatasetGenerator[
        list[AlibabaSchedulerDataAccessor],
        AlibabaSchedulerDataAccessor,
        AlibabaSchedulerDataset,
    ]
):
    dataset_cls = AlibabaSchedulerDataset
    accessor_cls = AlibabaSchedulerDataAccessor

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str = "cpu_avg",
        n_labels: int = 4,
        n_split: int = 4,
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: BaseTransform | None = None,
    ):
        super(AlibabaSchedulerDatasetChunkGenerator, self).__init__(
            file=file,
            n_labels=n_labels,
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self.n_split = n_split

    def __call__(
        self, shuffle: bool = False
    ) -> list[AlibabaSchedulerDataAccessor]:
        size = len(self.data)
        split_size = size // self.n_split
        subsets: list[AlibabaSchedulerDataAccessor] = []

        for i in range(self.n_split):
            if i == self.n_split - 1:
                data = self.data.iloc[i * split_size :]
            else:
                data = self.data.iloc[i * split_size : (i + 1) * split_size]
            subsets.append(
                self.__base_call__(data.reset_index(drop=True), shuffle)
            )
        return subsets


class AlibabaSchedulerDatasetDistChunkGenerator(
    BaseAlibabaSchedulerDatasetGenerator[
        list[AlibabaSchedulerDataAccessor],
        AlibabaSchedulerDataAccessor,
        AlibabaSchedulerDataset,
    ]
):
    dataset_cls = AlibabaSchedulerDataset
    accessor_cls = AlibabaSchedulerDataAccessor

    def __init__(
        self,
        file: str | Path | pd.DataFrame,
        target: str = "cpu_avg",
        n_labels: int = 4,
        dist_col: str = "dist_id",
        train_ratio: float = BaseDataset.TRAIN_RATIO,
        transform: BaseTransform | list[BaseTransform] | None = None,
    ):
        super(AlibabaSchedulerDatasetDistChunkGenerator, self).__init__(
            file=file,
            n_labels=n_labels,
            target=target,
            train_ratio=train_ratio,
            transform=transform,
        )
        self.dist_col = dist_col

    def __call__(
        self, shuffle: bool = False
    ) -> list[AlibabaSchedulerDataAccessor]:
        subsets: list[AlibabaSchedulerDataAccessor] = []
        grouped = self.data.groupby(self.dist_col)

        for _, data in grouped:
            subsets.append(
                self.__base_call__(data.reset_index(drop=True), shuffle)
            )

        return subsets


__all__ = [
    "NON_FEATURE_COLUMNS",
    "AlibabaSchedulerDataset",
    "AlibabaSchedulerDataAccessor",
    "AlibabaSchedulerDatasetGenerator",
    "AlibabaSchedulerDatasetChunkGenerator",
    "AlibabaSchedulerDatasetDistChunkGenerator",
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

    generator = AlibabaSchedulerDatasetGenerator(
        file=args.data,
        n_labels=args.n_labels,
        target=args.y,
    )
    data = generator.generate()
    train_dataset = data.train
    print("INPUT SIZE", train_dataset.input_size)
    print("FEATURES", train_dataset.features.columns)
    print("OUTPUT", train_dataset.targets)
