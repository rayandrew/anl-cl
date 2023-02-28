import math
from pathlib import Path
from typing import Union, Tuple, Literal

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_evenly_by_classes(
    data: pd.DataFrame,
    class_label: str,
    train: bool,
    train_ratio: float,
    shuffle: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[class_label],
        test_size=1 - train_ratio,
        stratify=data[class_label],
        shuffle=shuffle,
    )

    if train:
        return X_train, y_train

    return X_test, y_test


def label_transform(x: float | int, n_labels: int = 10) -> int:
    # we know that the range spans between 0.0 and 100.0
    # we can divide them equally right away based on `n_labels`

    if x <= 0.0:  # edge cases
        return 0

    if x >= 100.0:
        return n_labels - 1

    divider = 100.0 / n_labels

    return min(math.ceil(x / divider) - 1, n_labels - 1)


class AlibabaDataset(Dataset):
    TRAIN_RATIO = 0.8

    FEATURE_COLUMNS = [
        "time_stamp",
        "cpu_avg",
        "cpu_max",
        "mem_avg",
        "mem_max",
        "plan_cpu",
        "plan_mem",
        # "cpu_util_percent",
        # "mem_util_percent",
        # "disk_io_percent"
    ]

    def __init__(
        self,
        filename: Union[str, Path],
        n_labels: int,
        train_ratio: float = TRAIN_RATIO,
        train: bool = False,
        y: Literal[
            "cpu_util_percent", "mem_util_percent", "disk_io_percent"
        ] = "cpu_util_percent",
    ):
        self.filename = filename
        self.train_ratio = train_ratio
        self.train = train
        self.n_labels = n_labels
        self.y_var = y
        self._load_data()

    def _load_data(self):
        self.raw_data = pd.read_csv(self.filename)
        self.raw_data["y"] = self.raw_data[self.y_var]
        self.raw_data.loc[:, "y"] = self.raw_data.y.apply(label_transform)
        self.raw_data = self.raw_data.reset_index(drop=True)

        self.data, self.y = split_evenly_by_classes(
            self.raw_data,
            class_label="y",
            train=self.train,
            train_ratio=self.train_ratio,
        )

        # print("Class distributions:")
        # print(self.data.groupby(["y"]).size())

        self.x = self.data[self.FEATURE_COLUMNS].reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
        self.targets = self.data["dist_label"].reset_index(drop=True).to_numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x.iloc[idx].astype(np.float32).values,
            self.targets[idx],
            self.y[idx],
        )


__all__ = [
    "AlibabaDataset",
]
