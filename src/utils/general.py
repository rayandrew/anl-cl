import collections.abc
import math
from numbers import Number
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split


def set_seed(random_seed: Number) -> None:
    from avalanche.training.determinism.rng_manager import RNGManager

    RNGManager.set_random_seeds(random_seed)


split_dataset = train_test_split


def split_evenly_by_classes(
    X: Sequence[Tuple[npt.ArrayLike, Number, Number]],
    y: Sequence[Number],
    train_ratio: float,
    shuffle: bool = True,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=1 - train_ratio,
        stratify=y,
        shuffle=shuffle,
    )

    return X_train, X_test, y_train, y_test


def label_transform(x: float | int, n_labels: int = 10) -> int:
    # we know that the range spans between 0.0 and 100.0
    # we can divide them equally right away based on `n_labels`

    if x <= 0.0:  # edge cases
        return 0

    if x >= 100.0:
        return n_labels - 1

    divider = 100.0 / n_labels

    return min(math.ceil(x / divider) - 1, n_labels - 1)


def custom_round(x, base: float = 10):
    return int(base * round(float(x) / base))


def ceil_up(x, base: float = 10):
    return math.ceil(x / base) * base


def get_checkpoint_fname(cfg: Any):
    return "checkpoint.pt"


def get_model_fname(cfg: Any):
    # return f"{cfg.machine_id}_{cfg.strategy}.pt"
    return "model.pt"


def head(data: Any | Sequence[Any]):
    if isinstance(data, collections.abc.Sequence):
        return data[0]
    if isinstance(data, collections.abc.Set):
        return next(iter(data))
    return data


def discretize_column(series: pd.Series, n_bins: int = 4):
    # return pd.cut(series, bins=n_bins, labels=list(range(n_bins)))
    return pd.cut(series, bins=n_bins, labels=False)


def append_prev_feature(df: pd.DataFrame, num: int, colname: str):
    for i in range(1, num + 1):
        df["prev_" + colname + "_" + str(i)] = (
            df[colname].shift(i).values
        )


def read_dataframe(file: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(file, pd.DataFrame):
        return file

    file = Path(file)
    if file.suffix == ".parquet":
        return pd.read_parquet(file, engine="fastparquet")
    elif file.suffix == ".csv":
        return pd.read_csv(file)
    else:
        raise ValueError(
            "File must be either a parquet or a csv file"
        )


__all__ = [
    "set_seed",
    "split_dataset",
    "split_evenly_by_classes",
    "label_transform",
    "custom_round",
    "ceil_up",
    "get_model_fname",
    "get_checkpoint_fname",
    "head",
    "discretize_column",
    "append_prev_feature",
    "read_dataframe",
    # "add_src_to_path",
]
