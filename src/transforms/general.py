from typing import Callable, Literal

import numpy.typing as npt
import pandas as pd

from src.transforms.base import (
    BaseFeatureEngineering,
    Transform,
    apply_transforms,
)
from src.utils.general import append_prev_feature, discretize_column
from src.utils.logging import logging

from .base import BaseTransform

log = logging.getLogger(__name__)


class ColumnsDropTransform(BaseTransform):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop(columns=self.columns, errors="ignore")
        data = data.reset_index(drop=True)
        return data

    def __repr__(self) -> str:
        return f"ColumnsDropTransform(columns={self.columns})"


class DiscretizeColumnTransform(BaseTransform):
    def __init__(
        self,
        column: str,
        new_column: str | None = None,
        n_bins: int = 4,
        drop_original: bool = True,
    ):
        self.column = column
        self.n_bins = n_bins
        self.new_column = column if new_column is None else new_column
        self.drop_original = drop_original

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        column_data = data[self.column]
        data[self.new_column] = discretize_column(column_data, self.n_bins)
        if self.drop_original and self.new_column != self.column:
            data = data.drop(columns=[self.column], errors="ignore")
        return data

    def __repr__(self) -> str:
        return f"DiscretizeColumnTransform(n_bins={self.n_bins})"


class EnumColumnTransform(BaseTransform):
    def __init__(self, column: str, new_column: str | None = None):
        self.column = column
        self.new_column = column if new_column is None else new_column

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.new_column] = data[self.column].astype("category").cat.codes
        return data


class ApplyFnOnColumnTransform(BaseTransform):
    def __init__(
        self,
        fn: Callable[[pd.Series | npt.ArrayLike], pd.Series],
        columns: list[str],
        prefix: str | None = None,
        **kwargs,
    ):
        self.fn = fn
        self.columns = columns
        self.prefix = prefix
        self.kwargs = kwargs

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            column_data = data[column]
            column_name = column
            if self.prefix is not None:
                column_name = f"{self.prefix}_{column_name}"
            data[column_name] = self.fn(column_data, **self.kwargs)
        return data

    def __repr__(self) -> str:
        return "ApplyFnOnColumnTransform()"


class AppendPrevFeatureTransform(BaseTransform):
    def __init__(self, columns: list[str], n_historical: int = 4):
        self.n_historical = n_historical
        self.columns = columns

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            append_prev_feature(data, self.n_historical, column)
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data

    def __repr__(self) -> str:
        return f"AppendPrevFeatureTransform(n_historical={self.n_historical})"


class OneHotColumnTransform(BaseTransform):
    def __init__(
        self,
        column: str,
        prefix: str | None = None,
        drop_first: bool = True,
    ):
        self.column = column
        self.prefix = prefix
        self.drop_first = drop_first

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        dummies_data = pd.get_dummies(
            data[self.column],
            prefix=self.prefix,
            dtype=int,
            drop_first=self.drop_first,
        )
        data = pd.concat([data, dummies_data], axis=1)
        data = data.reset_index(drop=True)
        return data

    def __repr__(self) -> str:
        return f"OneHotColumnTransform(column={self.column})"


class OneHotColumnsTransform(BaseTransform):
    def __init__(self, columns: list[str], drop_first: bool = True):
        self.columns = columns
        self.drop_first = drop_first

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for column in self.columns:
            transform = OneHotColumnTransform(
                column, prefix=column, drop_first=self.drop_first
            )
            data = transform(data)
        return data

    def __repr__(self) -> str:
        return "OneHotColumnsTransform()"


class PassThroughTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def __repr__(self) -> str:
        return "PassThroughTransform()"


class PrintColumnsTransform(BaseTransform):
    def __init__(self, identifier: str = ""):
        super().__init__()
        self.identifier = identifier

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.identifier != "":
            log.info("%s: %s", self.identifier, list(data.columns))
        else:
            log.info("%s", list(data.columns))
        return data

    def __repr__(self) -> str:
        return f"PrintColumnsTransform(identifier={self.identifier})"


class NamedInjectTransform(BaseTransform):
    def __init__(self, identifier: str) -> None:
        super().__init__()
        self.identifier = identifier
        self._transform_fn: Transform | None = None

    def set_transform_fn(self, transform_fn: Transform) -> None:
        self._transform_fn = transform_fn

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return apply_transforms(data, self._transform_fn)

    def __repr__(self) -> str:
        return f"""NamedInjectTransform(
            identifier={self.identifier})"""


def add_transform_to_feature_engineering(
    feature_engineering: BaseFeatureEngineering,
    transform: Transform,
    pos: int | Literal["start", "end"] | str = 0,
    sections: list[Literal["preprocess", "chunk"]] = [],
):
    if len(sections) == 0:
        raise ValueError("sections must be a non-empty list")

    class _AddTransformSet(BaseFeatureEngineering):
        def __init__(
            self,
        ):
            super().__init__()

        @property
        def target_name(self) -> str:
            return feature_engineering.target_name

        @property
        def preprocess_transform_set(self) -> list[Transform] | None:
            preprocess_set = feature_engineering.preprocess_transform_set
            if preprocess_set is None:
                preprocess_set = []
            else:
                preprocess_set = preprocess_set[:]

            if "preprocess" in sections:
                if pos == "start":
                    return [transform] + preprocess_set
                elif pos == "end":
                    return preprocess_set + [transform]
                elif isinstance(pos, int):
                    preprocess_set.insert(pos, transform)
                    return preprocess_set
                else:
                    found = False
                    for _transform in preprocess_set:
                        if isinstance(_transform, NamedInjectTransform):
                            if _transform.identifier == pos:
                                _transform.set_transform_fn(transform)
                                found = True

                    if not found:
                        log.warning(
                            "No transform with identifier %s, skipping...", pos
                        )
                    return preprocess_set
            return preprocess_set

        @property
        def chunk_transform_set(self) -> list[Transform] | None:
            preprocess_set = feature_engineering.chunk_transform_set
            if preprocess_set is None:
                preprocess_set = []
            else:
                preprocess_set = preprocess_set[:]

            if "chunk" in sections:
                if pos == "start":
                    return [transform] + preprocess_set
                elif pos == "end":
                    return preprocess_set + [transform]
                elif isinstance(pos, int):
                    preprocess_set.insert(pos, transform)
                    return preprocess_set
                else:
                    found = False
                    for _transform in preprocess_set:
                        if isinstance(_transform, NamedInjectTransform):
                            if _transform.identifier == pos:
                                _transform.set_transform_fn(transform)
                                found = True
                    if not found:
                        log.warning(
                            "No transform with identifier %s, skipping...", pos
                        )
                    return preprocess_set
            return preprocess_set

        def __repr__(self):
            return feature_engineering.__repr__()

    return _AddTransformSet()


__all__ = [
    "ColumnsDropTransform",
    "DiscretizeColumnTransform",
    "EnumColumnTransform",
    "AppendPrevFeatureTransform",
    "ApplyFnOnColumnTransform",
    "OneHotColumnTransform",
    "OneHotColumnsTransform",
    "add_transform_to_feature_engineering",
    "PassThroughTransform",
    "PrintColumnsTransform",
    "NamedInjectTransform",
]
