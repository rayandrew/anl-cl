from dataclasses import dataclass
from typing import Any
from pathlib import Path
from typing import Generic, Sequence, TypeVar, cast
from functools import cached_property

import pandas as pd

from src.dataset.base import (
    BaseDataset,
    BaseDatasetAccessor,
    BaseDatasetPrototype,
)


class GoogleSchedulerDataset(BaseDataset):
    pass


@dataclass
class GoogleSchedulerDataAccessor(BaseDatasetAccessor[GoogleSchedulerDataset]):
    pass


class GoogleSchedulerDatasetPrototype(
    BaseDatasetPrototype[GoogleSchedulerDataset, GoogleSchedulerDataAccessor]
):
    def create_accessor(
        self,
        train: GoogleSchedulerDataset,
        test: GoogleSchedulerDataset,
        **kwargs,
    ) -> GoogleSchedulerDataAccessor:
        return GoogleSchedulerDataAccessor(
            train=train,
            test=test,
            **kwargs,
        )

    def create_dataset(
        self, X: Any, y: Any, **kwargs
    ) -> GoogleSchedulerDataset:
        return GoogleSchedulerDataset(X, y, **kwargs)


__all__ = [
    "GoogleSchedulerDataset",
    "GoogleSchedulerDataAccessor",
    "GoogleSchedulerDatasetPrototype",
]
