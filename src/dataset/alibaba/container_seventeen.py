from dataclasses import dataclass
from typing import Any

from src.dataset.base import (
    BaseDataset,
    BaseDatasetAccessor,
    BaseDatasetPrototype,
)


class AlibabaContainerDataset(BaseDataset):
    pass


@dataclass
class AlibabaContainerDataAccessor(
    BaseDatasetAccessor[AlibabaContainerDataset]
):
    pass


class AlibabaContainerDatasetPrototype(
    BaseDatasetPrototype[AlibabaContainerDataset, AlibabaContainerDataAccessor]
):
    def create_accessor(
        self,
        train: AlibabaContainerDataset,
        test: AlibabaContainerDataset,
        **kwargs,
    ) -> AlibabaContainerDataAccessor:
        return AlibabaContainerDataAccessor(
            train=train,
            test=test,
            **kwargs,
        )

    def create_dataset(
        self, X: Any, y: Any, **kwargs
    ) -> AlibabaContainerDataset:
        return AlibabaContainerDataset(X, y, **kwargs)


__all__ = [
    "AlibabaContainerDataset",
    "AlibabaContainerDataAccessor",
    "AlibabaContainerDatasetPrototype",
]
