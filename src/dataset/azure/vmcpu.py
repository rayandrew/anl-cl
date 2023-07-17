from dataclasses import dataclass
from typing import Any

from src.dataset.base import (
    BaseDataset,
    BaseDatasetAccessor,
    BaseDatasetPrototype,
)
from src.utils.logging import logging

log = logging.getLogger(__name__)


class AzureVMDataset(BaseDataset):
    def __init__(
        self,
        features: Any,
        targets: Any,
        tasks: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(
            features=features,
            targets=targets,
            tasks=tasks,
            **kwargs,
        )

    def __getitem__(self, index: int):
        return super().__getitem__(index)


@dataclass
class AzureVMDataAccessor(BaseDatasetAccessor[AzureVMDataset]):
    pass


class AzureVMDatasetPrototype(
    BaseDatasetPrototype[AzureVMDataset, AzureVMDataAccessor]
):
    def create_accessor(
        self, train: AzureVMDataset, test: AzureVMDataset, **kwargs
    ) -> AzureVMDataAccessor:
        return AzureVMDataAccessor(
            train=train,
            test=test,
            **kwargs,
        )

    def create_dataset(self, X: Any, y: Any, **kwargs) -> AzureVMDataset:
        return AzureVMDataset(X, y, **kwargs)


__all__ = ["AzureVMDataset", "AzureVMDataAccessor", "AzureVMDatasetPrototype"]
