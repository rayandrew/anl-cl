from dataclasses import dataclass
from typing import Any

from src.dataset.base import (
    BaseDataset,
    BaseDatasetAccessor,
    BaseDatasetPrototype,
)


class AzureVMDataset(BaseDataset):
    pass


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
