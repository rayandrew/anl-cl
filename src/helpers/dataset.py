from collections.abc import Sequence
from typing import TypeVar

from avalanche.benchmarks.utils import (
    AvalancheDataset,
    make_classification_dataset,
)

from src.dataset.base import BaseDatasetAccessor
from src.transforms.base import BaseFeatureEngineering

TFeatureEngineering = TypeVar(
    "TFeatureEngineering", bound=BaseFeatureEngineering
)


class AvalancheClassificationDatasetAccessor(
    BaseDatasetAccessor[AvalancheDataset]
):
    pass


def create_avalanche_classification_dataset(
    dataset: BaseDatasetAccessor,
) -> AvalancheClassificationDatasetAccessor:
    return AvalancheClassificationDatasetAccessor(
        train=make_classification_dataset(
            dataset.train,
            task_labels=getattr(dataset.train, "tasks", None),
        ),
        test=make_classification_dataset(
            dataset.test,
            task_labels=getattr(dataset.test, "tasks", None),
        ),
    )


def create_avalanche_classification_datasets(
    datasets: Sequence[BaseDatasetAccessor],
) -> Sequence[AvalancheClassificationDatasetAccessor]:
    return [create_avalanche_classification_dataset(d) for d in datasets]


__all__ = [
    "AvalancheClassificationDatasetAccessor",
    "create_avalanche_classification_dataset",
    "create_avalanche_classification_datasets",
]
