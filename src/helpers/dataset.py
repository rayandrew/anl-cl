from collections.abc import Sequence

from avalanche.benchmarks.utils import (
    AvalancheDataset,
    make_classification_dataset,
)

from src.dataset.base import BaseDatasetAccessor


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
        ),
        test=make_classification_dataset(
            dataset.test,
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
