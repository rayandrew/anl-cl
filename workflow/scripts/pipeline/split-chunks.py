from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from src.helpers.config import Config
from src.helpers.definitions import Snakemake
from src.helpers.scenario import train_classification_scenario

if TYPE_CHECKING:
    snakemake: Snakemake


def get_dataset(config: Config, input_path: Path):
    from src.dataset.alibaba import (
        AlibabaSchedulerDatasetChunkGenerator,
    )

    generator = AlibabaSchedulerDatasetChunkGenerator(
        file=input_path,
        y=config.dataset.y,
        n_labels=config.num_classes,
        n_split=config.scenario.num_split,
    )
    dataset = generator()
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    return dataset, dataset[0].original_test_dataset.input_size


def get_benchmark(dataset: Sequence[Any]):
    from avalanche.benchmarks.generators import dataset_benchmark

    train_subsets = [subset.train_dataset for subset in dataset]
    test_subsets = [subset.test_dataset for subset in dataset]
    benchmark = dataset_benchmark(train_subsets, test_subsets)
    return benchmark


def main():
    train_classification_scenario(
        snakemake,
        get_dataset=get_dataset,
        get_benchmark=get_benchmark,
    )


if __name__ == "__main__":
    main()
