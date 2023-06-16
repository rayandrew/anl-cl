from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.helpers.config import Config
from src.helpers.definitions import Snakemake

# from src.helpers.dataset import (
#     get_splitted_dataset as get_splitted_dataset_helper,
# )
from src.helpers.scenario import train_classification_scenario

if TYPE_CHECKING:
    snakemake: Snakemake


def get_dataset(config: Config, input_path: Path):
    from src.dataset.alibaba import AlibabaSchedulerDatasetGenerator

    generator = AlibabaSchedulerDatasetGenerator(
        file=input_path,
        y=config.dataset.y,
        n_labels=config.num_classes,
    )
    dataset = generator()
    return dataset, dataset.original_test_dataset.input_size


def get_benchmark(dataset: Any):
    raise NotImplementedError


def main():
    train_classification_scenario(
        snakemake,
        get_dataset=get_dataset,
        get_benchmark=get_benchmark,
    )


if __name__ == "__main__":
    main()
