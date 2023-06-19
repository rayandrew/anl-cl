from pathlib import Path
from typing import TYPE_CHECKING

import src.transforms.alibaba_scheduler as transforms
from src.helpers.config import Config, assert_config_params
from src.helpers.dataset import (
    AvalancheClassificationDatasetAccessor,
    create_avalanche_classification_datasets,
)
from src.helpers.definitions import Snakemake
from src.helpers.scenario import train_classification_scenario
from src.utils.logging import logging, setup_logging

if TYPE_CHECKING:
    snakemake: Snakemake

setup_logging(snakemake.log[0])
log = logging.getLogger(__name__)


def get_dataset(config: Config, input_path: Path):
    from src.dataset.alibaba import (
        AlibabaSchedulerDatasetChunkGenerator,
    )

    generator = AlibabaSchedulerDatasetChunkGenerator(
        file=input_path,
        target=config.dataset.target,
        n_labels=config.num_classes,
        n_split=config.scenario.num_split,  # type: ignore
        transform=[
            transforms.CleanDataTransform(
                target=config.dataset.target,
            ),
            transforms.AppendPrevFeatureTransform(
                columns=["plan_cpu", "plan_mem", "instance_num"]
            ),
            transforms.DiscretizeOutputTransform(
                target=config.dataset.target,
                n_bins=config.num_classes,
            ),
        ],
    )
    dataset = generator()
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    return (
        create_avalanche_classification_datasets(dataset),
        dataset[0].train.input_size,
    )


def get_benchmark(
    dataset: list[AvalancheClassificationDatasetAccessor],
):
    from avalanche.benchmarks.generators import dataset_benchmark

    train_subsets = [subset.train for subset in dataset]
    test_subsets = [subset.test for subset in dataset]
    benchmark = dataset_benchmark(train_subsets, test_subsets)
    return benchmark


def main():
    params = snakemake.params
    config = snakemake.config
    config = Config(**config)
    assert_config_params(config, params)

    log.info("Config: %s", config)

    train_classification_scenario(
        config=config,
        snakemake=snakemake,
        log=log,
        get_dataset=get_dataset,
        get_benchmark=get_benchmark,
    )


if __name__ == "__main__":
    main()
