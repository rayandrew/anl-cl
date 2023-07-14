# pyright: reportUndefinedVariable=false
# noqa: F821

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.dataset.generator import SplitChunkGenerator
from src.helpers.config import Config, assert_config_params
from src.helpers.dataset import (
    AvalancheClassificationDatasetAccessor,
    create_avalanche_classification_datasets,
)
from src.helpers.definitions import Dataset, Snakemake
from src.helpers.features import get_features
from src.helpers.scenario import train_classification_scenario
from src.utils.logging import logging, setup_logging

if TYPE_CHECKING:
    snakemake: Snakemake = Snakemake()

setup_logging(snakemake.log[0])
log = logging.getLogger(__name__)


def get_dataset(config: Config, input_path: Path):
    feature_engineering = get_features(config)
    match config.dataset.name:
        case Dataset.ALIBABA:
            from src.dataset.alibaba.container_seventeen import (
                AlibabaContainerDataAccessor,
                AlibabaContainerDataset,
                AlibabaContainerDatasetPrototype,
            )

            generator: SplitChunkGenerator[
                AlibabaContainerDataset, AlibabaContainerDataAccessor
            ] = SplitChunkGenerator(
                file=input_path,
                prototype=AlibabaContainerDatasetPrototype(),
                target=feature_engineering.target_name,
                n_split=config.scenario.num_split,  # type: ignore
                feature_engineering=feature_engineering,
                train_ratio=config.train_ratio,
            )
        case Dataset.AZURE:
            from src.dataset.azure.vmcpu import (
                AzureVMDataAccessor,
                AzureVMDataset,
                AzureVMDatasetPrototype,
            )

            generator: SplitChunkGenerator[
                AzureVMDataset, AzureVMDataAccessor
            ] = SplitChunkGenerator(
                data=input_path,
                prototype=AzureVMDatasetPrototype(),
                target=feature_engineering.target_name,
                n_split=config.scenario.num_split,  # type: ignore
                feature_engineering=feature_engineering,
                train_ratio=config.train_ratio,
            )
        case "google":
            from src.dataset.google.scheduler2 import (
                GoogleSchedulerDataAccessor,
                GoogleSchedulerDataset,
                GoogleSchedulerDatasetPrototype,
            )

            generator: SplitChunkGenerator[
                GoogleSchedulerDataset, GoogleSchedulerDataAccessor
            ] = SplitChunkGenerator(
                data=input_path,
                prototype=GoogleSchedulerDatasetPrototype(),
                target=feature_engineering.target_name,
                n_split=config.scenario.num_split,  # type: ignore
                feature_engineering=feature_engineering,
                train_ratio=config.train_ratio,
            )
        case _:
            raise ValueError(f"Unknown dataset: {config.dataset.name}")

    log.info("%s", feature_engineering)
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

    train_subsets: list[Any] = [subset.train for subset in dataset]
    test_subsets: list[Any] = [subset.test for subset in dataset]
    benchmark = dataset_benchmark(train_subsets, test_subsets)
    return benchmark


def main():
    params = snakemake.params
    config = snakemake.config
    print(config)
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
