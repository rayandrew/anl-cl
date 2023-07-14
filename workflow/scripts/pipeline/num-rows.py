from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.drift_detection.voting import get_offline_voting_drift_detector
from src.helpers.config import Config, assert_config_params
from src.helpers.dataset import create_avalanche_classification_datasets
from src.helpers.definitions import Dataset, Snakemake
from src.helpers.features import get_features
from src.helpers.scenario import train_classification_scenario
from src.transforms import add_transform_to_transform_set
from src.utils.logging import logging, setup_logging

if TYPE_CHECKING:
    snakemake: Snakemake = Snakemake()

DIST_COLUMN = "dist_id"

setup_logging(snakemake.log[0])
log = logging.getLogger(__name__)


def add_dist_label_row(data: pd.DataFrame, num_rows:int, start_from=0):
    distributions = np.zeros(len(data), dtype=int)

    log.info(f"Number of distributions: {(len(data) + num_rows - 1) // num_rows}")
    for idx in range(len(data)):
        if idx % num_rows == 0 and idx != 0:
            start_from += 1
        distributions[idx] = start_from

    data[DIST_COLUMN] = pd.Series(distributions)
    # distribution_counts = data.groupby('dist_id').size()
    # print(distribution_counts.to_string(index=False))
    return data



def num_rows_transform(config: Config, target_name: str):
    def transform(data: pd.DataFrame) -> pd.DataFrame:
        data = add_dist_label_row(data, config.scenario.period)
        return data

    return transform


def get_dataset(config: Config, input_path: Path):

    data_transformer = get_features(config)
    num_rows_transformer = num_rows_transform(config, config.dataset)

    data_transformer = add_transform_to_transform_set(
        data_transformer, num_rows_transformer, 'start'
    )

    match config.dataset.name:
        case Dataset.ALIBABA:
            from src.dataset.alibaba.container_seventeen import (
                AlibabaContainerDatasetDistChunkGenerator,
            )

            generator = AlibabaContainerDatasetDistChunkGenerator(
                file=input_path,
                target=data_transformer.target_name,
                dist_col=DIST_COLUMN,
                transform=data_transformer,
            )
        case Dataset.AZURE:
            from src.dataset.azure.vmcpu import AzureVMDatasetDistChunkGenerator

            generator = AzureVMDatasetDistChunkGenerator(  # type: ignore
                file=input_path,
                target=data_transformer.target_name,
                dist_col=DIST_COLUMN,
                transform=data_transformer,
            )
        case Dataset.GOOGLE:
            from src.dataset.google.scheduler2 import GoogleSchedulerDatasetDistChunkGenerator

            generator = GoogleSchedulerDatasetDistChunkGenerator(
                file=input_path,
                target=data_transformer.target_name,
                n_labels=config.num_classes,
                transform=data_transformer,
            )
        
        case _:
            raise ValueError(f"Unknown dataset: {config.dataset.name}")

    dataset = generator()
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    return (
        create_avalanche_classification_datasets(dataset),
        dataset[0].train.input_size,
    )




def get_benchmark(dataset: Sequence[Any]):
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
