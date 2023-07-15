from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.dataset.generator import DistributionColumnBasedGenerator
from src.drift_detection.voting import get_offline_voting_drift_detector
from src.helpers.config import Config, assert_config_params
from src.helpers.dataset import create_avalanche_classification_datasets
from src.helpers.definitions import DD_DIST_COLUMN, DD_ID, Dataset, Snakemake
from src.helpers.features import get_features
from src.helpers.scenario import train_classification_scenario
from src.transforms.general import add_transform_to_feature_engineering
from src.utils.logging import logging, setup_logging

if TYPE_CHECKING:
    snakemake: Snakemake = Snakemake()


setup_logging(snakemake.log[0])
log = logging.getLogger(__name__)


def add_dist_label(data: pd.DataFrame, dist: Sequence[int], start_from=0):
    distributions = np.zeros(len(data), dtype=int)

    log.info(f"Number of distributions: {len(dist)}")

    log.debug(f"Dist from 0 to {dist[0]}: 0")
    distributions[: dist[0]] = start_from
    for i in range(len(dist) - 1):
        log.debug(f"Dist from {dist[i]} to {dist[i+1]}: {i+1}")
        distributions[dist[i] : dist[i + 1]] = i + 1 + start_from
    distributions[dist[-1] :] = len(dist) + start_from

    data[DD_DIST_COLUMN] = pd.Series(distributions)
    return data


def dd_transform(config: Config, target_name: str):
    dd_params = (
        {}
        if config.drift_detection is None
        else config.drift_detection.dict(exclude={"name"})
    )
    log.info("DD PARAMS: %s", dd_params)
    dd = get_offline_voting_drift_detector(
        **dd_params,
    )

    def transform(data: pd.DataFrame) -> pd.DataFrame:
        change_list = dd.predict(data[target_name].values)
        data = add_dist_label(data, change_list)
        return data

    return transform


def get_dataset(config: Config, input_path: Path):
    feature_engineering = get_features(config)
    dd_transformer = dd_transform(config, config.dataset.target)
    feature_engineering = add_transform_to_feature_engineering(
        feature_engineering,
        dd_transformer,
        pos=DD_ID,
        sections=["preprocess"],
    )

    match config.dataset.name:
        case Dataset.ALIBABA:
            from src.dataset.alibaba.container_seventeen import (
                AlibabaContainerDataAccessor,
                AlibabaContainerDataset,
                AlibabaContainerDatasetPrototype,
            )

            generator: DistributionColumnBasedGenerator[
                AlibabaContainerDataset, AlibabaContainerDataAccessor
            ] = DistributionColumnBasedGenerator(
                data=input_path,
                prototype=AlibabaContainerDatasetPrototype(),
                target=feature_engineering.target_name,
                dist_col=DD_DIST_COLUMN,
                feature_engineering=feature_engineering,
                train_ratio=config.train_ratio,
            )
        case Dataset.AZURE:
            from src.dataset.azure.vmcpu import (
                AzureVMDataAccessor,
                AzureVMDataset,
                AzureVMDatasetPrototype,
            )

            generator: DistributionColumnBasedGenerator[
                AzureVMDataset, AzureVMDataAccessor
            ] = DistributionColumnBasedGenerator(
                data=input_path,
                prototype=AzureVMDatasetPrototype(),
                target=feature_engineering.target_name,
                dist_col=DD_DIST_COLUMN,
                feature_engineering=feature_engineering,
                train_ratio=config.train_ratio,
            )
        case Dataset.GOOGLE:
            from src.dataset.google.scheduler2 import (
                GoogleSchedulerDataAccessor,
                GoogleSchedulerDataset,
                GoogleSchedulerDatasetPrototype,
            )

            generator: DistributionColumnBasedGenerator[
                GoogleSchedulerDataset, GoogleSchedulerDataAccessor
            ] = DistributionColumnBasedGenerator(
                data=input_path,
                prototype=GoogleSchedulerDatasetPrototype(),
                target=feature_engineering.target_name,
                dist_col=DD_DIST_COLUMN,
                feature_engineering=feature_engineering,
                train_ratio=config.train_ratio,
            )
        case Dataset.GOOGLE:
            from src.dataset.google.scheduler2 import (
                GoogleSchedulerDatasetDistChunkGenerator,
            )

            generator = GoogleSchedulerDatasetDistChunkGenerator(
                file=input_path,
                target=data_transformer.target_name,
                n_labels=config.num_classes,
                transform=data_transformer,
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


def assert_training(config: Config):
    assert (
        config.drift_detection is not None
    ), "Drift detection must be specified in config"


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
