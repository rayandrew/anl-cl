from collections import defaultdict
from pathlib import Path
from typing import List

DATASETS = ["alibaba"]


def get_datasets(config: dict):
    return [
        dataset for dataset in config["dataset"] if dataset != "cori"
    ]


def get_database_config(config):
    return lambda wildcards: config["dataset"][wildcards.dataset]


def get_dataset_files(config: dict, dataset: str):
    for file in Path(config["dataset"][dataset]["path"]).glob(
        "**/*.csv"
    ):
        yield file

    for file in Path(config["dataset"][dataset]["path"]).glob(
        "**/*.parquet"
    ):
        yield file


def get_all_dataset_files(
    config: dict, datasets: List[str] = DATASETS
):
    result = defaultdict(list)
    for dataset in datasets:
        for file in get_dataset_files(config, dataset):
            result[dataset].append(file.stem)
        # for file in Path(config["dataset"][dataset]["path"]).glob(
        #     "**/*.csv"
        # ):
        #     result[dataset].append(file.stem)
        # for file in Path(config["dataset"][dataset]["path"]).glob(
        #     "**/*.parquet"
        # ):
        #     result[dataset].append(file.stem)
    return result


__all__ = [
    "get_dataset_files",
    "get_database_config",
    "get_datasets",
    "get_dataset_files",
]
