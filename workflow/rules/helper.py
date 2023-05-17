from pathlib import Path


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


__all__ = ["get_dataset_files", "get_database_config", "get_datasets"]
