from collections import defaultdict
from pathlib import Path
from typing import List

DATASETS = ["alibaba", "google", "cori"]
EXTENSIONS = ["csv", "parquet"]
SCENARIOS = ["split-chunks"]
STRATEGIES = [
    "no-retrain",
    "from-scratch",
    "naive",
    "gss",
    "gdumb",
    "agem",
    "gem",
    "ewc",
]
TRAININGS = ["online", "offline"]
TASKS = [
    "classification",
    #  "regression"
]
# PIPELINE_FLOWS = list(
#     itertools.product(SCENARIOS, STRATEGIES, TRAININGS, TASKS)
# )


def get_dataset_files(
    base_path: Path | str, dataset: str, remove_ext: bool = False
):
    base_path = Path(base_path)
    files: List[Path] = []
    for ext in EXTENSIONS:
        for file in (base_path / dataset).glob(f"**/*.{ext}"):
            if file.is_file():
                files.append(
                    file.with_suffix("") if remove_ext else file
                )
    return files


def get_all_dataset_files_as_dict(
    base_path: Path | str,
    datasets: List[str] = DATASETS,
    return_stem: bool = False,
    remove_ext: bool = False,
):
    result = defaultdict(list)
    for dataset in datasets:
        for file in get_dataset_files(base_path, dataset, remove_ext):
            result[dataset].append(file.stem if return_stem else file)
    return result


def get_all_dataset_files(
    base_path: Path | str,
    datasets: List[str] = DATASETS,
    return_stem: bool = False,
    remove_ext: bool = False,
):
    result = []
    for dataset in datasets:
        result += [
            file.stem if return_stem else file
            for file in get_dataset_files(
                base_path, dataset, remove_ext
            )
        ]
    return result


__all__ = [
    "DATASETS",
    "EXTENSIONS",
    "SCENARIOS",
    "STRATEGIES",
    "TRAININGS",
    "TASKS",
    "get_datasets",
    "get_all_dataset_files",
    "get_all_dataset_files_as_dict",
]
