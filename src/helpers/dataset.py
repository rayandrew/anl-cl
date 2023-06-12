from enum import Enum


class Dataset(Enum):
    ALIBABA = "alibaba"
    GOOGLE = "google"

def get_splitted_dataset(dataset: Dataset):
    match dataset:
        case Dataset.ALIBABA:
            from src.dataset.alibaba import (
                get_classification_alibaba_scheduler_dataset_splitted as DatasetFactory,  # get_classification_alibaba_machine_dataset_splitted as Dataset,
            )
        case Dataset.GOOGLE:
            # TODO: @william change this
            from src.dataset.google import (
                get_classification_google_machine_dataset_splitted as DatasetFactory,
            )
        case _:
            raise ValueError("Unknown dataset")
    return DatasetFactory

def get_dataset(dataset: Dataset):
    match dataset:
        case Dataset.ALIBABA:
            from src.dataset.alibaba import (
                get_classification_alibaba_scheduler_dataset as DatasetFactory,
            )
        case Dataset.GOOGLE:
            # TODO: @william change this to get_classification_google_scheduler_dataset
            from src.dataset.google import (
                get_classification_google_machine_dataset as DatasetFactory,
            )
        case _:
            raise ValueError("Unknown dataset")
    return DatasetFactory


__all__ = ["Dataset", "get_dataset", "get_splitted_dataset"]
