from enum import Enum


class InferenceType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


def get_num_classes_from_config(cfg: dict):
    return cfg["classification"]["num_classes"]


__all__ = [
    "InferenceType",
]
