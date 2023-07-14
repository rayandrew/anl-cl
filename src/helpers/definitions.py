from abc import ABCMeta
from typing import Any, Dict

from snakemake.io import InputFiles, Log, OutputFiles, Params, Wildcards

from src.utils.ds import StrEnum


class Dataset(StrEnum):
    ALIBABA = "alibaba"
    GOOGLE = "google"
    CORI = "cori"
    AZURE = "azure"


class Optimizer(StrEnum):
    ADAM = "adam"
    SGD = "sgd"


class Scenario(StrEnum):
    SPLIT_CHUNKS = "split-chunks"
    DRIFT_DETECTION = "drift-detection"
    NUM_ROWS = "num-rows"
    TIME = "time"


class Strategy(StrEnum):
    NO_RETRAIN = "no-retrain"
    FROM_SCRATCH = "from-scratch"
    NAIVE = "naive"
    GSS = "gss"
    AGEM = "agem"
    GEM = "gem"
    EWC = "ewc"
    MAS = "mas"
    SI = "si"
    LWF = "lwf"
    GDUMB = "gdumb"


class Task(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Model(StrEnum):
    A = "model-a"
    B = "model-b"
    # MLP = "mlp"


class Training(StrEnum):
    ONLINE = "online"
    BATCH = "batch"


class DriftDetector(StrEnum):
    VOTING = "voting"
    RUPTURES = "ruptures"
    ONLINE = "online"


class Snakemake(metaclass=ABCMeta):
    input: InputFiles
    output: OutputFiles
    log: Log
    params: Params
    wildcards: Wildcards
    config: Dict[str, Any]


__all__ = [
    "StrEnum",
    "Optimizer",
    "Dataset",
    "Scenario",
    "Strategy",
    "Task",
    "Model",
    "Training",
    "DriftDetector",
    "Snakemake",
]
