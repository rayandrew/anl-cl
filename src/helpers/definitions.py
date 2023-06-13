from enum import Enum


class Dataset(Enum):
    ALIBABA = "alibaba"
    GOOGLE = "google"
    # CORI = "cori"
    # AZURE = "azure"


class Optimizer(Enum):
    ADAM = "adam"
    SGD = "sgd"


class Scenario(Enum):
    SPLIT_CHUNKS = "split-chunks"
    DRIFT_DETECTION = "drift-detection"


class Strategy(Enum):
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


class Task(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Model(Enum):
    MLP = "mlp"


__all__ = [
    "Optimizer",
    "Dataset",
    "Scenario",
    "Strategy",
    "Task",
    "Model",
]
