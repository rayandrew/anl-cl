from enum import Enum


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class Dataset(StrEnum):
    ALIBABA = "alibaba"
    GOOGLE = "google"
    # CORI = "cori"
    # AZURE = "azure"


class Optimizer(StrEnum):
    ADAM = "adam"
    SGD = "sgd"


class Scenario(StrEnum):
    SPLIT_CHUNKS = "split-chunks"
    DRIFT_DETECTION = "drift-detection"


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
    MLP = "mlp"


class Training(StrEnum):
    ONLINE = "online"
    BATCH = "batch"


__all__ = [
    "StrEnum",
    "Optimizer",
    "Dataset",
    "Scenario",
    "Strategy",
    "Task",
    "Model",
    "Training",
]
