from enum import Enum
from typing import Optional


class Scenario(Enum):
    SPLIT_CHUNKS = "split_chunks"
    DRIFT_DETECTION = "drift_detection"


class Strategy(Enum):
    NO_RETRAIN = "no_retrain"
    FROM_SCRATCH = "from_scratch"
    NAIVE = "naive"
    GSS_GREEDY = "gss_greedy"
    AGEM = "agem"
    GEM = "gem"
    EWC = "ewc"
    MAS = "mas"
    SI = "si"
    LWF = "lwf"
