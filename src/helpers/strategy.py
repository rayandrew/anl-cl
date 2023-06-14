from .config import Config
from .definitions import Strategy


def _get_strategy(strategy: Strategy):
    match strategy:
        case Strategy.NO_RETRAIN:
            return None
        case Strategy.FROM_SCRATCH:
            from avalanche.training.supervised import (
                FromScratchTraining,
            ) 
            return FromScratchTraining
        case Strategy.NAIVE:
            from avalanche.training.supervised import Naive
            return Naive
        case Strategy.GSS:
            from avalanche.training.supervised import GSS_greedy
            return GSS_greedy
        case Strategy.AGEM:
            from avalanche.training.supervised import AGEM
            return AGEM
        case Strategy.GEM:
            from avalanche.training.supervised import GEM
            return GEM
        case Strategy.EWC:
            from avalanche.training.supervised import EWC
            return EWC
        case Strategy.MAS:
            from avalanche.training.supervised import MAS
            return MAS
        case Strategy.SI:
            from avalanche.training.supervised import SI
            return SI
        case Strategy.LWF:
            from avalanche.training.supervised import LwF
            return LwF
        case _:
            raise ValueError("Unknown strategy")

def get_strategy(cfg: Config):
    return _get_strategy(cfg.strategy.name)


__all__ = [
    "_get_strategy",
    "get_strategy",
]
