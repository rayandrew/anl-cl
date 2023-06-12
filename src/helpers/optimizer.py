from enum import Enum
from typing import Optional

import torch


class Optimizer(Enum):
    ADAM = "adam"
    SGD = "sgd"

def get_optimizer(model: torch.nn.Module, name: Optional[Optimizer] = None, **kwargs):
    if name is None:
        from torch.optim import Adam

        optimizer = Adam(model.parameters(), lr=0.001, **kwargs)
    else:
        match name:
            case Optimizer.ADAM:
                from torch.optim import Adam
                optimizer = Adam(model.parameters(), **kwargs)
            case Optimizer.SGD:
                from torch.optim import SGD
                optimizer = SGD(model.parameters(), **kwargs)
            case _:
                raise ValueError("Unknown optimizer")

    return optimizer

def get_optimizer_from_config(cfg: dict, model: torch.nn.Module):
    cfg = cfg.copy()
    optim = Optimizer.ADAM
    if "name" in cfg:
        optim = Optimizer(cfg["name"].upper())

    optimizer = get_optimizer(model, optim, **cfg)
    return optimizer
    
__all__ = ["Optimizer", "get_optimizer", "get_optimizer_from_config"]
