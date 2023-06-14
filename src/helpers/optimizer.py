from enum import Enum
from typing import Optional

from .config import Config
from .definitions import Optimizer

# def get_optimizer(model: torch.nn.Module, name: Optional[Optimizer] = None, **kwargs):
#     if name is None:
#         from torch.optim import Adam

#         optimizer = Adam(model.parameters(), lr=0.001, **kwargs)
#     else:
#         match name:
#             case Optimizer.ADAM:
#                 from torch.optim import Adam
#                 optimizer = Adam(model.parameters(), **kwargs)
#             case Optimizer.SGD:
#                 from torch.optim import SGD
#                 optimizer = SGD(model.parameters(), **kwargs)
#             case _:
#                 raise ValueError("Unknown optimizer")

#     return optimizer


def _get_optimizer(optimizer: Optimizer):
    if optimizer == Optimizer.ADAM:
        from torch.optim import Adam

        return Adam

    if optimizer == Optimizer.SGD:
        from torch.optim import SGD

        return SGD

    raise ValueError("Unknown optimizer")


def get_optimizer(config: Config):
    Optimizer = _get_optimizer(config.optimizer)
    return Optimizer


__all__ = ["Optimizer", "get_optimizer"]
