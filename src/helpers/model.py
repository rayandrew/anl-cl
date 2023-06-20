import torch.nn as nn

from src.models.mlp import MLP

from .config import Config
from .definitions import Model


class Model_A(MLP):
    pass

class Model_B(MLP):
    def __init__(self, num_classes=10, input_size=28 * 28, hidden_size=512, hidden_layers=1, drop_rate=0.5):
        super().__init__(num_classes, input_size, hidden_size, hidden_layers, drop_rate)
        self.features = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_size),
                # nn.BatchNorm1d(hidden_size),
                nn.InstanceNorm1d(hidden_size),
                # nn.BatchNorm2d(hidden_size),
                *list(self.features.children())[1:],
            )
        )



def _get_model(model: Model):
    match model:
        case Model.A:
            return Model_A
        case Model.B:
            return Model_B
        case _:
            raise ValueError("Unknown model")
        


def get_model(cfg: Config):
    return _get_model(cfg.model.name)

__all__ = [
    "_get_model",
    "get_model"
]
