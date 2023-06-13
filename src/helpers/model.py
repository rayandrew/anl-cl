from .config import Config
from .definitions import Model


def _get_model(model: Model):
    match model:
        case Model.MLP:
            from src.models.mlp import MLP

            return MLP
        case _:
            raise ValueError("Unknown model")

def get_model(cfg: Config):
    return _get_model(cfg.model.name)

__all__ = [
    "_get_model",
    "get_model"
]
