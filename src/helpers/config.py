# from types import SimpleNamespace
from abc import ABCMeta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Extra

from .definitions import (
    Dataset,
    Model,
    Optimizer,
    Scenario,
    Strategy,
    Training,
)


class DynamicConfig(BaseModel, metaclass=ABCMeta, extra=Extra.allow):
    pass


class ModelConfig(DynamicConfig):
    name: Model
    hidden_layers: int = 3
    hidden_size: int = 512
    drop_rate: float = 0.2


class ScenarioConfig(DynamicConfig):
    name: Scenario


class DatasetConfig(DynamicConfig):
    name: Dataset
    y: str


class StrategyConfig(DynamicConfig):
    name: Strategy


class TuneConfig(DynamicConfig):
    learning_rate: list[float] = [1e-3]


class GeneralConfig(DynamicConfig):
    model_name: str = "model"
    base_path: str | Path
    seed: int = 0
    num_workers: int = 4
    epochs: int = 4
    num_classes: int = 4
    eval_tol: int = 0
    optimizer: Optimizer = Optimizer.SGD
    batch: int = 32
    test_batch: int = 32
    wandb: str | None = None
    online: bool = False


class Config(GeneralConfig):
    model: ModelConfig
    scenario: ScenarioConfig
    dataset: DatasetConfig
    strategy: StrategyConfig
    tune: TuneConfig = TuneConfig()


def assert_config_params(config: Config, params: Any):
    assert Dataset(params.dataset) == config.dataset.name
    assert Scenario(params.scenario) == config.scenario.name
    if config.online:
        assert params.training == Training.ONLINE


__all__ = [
    "DynamicConfig",
    "Config",
    "GeneralConfig",
    "ModelConfig",
    "ScenarioConfig",
    "DatasetConfig",
    "StrategyConfig",
    "TuneConfig",
    "assert_config_params",
]
