from abc import ABCMeta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Extra, field_validator

from .definitions import (
    Dataset,
    DriftDetector,
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
    target: str
    feature: str
    num_classes: int
    time_column: str

    @field_validator("feature")
    def check_feature_prefix(cls, v: str):
        if v == "no-feature":
            return v
        if not v.startswith("feature-"):
            raise ValueError("Feature must start with 'feature-'")
        return v


class StrategyConfig(DynamicConfig):
    name: Strategy


class DriftDetectionConfig(DynamicConfig):
    name: DriftDetector


class TuneConfig(DynamicConfig):
    learning_rate: list[float] = [1e-3]


class GeneralConfig(DynamicConfig):
    model_name: str = "model"
    base_path: str | Path
    seed: int = 0
    num_workers: int = 4
    epochs: int = 4
    eval_tol: int = 0
    optimizer: Optimizer = Optimizer.SGD
    batch: int = 32
    test_batch: int = 32
    wandb: str | None = None
    online: bool = False
    train_ratio: float = 0.8


class Config(GeneralConfig):
    model: ModelConfig
    scenario: ScenarioConfig
    dataset: DatasetConfig
    strategy: StrategyConfig
    tune: TuneConfig = TuneConfig()
    drift_detection: DriftDetectionConfig | None = None


class DDOnlyConfig(GeneralConfig):
    dataset: DatasetConfig
    drift_detection: DriftDetectionConfig


def assert_config_params(config: Config, params: Any):
    assert Dataset(params.dataset) == config.dataset.name, (
        f"Dataset mismatch: got {params.dataset} instead of "
        f"{config.dataset.name}"
    )
    assert Scenario(params.scenario) == config.scenario.name, (
        f"Scenario mismatch: got {params.scenario} instead of "
        f"{config.scenario.name}"
    )
    assert (
        Model(params.model) == config.model.name
    ), f"Model mismatch: got {params.model} instead of {config.model.name}"
    assert params.feature == config.dataset.feature, (
        f"Feature mismatch: got {params.feature} instead of "
        f"{config.dataset.feature}"
    )
    if config.online:
        assert params.training == Training.ONLINE, (
            f"Training mismatch: got {params.training} instead of "
            f"{Training.ONLINE}"
        )
    else:
        assert params.training == Training.BATCH, (
            f"Training mismatch: Got {params.training} instead of "
            f"{Training.BATCH}"
        )


__all__ = [
    "DynamicConfig",
    "Config",
    "DDOnlyConfig",
    "GeneralConfig",
    "ModelConfig",
    "ScenarioConfig",
    "DatasetConfig",
    "StrategyConfig",
    "TuneConfig",
    "assert_config_params",
]
