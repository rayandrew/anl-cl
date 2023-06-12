from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DatasetConfig:
    path: str
    y: str


@dataclass
class ScenarioSplitChunksConfig:
    num_split: int = 4


@dataclass
class ScenarioDriftDetectionConfig:
    pass


@dataclass
class ScenarioConfig:
    split_chunks: ScenarioSplitChunksConfig
    drift_detection: ScenarioDriftDetectionConfig


@dataclass
class ModelConfig:
    hidden_size: int = 2
    hidden_layers: int = 512
    drop_rate: float = 0.0


@dataclass
class OptimizerConfig:
    lr: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.0


@dataclass
class ClassificationConfig:
    num_classes: int = 4


@dataclass
class OfflineTrainingConfig:
    batch_size: int = 32
    epochs: int = 4
    lr: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.0
    patience: int = 10
    num_workers: int = 0
    classification: dict = None
    regression: dict = None


@dataclass
class TrainingConfig:
    offline: OfflineTrainingConfig


@dataclass
class Config:
    seed: int = 0
    base_path: Path | str
    scenario_config: ScenarioConfig
    training_config: TrainingConfig
    dataset_config: DatasetConfig
    model_config: ModelConfig
