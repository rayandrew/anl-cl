from abc import ABC, abstractmethod
from typing import Dict, Protocol


class Trainer(Protocol):
    def train(self) -> Dict[int, Dict[str, float]]:
        pass


class BaseTrainer(ABC):
    @abstractmethod
    def train(self) -> Dict[int, Dict[str, float]]:
        pass


__all__ = [
    "BaseTrainer",
]
