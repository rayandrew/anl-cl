from abc import ABC, abstractmethod
from typing import Dict


class BaseTrainer(ABC):
    @abstractmethod
    def train(self) -> Dict[int, Dict[str, float]]:
        pass


__all__ = [
    "BaseTrainer",
]
