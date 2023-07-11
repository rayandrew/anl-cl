from typing import Any, Dict

from avalanche.benchmarks.scenarios import GenericCLScenario
from avalanche.training.templates import SupervisedTemplate

from src.trainers.base import BaseTrainer


class BatchNoRetrainTrainer(BaseTrainer):
    def __init__(
        self,
        strategy: SupervisedTemplate,
        benchmark: GenericCLScenario,
        num_workers: int = 4,
    ):
        self.strategy = strategy
        self.benchmark = benchmark
        self.num_workers = num_workers

    def train(self) -> Dict[int, Dict[str, float]]:
        assert (
            len(self.benchmark.train_stream) > 1
        ), "BatchNoRetrainTrainer requires at least 2 experiences"

        first_experience = self.benchmark.train_stream[0]
        results = {}
        self.strategy.train(
            first_experience,
            num_workers=self.num_workers,
        )
        result = self.strategy.eval(self.benchmark.test_stream)
        results[0] = result
        return results


class BatchSimpleRetrainTrainer(BatchNoRetrainTrainer):
    def train(self) -> Dict[int, Dict[str, float]]:
        results = {}
        for experience in self.benchmark.train_stream:
            self.strategy.train(
                experience,
                num_workers=self.num_workers,
            )
            result = self.strategy.eval(self.benchmark.test_stream)
            results[experience.current_experience] = result
        return results


__all__ = [
    "BatchNoRetrainTrainer",
    "BatchSimpleRetrainTrainer",
]
