from typing import Dict

from avalanche.benchmarks.scenarios import GenericCLScenario
from avalanche.training.templates import SupervisedTemplate

from src.trainers.base import BaseTrainer
from src.utils.logging import logging

log = logging.getLogger(__name__)


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
        len(self.benchmark.train_stream)
        assert (
            len(self.benchmark.train_stream) > 1
        ), "BatchNoRetrainTrainer requires at least 2 experiences"

        log.info("Training on first experience only")
        first_experience = self.benchmark.train_stream[0]
        results = {}
        self.strategy.train(
            first_experience,
            num_workers=self.num_workers,
        )
        log.info("Evaluation on all experiences")
        result = self.strategy.eval(self.benchmark.test_stream)
        results[0] = result
        return results


class BatchSimpleRetrainTrainer(BatchNoRetrainTrainer):
    def train(self) -> Dict[int, Dict[str, float]]:
        results = {}
        for experience in self.benchmark.train_stream:
            log.info(
                "Training on experience: %s", experience.current_experience
            )
            self.strategy.train(
                experience,
                num_workers=self.num_workers,
            )
            log.info(
                "Evaluating model after experience: %s",
                experience.current_experience,
            )
            result = self.strategy.eval(self.benchmark.test_stream)
            results[experience.current_experience] = result
        return results


__all__ = [
    "BatchNoRetrainTrainer",
    "BatchSimpleRetrainTrainer",
]
