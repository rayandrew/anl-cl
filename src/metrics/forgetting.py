from typing import Dict, List, Union

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics.forgetting_bwt import (
    ExperienceForgetting,
    StreamForgetting,
    forgetting_to_bwt,
)

from .accuracy import TaskAwareAccuracyWithTolerance


class ExperienceForgettingWithTolerance(ExperienceForgetting):
    def __init__(self, tolerance: int = 0):
        """
        Creates an instance of the ExperienceForgettingWithTolerance metric.
        """

        super(ExperienceForgetting, self).__init__()

        self._current_metric = TaskAwareAccuracyWithTolerance(
            tolerance
        )

    def __str__(self):
        return "ExperienceForgetting_Tol"


class StreamForgettingWithTolerance(StreamForgetting):
    def __init__(self, tolerance: int = 0):
        """
        Creates an instance of the StreamForgettingWithTolerance metric.
        """
        super(StreamForgetting, self).__init__()

        self._current_metric = TaskAwareAccuracyWithTolerance(
            tolerance
        )

    def __str__(self):
        return "StreamForgetting_Tol"


def forgetting_metrics_with_tolerance(
    *, tolerance: int = 0, experience=False, stream=False
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param tolerance: The tolerance to use for the accuracy metric.
    :param experience: If True, will return a metric able to log
        the forgetting on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the forgetting averaged over the evaluation stream experiences,
        which have been observed during training.
    :return: A list of plugin metrics.
    """

    metrics = []

    if experience:
        metrics.append(
            ExperienceForgettingWithTolerance(tolerance=tolerance)
        )

    if stream:
        metrics.append(
            StreamForgettingWithTolerance(tolerance=tolerance)
        )

    return metrics


class ExperienceBWTWithTolerance(ExperienceForgettingWithTolerance):
    def result(self, k=None) -> float | None | Dict[int, float]:
        """
        See `Forgetting` documentation for more detailed information.
        k: optional key from which compute forgetting.
        """
        forgetting = super().result(k)
        return forgetting_to_bwt(forgetting)

    def __str__(self):
        return "ExperienceBWT_Tol"


class StreamBWTWithTolerance(StreamForgettingWithTolerance):
    """
    The StreamBWTWithTolerance metric, emitting the average BWT across all experiences
    encountered during training.
    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the last accuracy result
    obtained on an experience and the accuracy result obtained when first
    training on that experience.
    This metric is computed during the eval phase only.
    """

    def exp_result(self, k=None) -> float | None | Dict[int, float]:
        """
        Result for experience defined by a key.
        See `BWT` documentation for more detailed information.
        k: optional key from which compute backward transfer.
        """
        forgetting = super().exp_result(k)
        return forgetting_to_bwt(forgetting)

    def __str__(self):
        return "StreamBWT_Tol"


def bwt_metrics_with_tolerance(
    *, tolerance: int = 0, experience=False, stream=False
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.
    :param tolerance: The tolerance to use for the accuracy metric.
    :param experience: If True, will return a metric able to log
        the backward transfer on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the backward transfer averaged over the evaluation stream experiences
        which have been observed during training.
    :return: A list of plugin metrics.
    """

    metrics = []

    if experience:
        metrics.append(
            ExperienceBWTWithTolerance(tolerance=tolerance)
        )

    if stream:
        metrics.append(StreamBWTWithTolerance(tolerance=tolerance))

    return metrics


__all__ = [
    "ExperienceForgettingWithTolerance",
    "StreamForgettingWithTolerance",
    "forgetting_metrics_with_tolerance",
    "bwt_metrics_with_tolerance",
]
