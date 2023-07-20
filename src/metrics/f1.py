# flake8: noqa: E501
from typing import List

import torchmetrics as tm

from .pytorch_metrics import AverageType, PyTorchMetricPluginMetric, TaskType


class EpochF1(PyTorchMetricPluginMetric):
    def __init__(
        self,
        task: TaskType = "multiclass",
        num_classes: int = 10,
        average: AverageType = "macro",
    ):
        """
        Creates an instance of F1 metric
        """
        metric = tm.F1Score(
            task=task,
            num_classes=num_classes,
            average=average,
        )
        super(EpochF1, self).__init__(
            metric,
            reset_at="epoch",
            emit_at="epoch",
            mode="train",
        )

    def __str__(self):
        return "Top1_F1_Epoch"


class ExperienceF1(PyTorchMetricPluginMetric):
    def __init__(
        self,
        task: TaskType = "multiclass",
        num_classes: int = 10,
        average: AverageType = "macro",
    ):
        """
        Creates an instance of F1 metric
        """
        metric = tm.F1Score(
            task=task,
            num_classes=num_classes,
            average=average,
        )
        super(ExperienceF1, self).__init__(
            metric,
            reset_at="experience",
            emit_at="experience",
            mode="eval",
        )

    def __str__(self):
        return "Top1_F1_Exp"


class StreamF1(PyTorchMetricPluginMetric):
    def __init__(
        self,
        task: TaskType = "multiclass",
        num_classes: int = 10,
        average: AverageType = "macro",
    ):
        """
        Creates an instance of F1 metric
        """
        metric = tm.F1Score(
            task=task,
            num_classes=num_classes,
            average=average,
        )
        super(StreamF1, self).__init__(
            metric,
            reset_at="stream",
            emit_at="stream",
            mode="eval",
        )

    def __str__(self):
        return "Top1_F1_Stream"


def f1_metrics(
    *,
    task: TaskType = "multiclass",
    num_classes: int = 10,
    average: AverageType = "macro",
    epoch: bool = False,
    experience: bool = False,
    stream: bool = False,
) -> List[PyTorchMetricPluginMetric]:
    """
    Helper function to create F1 metrics

    :param task: Task type. It can be "binary" or "multiclass" or "multilabel"
    :type task: TaskType
    :param num_classes: Number of classes
    :type num_classes: int
    :param average: Type of averaging to use. It can be "macro", "micro" or "weighted" or None
    :type average: AverageType
    :param epoch: If True, it computes metric at epoch level, defaults to False
    :type epoch: bool, optional
    :param experience: If True, it computes metric at experience level, defaults to False
    :type experience: bool, optional
    :param stream: If True, it computes metric at stream level, defaults to False
    :type stream: bool, optional
    :return: List of metrics
    :rtype: List[PyTorchMetricPluginMetric]
    """

    metrics: List[PyTorchMetricPluginMetric] = []
    if epoch:
        metrics.append(
            EpochF1(
                task=task,
                num_classes=num_classes,
                average=average,
            )
        )

    if experience:
        metrics.append(
            ExperienceF1(
                task=task,
                num_classes=num_classes,
                average=average,
            )
        )

    if stream:
        metrics.append(
            StreamF1(
                task=task,
                num_classes=num_classes,
                average=average,
            )
        )

    return metrics


__all__ = ["EpochF1", "ExperienceF1", "StreamF1", "f1_metrics"]
