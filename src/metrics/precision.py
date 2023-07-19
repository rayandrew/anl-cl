# flake8: noqa: E501
from typing import List

import torchmetrics as tm

from .pytorch_metrics import AverageType, PyTorchMetricPluginMetric, TaskType


class EpochPrecision(PyTorchMetricPluginMetric):
    def __init__(
        self,
        task: TaskType = "multiclass",
        num_classes: int = 10,
        average: AverageType = "macro",
    ):
        """
        Creates an instance of Accuracy metric
        """
        metric = tm.Precision(
            task=task,
            num_classes=num_classes,
            average=average,
        )
        super(EpochPrecision, self).__init__(
            metric,
            reset_at="epoch",
            emit_at="epoch",
            mode="train",
        )

    def __str__(self):
        return "Top1_Precision_Epoch"


class ExperiencePrecision(PyTorchMetricPluginMetric):
    def __init__(
        self,
        task: TaskType = "multiclass",
        num_classes: int = 10,
        average: AverageType = "macro",
    ):
        """
        Creates an instance of Accuracy metric
        """
        metric = tm.Precision(
            task=task,
            num_classes=num_classes,
            average=average,
        )
        super(ExperiencePrecision, self).__init__(
            metric,
            reset_at="experience",
            emit_at="experience",
            mode="eval",
        )

    def __str__(self):
        return "Top1_Precision_Experience"


class StreamPrecision(PyTorchMetricPluginMetric):
    def __init__(
        self,
        task: TaskType = "multiclass",
        num_classes: int = 10,
        average: AverageType = "macro",
    ):
        """
        Creates an instance of Accuracy metric
        """
        metric = tm.Precision(
            task=task,
            num_classes=num_classes,
            average=average,
        )
        super(StreamPrecision, self).__init__(
            metric,
            reset_at="stream",
            emit_at="stream",
            mode="eval",
        )

    def __str__(self):
        return "Top1_Precision_Stream"


def precision_metrics(
    *,
    task: TaskType = "multiclass",
    num_classes: int = 10,
    average: AverageType = "macro",
    epoch: bool = False,
    experience: bool = False,
    stream: bool = False,
) -> List[PyTorchMetricPluginMetric]:
    """
    Helper function to create Precision metrics

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
            EpochPrecision(
                task=task,
                num_classes=num_classes,
                average=average,
            )
        )

    if experience:
        metrics.append(
            ExperiencePrecision(
                task=task,
                num_classes=num_classes,
                average=average,
            )
        )

    if stream:
        metrics.append(
            StreamPrecision(
                task=task,
                num_classes=num_classes,
                average=average,
            )
        )

    return metrics


__all__ = [
    "EpochPrecision",
    "ExperiencePrecision",
    "StreamPrecision",
    "precision_metrics",
]
