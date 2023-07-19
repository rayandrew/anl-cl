# flake8: noqa: E501
from typing import List

import torchmetrics as tm

from .pytorch_metrics import AverageType, PyTorchMetricPluginMetric, TaskType


class EpochAUROC(PyTorchMetricPluginMetric):
    def __init__(
        self,
        task: TaskType = "multiclass",
        num_classes: int = 10,
        average: AverageType = "macro",
    ):
        metric = tm.AUROC(
            task=task,
            num_classes=num_classes,
            average=average,
        )
        super(EpochAUROC, self).__init__(
            metric,
            reset_at="epoch",
            emit_at="epoch",
            mode="train",
        )

    def __str__(self):
        return "Top1_AUROC_Epoch"


class ExperienceAUROC(PyTorchMetricPluginMetric):
    def __init__(
        self,
        task: TaskType = "multiclass",
        num_classes: int = 10,
        average: AverageType = "macro",
    ):
        metric = tm.AUROC(
            task=task,
            num_classes=num_classes,
            average=average,
        )
        super(ExperienceAUROC, self).__init__(
            metric,
            reset_at="experience",
            emit_at="experience",
            mode="eval",
        )

    def __str__(self):
        return "Top1_AUROC_Experience"


class StreamAUROC(PyTorchMetricPluginMetric):
    def __init__(
        self,
        task: TaskType = "multiclass",
        num_classes: int = 10,
        average: AverageType = "macro",
    ):
        metric = tm.AUROC(
            task=task,
            num_classes=num_classes,
            average=average,
        )
        super(StreamAUROC, self).__init__(
            metric,
            reset_at="stream",
            emit_at="stream",
            mode="eval",
        )

    def __str__(self):
        return "Top1_AUROC_Stream"


def auroc_metrics(
    *,
    task: TaskType = "multiclass",
    num_classes: int = 10,
    average: AverageType = "macro",
    epoch: bool = False,
    experience: bool = False,
    stream: bool = False,
) -> List[PyTorchMetricPluginMetric]:
    """
    Helper function to create AUROC metrics

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
            EpochAUROC(
                task=task,
                num_classes=num_classes,
                average=average,
            )
        )

    if experience:
        metrics.append(
            ExperienceAUROC(
                task=task,
                num_classes=num_classes,
                average=average,
            )
        )

    if stream:
        metrics.append(
            StreamAUROC(
                task=task,
                num_classes=num_classes,
                average=average,
            )
        )

    return metrics


__all__ = ["EpochAUROC", "ExperienceAUROC", "StreamAUROC", "auroc_metrics"]
