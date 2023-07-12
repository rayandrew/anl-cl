from typing import TYPE_CHECKING, List, Literal, Union

import torch

from avalanche.evaluation import GenericPluginMetric, Metric, PluginMetric
from avalanche.evaluation.metrics.mean import Mean

import torchmetrics as tm

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

from src.utils.logging import logging

# all of these metrics are for multiclass classification

log = logging.getLogger(__name__)

TAverage = Literal["macro", "weighted", "none"]
# TResult = List[float] | float
TResult = float


def assert_classification_metric(
    predicted_y: torch.Tensor,
    true_y: torch.Tensor,
    task_labels: Union[int, torch.Tensor],
) -> None:
    if len(true_y) != len(predicted_y):
        raise ValueError("Size mismatch for true_y and predicted_y tensors")

    if isinstance(task_labels, torch.Tensor) and len(task_labels) != len(
        true_y
    ):
        raise ValueError("Size mismatch for true_y and task_labels tensors")

    # if not isinstance(task_labels, (int, torch.Tensor)):
    #     raise ValueError(
    #         f"Task label type: {type(task_labels)}, "
    #         f"expected int or Tensor"
    #     )


def convert_tensor_to_float_or_list(
    tensor: torch.Tensor,
) -> Union[float, List[float]]:
    if isinstance(tensor, torch.Tensor):
        if tensor.numel() == 1:
            return tensor.item()  # Convert single-element tensor to float
        else:
            return (
                tensor.tolist()
            )  # Convert multi-element tensor to list of floats
    else:
        raise ValueError("Input is not a PyTorch tensor.")


class BaseClassificationMetric(Metric[TResult]):
    """
    Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    for multiclass classification.
    """

    def __init__(
        self,
        metric: tm.Metric,
    ) -> None:
        self._metric = metric
        self._counter = 0
        self._result: TResult = 0.0
        self._mean = Mean()

    @torch.no_grad()
    def update(
        self,
        predicted_y: torch.Tensor,
        true_y: torch.Tensor,
        task_labels: Union[int, torch.Tensor],
    ) -> None:
        assert_classification_metric(predicted_y, true_y, task_labels)
        self._counter += 1
        result: torch.Tensor = self._metric(predicted_y, true_y)
        result = result.cpu().detach()
        # log.info("Metric: %s", self._metric)
        # log.info(
        #     "Metric: %s, Counter: %d", self._metric, self._counter
        # )
        # log.info("result: %s", result)
        result_mod = convert_tensor_to_float_or_list(result)
        if isinstance(result_mod, float):
            self._mean.update(result_mod)
        else:
            for r in result_mod:
                self._mean.update(r)
        # log.info("_result: %s", result)

    def result(self) -> TResult:
        return self._mean.result()

    def reset(self) -> None:
        # log.info("Resetting metric: %s", self._metric)
        self._mean.reset()
        # self._result: TResult = 0.0
        self._counter = 0


class ClassificationPluginMetric(
    GenericPluginMetric[TResult, BaseClassificationMetric]
):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, metric, reset_at, emit_at, mode):
        """Creates the Accuracy plugin

        :param metric: the metric to compute
        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super(ClassificationPluginMetric, self).__init__(
            metric=BaseClassificationMetric(metric),
            reset_at=reset_at,
            emit_at=emit_at,
            mode=mode,
        )

    def reset(self, strategy=None) -> None:
        self._metric.reset()

    def result(self, strategy=None) -> TResult:
        return self._metric.result()

    def update(self, strategy: "SupervisedTemplate"):
        self._metric._metric.to(strategy.device)
        self._metric.update(
            strategy.mb_output,  # type: ignore
            strategy.mb_y,
            strategy.mb_task_id,
        )


class AUROCMetrics(ClassificationPluginMetric):
    def __init__(self, num_classes: int = 10, average: TAverage = "macro"):
        metric = tm.AUROC(
            num_classes=num_classes,
            task="multiclass",
            average=average,
        )
        super(AUROCMetrics, self).__init__(
            metric,
            reset_at="stream",
            emit_at="stream",
            mode="eval",
        )

    def __str__(self):
        return "AUROC"


class PrecisionMetrics(ClassificationPluginMetric):
    def __init__(self, num_classes: int = 10, average: TAverage = "macro"):
        """
        Creates an instance of Accuracy metric
        """
        metric = tm.Precision(
            task="multiclass",
            num_classes=num_classes,
            average=average,
        )
        super(PrecisionMetrics, self).__init__(
            metric,
            reset_at="stream",
            emit_at="stream",
            mode="eval",
        )

    def __str__(self):
        return "Precision_"


class RecallMetrics(ClassificationPluginMetric):
    def __init__(self, num_classes: int = 10, average: TAverage = "macro"):
        """
        Creates an instance of Accuracy metric
        """
        metric = tm.Recall(
            task="multiclass",
            num_classes=num_classes,
            average=average,
        )
        super(RecallMetrics, self).__init__(
            metric,
            reset_at="stream",
            emit_at="stream",
            mode="eval",
        )

    def __str__(self):
        return "Recall"


class F1Metrics(ClassificationPluginMetric):
    def __init__(self, num_classes: int = 10, average: TAverage = "macro"):
        """
        Creates an instance of Accuracy metric
        """
        metric = tm.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average=average,
        )
        super(F1Metrics, self).__init__(
            metric,
            reset_at="stream",
            emit_at="stream",
            mode="eval",
        )

    def __str__(self):
        return "F1"


def classification_metrics(
    *,
    num_classes: int = 10,
    average: TAverage = "macro",
    # roc=False,
    auroc=False,
    recall=False,
    precision=False,
    f1=False,
) -> List[PluginMetric]:
    metrics = []

    if auroc:
        metrics.append(AUROCMetrics(num_classes=num_classes, average=average))

    if recall:
        metrics.append(RecallMetrics(num_classes=num_classes, average=average))

    if f1:
        metrics.append(F1Metrics(num_classes=num_classes, average=average))

    if precision:
        metrics.append(
            PrecisionMetrics(num_classes=num_classes, average=average)
        )

    return metrics


__all__ = [
    "classification_metrics",
    "AUROCMetrics",
    "PrecisionMetrics",
    "RecallMetrics",
    "F1Metrics",
    "TAverage",
]
