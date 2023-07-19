# flake8: noqa: E501
from avalanche.evaluation.metrics import (  # cpu_usage_metrics,; gpu_usage_metrics,; ram_usage_metrics,; timing_metrics,
    accuracy_metrics,
    bwt_metrics,
    class_accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
)

from src.metrics.accuracy import accuracy_metrics_with_tolerance
from src.metrics.class_accuracy import class_accuracy_metrics_with_tolerance
from src.metrics.f1 import f1_metrics
from src.metrics.forgetting import (
    bwt_metrics_with_tolerance,
    forgetting_metrics_with_tolerance,
)
from src.metrics.precision import precision_metrics
from src.metrics.pytorch_metrics import AverageType
from src.metrics.recall import recall_metrics
from src.metrics.roc import auroc_metrics


def get_classification_default_metrics(
    num_classes: int = 10,
    average: AverageType = "macro",
    tolerance: int = 1,
):
    return [
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        accuracy_metrics_with_tolerance(
            tolerance=1,
            minibatch=True,
            epoch=True,
            epoch_running=False,
            experience=True,
            stream=True,
        )
        if tolerance > 0
        else accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=False,
            experience=True,
            stream=True,
        ),
        forgetting_metrics_with_tolerance(
            tolerance=tolerance, experience=True, stream=True
        )
        if tolerance > 0
        else forgetting_metrics(experience=True, stream=True),
        bwt_metrics_with_tolerance(
            tolerance=tolerance, experience=True, stream=True
        )
        if tolerance > 0
        else bwt_metrics(experience=True, stream=True),
        class_accuracy_metrics_with_tolerance(
            tolerance=tolerance, experience=True, stream=True
        )
        if tolerance > 0
        else class_accuracy_metrics(experience=True, stream=True),
        auroc_metrics(
            task="multiclass",
            num_classes=num_classes,
            average=average,
            epoch=False,
            experience=True,
            stream=True,
        ),
        f1_metrics(
            task="multiclass",
            num_classes=num_classes,
            average=average,
            epoch=False,
            experience=True,
            stream=True,
        ),
        precision_metrics(
            task="multiclass",
            num_classes=num_classes,
            average=average,
            epoch=False,
            experience=True,
            stream=True,
        ),
        recall_metrics(
            task="multiclass",
            num_classes=num_classes,
            average=average,
            epoch=False,
            experience=True,
            stream=True,
        ),
    ]


__all__ = [
    "get_classification_default_metrics",
]
