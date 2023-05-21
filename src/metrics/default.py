from avalanche.evaluation.metrics import (
    cpu_usage_metrics,
    gpu_usage_metrics,
    loss_metrics,
    ram_usage_metrics,
    timing_metrics,
)

from src.metrics.accuracy import accuracy_metrics_with_tolerance
from src.metrics.class_accuracy import (
    class_accuracy_metrics_with_tolerance,
)
from src.metrics.classification import (
    TAverage,
    classification_metrics,
)
from src.metrics.forgetting import (
    bwt_metrics_with_tolerance,
    forgetting_metrics_with_tolerance,
)


def get_classification_default_metrics(
    num_classes: int = 10,
    average: TAverage = "macro",
    tolerance: int = 1,
):
    return [
        loss_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        accuracy_metrics_with_tolerance(
            tolerance=1,
            minibatch=True,
            epoch=True,
            epoch_running=False,
            experience=True,
            stream=True,
        ),
        forgetting_metrics_with_tolerance(
            tolerance=tolerance, experience=True, stream=True
        ),
        bwt_metrics_with_tolerance(
            tolerance=tolerance, experience=True, stream=True
        ),
        class_accuracy_metrics_with_tolerance(
            tolerance=tolerance, experience=True, stream=True
        ),
        classification_metrics(
            num_classes=num_classes,
            average=average,
            auroc=True,
            recall=True,
            precision=True,
            f1=True,
        ),
    ]


__all__ = [
    "get_classification_default_metrics",
]
