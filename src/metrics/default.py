from avalanche.evaluation.metrics import (
    cpu_usage_metrics,
    gpu_usage_metrics,
    loss_metrics,
    ram_usage_metrics,
    timing_metrics,
)


def get_classification_default_metrics(tolerance: int = 1):
    from src.metrics.accuracy import accuracy_metrics_with_tolerance
    from src.metrics.class_accuracy import (
        class_accuracy_metrics_with_tolerance,
    )
    from src.metrics.forgetting import (
        bwt_metrics_with_tolerance,
        forgetting_metrics_with_tolerance,
    )

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
    ]


__all__ = [
    "get_classification_default_metrics",
]
