from avalanche.evaluation.metrics.forward_transfer import (
    ExperienceForwardTransfer,
    StreamForwardTransfer,
)

from .accuracy import TaskAwareAccuracyWithTolerance


class ExperienceForwardTransferWithTolerance(
    ExperienceForwardTransfer
):
    """
    The Forward Transfer with Tolerance computed on each experience separately.
    The transfer is computed based on the accuracy metric.
    """

    def __init__(self, tolerance: int = 0):
        super(ExperienceForwardTransfer, self).__init__()

        self.tolerance = tolerance
        self._current_metric = TaskAwareAccuracyWithTolerance(
            tolerance=self.tolerance
        )


class StreamForwardTransferWithTolerance(StreamForwardTransfer):
    """
    The Forward Transfer with Tolerance averaged over all the evaluation experiences.
    This plugin metric, computed over all observed experiences during training,
    is the average over the difference between the accuracy result obtained
    after the previous experience and the accuracy result obtained
    on random initialization.
    """

    def __init__(self, tolerance: int = 0):
        super(StreamForwardTransfer, self).__init__()
        self.tolerance = tolerance
        self._current_metric = TaskAwareAccuracyWithTolerance(
            tolerance=self.tolerance
        )


def forward_transfer_metrics_with_tolerance(
    *, tolerance: int = 0, experience=False, stream=False
):
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.
    :param tolerance: The tolerance to use for the accuracy metric.
    :param experience: If True, will return a metric able to log
        the forward transfer on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the forward transfer averaged over the evaluation stream experiences,
        which have been observed during training.
    :return: A list of plugin metrics.
    """

    metrics = []

    if experience:
        metrics.append(
            ExperienceForwardTransferWithTolerance(
                tolerance=tolerance
            )
        )

    if stream:
        metrics.append(
            StreamForwardTransferWithTolerance(tolerance=tolerance)
        )

    return metrics


__all__ = [
    "ExperienceForwardTransferWithTolerance",
    "StreamForwardTransferWithTolerance",
    "forward_transfer_metrics_with_tolerance",
]
