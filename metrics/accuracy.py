from collections import defaultdict
from typing import List

import torch
from torch import Tensor

from avalanche.evaluation import GenericPluginMetric, PluginMetric
from avalanche.evaluation.metrics.accuracy import (
    Accuracy,
    AccuracyPluginMetric,
    TaskAwareAccuracy,
)


class AccuracyWithTolerance(Accuracy):
    def __init__(self, tolerance: int = 0):
        """Accuracy metric with tolerance.

        :param tolerance: The tolerance for the accuracy.
               tolerance = 0 means that the predicted label must be the same with the true label (similar to usual accuracy).
        """
        super().__init__()
        self._tolerance = tolerance

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
    ) -> None:
        """Update the running accuracy given the true and predicted labels.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.

        :return: None.
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError(
                "Size mismatch for true_y and predicted_y tensors"
            )

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        # Compute the accuracy if the difference between the true and predicted labels is less or equal than the tolerance
        diffs = torch.abs(true_y - predicted_y)
        diffs = torch.where(
            diffs <= self._tolerance,
            torch.tensor(1.0),
            torch.tensor(0.0),
        )

        true_positives = float(torch.sum(diffs))
        total_patterns = len(true_y)
        self._mean_accuracy.update(
            true_positives / total_patterns, total_patterns
        )


class TaskAwareAccuracyWithTolerance(TaskAwareAccuracy):
    def __init__(self, tolerance: int = 0):
        self._mean_accuracy = defaultdict(
            lambda: AccuracyWithTolerance(tolerance)
        )


class AccuracyPluginMetricWithTolerance(AccuracyPluginMetric):
    """
    Base class for all accuracies with tolerance plugin metrics
    """

    def __init__(
        self,
        reset_at,
        emit_at,
        mode,
        tolerance: int = 0,
        split_by_task=False,
    ):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        self.split_by_task = split_by_task
        if self.split_by_task:
            self._accuracy = TaskAwareAccuracy(tolerance=tolerance)
        else:
            self._accuracy = Accuracy(tolerance=tolerance)
        super(AccuracyPluginMetric, self).__init__(
            self._accuracy,
            reset_at=reset_at,
            emit_at=emit_at,
            mode=mode,
        )

    def update(self, strategy):
        if isinstance(self._accuracy, AccuracyWithTolerance):
            self._accuracy.update(strategy.mb_output, strategy.mb_y)
        elif isinstance(
            self._accuracy, TaskAwareAccuracyWithTolerance
        ):
            self._accuracy.update(
                strategy.mb_output, strategy.mb_y, strategy.mb_task_id
            )
        else:
            assert False, "should never get here."


class MinibatchAccuracyWithTolerance(
    AccuracyPluginMetricWithTolerance
):
    """
    The minibatch plugin accuracy metric.
    This metric only works at training time.

    This metric computes the average accuracy with tolerance over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracyWithToleranec` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchAccuracyWithTolerance metric.
        """
        super(MinibatchAccuracyWithTolerance, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Top1_Acc_MB_Tol"


class EpochAccuracyWithTolerance(AccuracyPluginMetricWithTolerance):
    """
    The average accuracy with tolerance over a single training epoch.
    This plugin metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochAccuracyWithTolerance metric.
        """

        super(EpochAccuracyWithTolerance, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "Top1_Acc_Epoch_Tol"


class RunningEpochAccuracyWithTolerance(
    AccuracyPluginMetricWithTolerance
):
    """
    The average accuracy with tolerance across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self, tolerance: int = 0):
        """
        Creates an instance of the RunningEpochAccuracyWithTolerance metric.
        """

        super(RunningEpochAccuracyWithTolerance, self).__init__(
            tolerance=tolerance,
            reset_at="epoch",
            emit_at="iteration",
            mode="train",
        )

    def __str__(self):
        return "Top1_RunningAcc_Epoch_Tol"


class ExperienceAccuracyWithTolerance(
    AccuracyPluginMetricWithTolerance
):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy with tolerance over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self, tolerance: int = 0):
        """
        Creates an instance of ExperienceAccuracyWithTolerance metric
        """
        super(ExperienceAccuracyWithTolerance, self).__init__(
            tolerance=tolerance,
            reset_at="experience",
            emit_at="experience",
            mode="eval",
        )

    def __str__(self):
        return "Top1_Acc_Exp_Tol"


class StreamAccuracyWithTolerance(AccuracyPluginMetricWithTolerance):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy with tolerance over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self, tolerance: int = 0):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamAccuracyWithTolerance, self).__init__(
            tolerance=tolerance,
            reset_at="stream",
            emit_at="stream",
            mode="eval",
        )

    def __str__(self):
        return "Top1_Acc_Stream_Tol"


class TrainedExperienceAccuracyWithTolerance(
    AccuracyPluginMetricWithTolerance
):
    """
    At the end of each experience, this plugin metric reports the average
    accuracy with tolerance for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self, tolerance: int = 0):
        """
        Creates an instance of TrainedExperienceAccuracy metric by first
        constructing AccuracyPluginMetric
        """
        super(TrainedExperienceAccuracyWithTolerance, self).__init__(
            tolerance=tolerance,
            reset_at="stream",
            emit_at="stream",
            mode="eval",
        )
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = (
            strategy.experience.current_experience
        )
        # Reset average after learning from a new experience
        AccuracyPluginMetricWithTolerance.reset(self, strategy)
        return AccuracyPluginMetricWithTolerance.after_training_exp(
            self, strategy
        )

    def update(self, strategy):
        """
        Only update the accuracy with results from experiences that have been
        trained on
        """
        if (
            strategy.experience.current_experience
            <= self._current_experience
        ):
            AccuracyPluginMetricWithTolerance.update(self, strategy)

    def __str__(self):
        return "Accuracy_On_Trained_Experiences_Tol"


def accuracy_metrics_with_tolerance(
    *,
    tolerance: int = 0,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch accuracy at training time.
    :param experience: If True, will return a metric able to log
        the accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the accuracy averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation accuracy only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(
            MinibatchAccuracyWithTolerance(tolerance=tolerance)
        )

    if epoch:
        metrics.append(
            EpochAccuracyWithTolerance(tolerance=tolerance)
        )

    if epoch_running:
        metrics.append(
            RunningEpochAccuracyWithTolerance(tolerance=tolerance)
        )

    if experience:
        metrics.append(
            ExperienceAccuracyWithTolerance(tolerance=tolerance)
        )

    if stream:
        metrics.append(
            StreamAccuracyWithTolerance(tolerance=tolerance)
        )

    if trained_experience:
        metrics.append(
            TrainedExperienceAccuracyWithTolerance(
                tolerance=tolerance
            )
        )

    return metrics


__all__ = [
    "MinibatchAccuracyWithTolerance",
    "EpochAccuracyWithTolerance",
    "RunningEpochAccuracyWithTolerance",
    "ExperienceAccuracyWithTolerance",
    "StreamAccuracyWithTolerance",
    "TrainedExperienceAccuracyWithTolerance",
    "accuracy_metrics_with_tolerance",
]
