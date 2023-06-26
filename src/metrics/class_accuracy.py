from typing import List, Union, cast

import torch
from torch import Tensor

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics.class_accuracy import (
    ClassAccuracy,
    ClassAccuracyPluginMetric,
    TrackedClassesType,
)


class ClassAccuracyWithTolerance(ClassAccuracy):
    def __init__(
        self,
        classes: TrackedClassesType | None = None,
        tolerance: int = 0,
    ):
        super().__init__(classes)
        self.tolerance = tolerance

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        task_labels: Union[int, Tensor],
    ) -> None:
        """
        Update the running accuracy given the true and predicted labels for each
        class.
        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.
        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError(
                "Size mismatch for true_y and predicted_y tensors"
            )

        if isinstance(task_labels, Tensor) and len(
            task_labels
        ) != len(true_y):
            raise ValueError(
                "Size mismatch for true_y and task_labels tensors"
            )

        if not isinstance(task_labels, (int, Tensor)):
            raise ValueError(
                f"Task label type: {type(task_labels)}, "
                f"expected int or Tensor"
            )

        if isinstance(task_labels, int):
            task_labels = [task_labels] * len(true_y)  # type: ignore

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        for pred, true, t in zip(predicted_y, true_y, task_labels):  # type: ignore
            t = int(t)

            if self.dynamic_classes:
                self.classes[t].add(int(true))
            else:
                if t not in self.classes:
                    continue
                if int(true) not in self.classes[t]:
                    continue

            diff = torch.abs(pred - true)

            true_positives = (
                torch.where(
                    diff <= self.tolerance,
                    torch.tensor(1.0),
                    torch.tensor(0.0),
                )
                .float()
                .item()
            )

            self._class_accuracies[t][int(true)].update(
                true_positives, 1
            )


class ClassAccuracyPluginMetricWithTolerance(
    ClassAccuracyPluginMetric
):
    """
    Base class for all class accuracy plugin metrics
    """

    def __init__(
        self,
        reset_at,
        emit_at,
        mode,
        tolerance: int = 0,
        classes=None,
    ):
        self._class_accuracy = ClassAccuracyWithTolerance(
            classes=classes, tolerance=tolerance
        )
        super(ClassAccuracyPluginMetric, self).__init__(
            self._class_accuracy,
            reset_at=reset_at,
            emit_at=emit_at,
            mode=mode,
        )


class MinibatchClassAccuracyWithTolerance(
    ClassAccuracyPluginMetricWithTolerance
):
    """
    The minibatch plugin class accuracy metric.
    This metric only works at training time.
    This metric computes the average accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.
    If a more coarse-grained logging is needed, consider using
    :class:`EpochClassAccuracyWithTolerance` instead.
    """

    def __init__(self, classes=None, tolerance: int = 0):
        """
        Creates an instance of the MinibatchClassAccuracyWithTolerance metric.
        """
        super().__init__(
            reset_at="iteration",
            emit_at="iteration",
            mode="train",
            classes=classes,
            tolerance=tolerance,
        )

    def __str__(self):
        return "Top1_ClassAcc_MB_Tol"


class EpochClassAccuracyWithTolerance(
    ClassAccuracyPluginMetricWithTolerance
):
    """
    The average class accuracy over a single training epoch.
    This plugin metric only works at training time.
    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch (separately
    for each class).
    """

    def __init__(self, classes=None, tolerance: int = 0):
        """
        Creates an instance of the EpochClassAccuracyWithTolerance metric.
        """
        super().__init__(
            reset_at="epoch",
            emit_at="epoch",
            mode="train",
            classes=classes,
            tolerance=tolerance,
        )

    def __str__(self):
        return "Top1_ClassAcc_Epoch_Tol"


class RunningEpochClassAccuracyWithTolerance(
    ClassAccuracyPluginMetricWithTolerance
):
    """
    The average class accuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.
    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch (separately for each class).
    The metric resets its state after each training epoch.
    """

    def __init__(self, classes=None, tolerance: int = 0):
        """
        Creates an instance of the RunningEpochClassAccuracyWithTolerance metric.
        """

        super().__init__(
            reset_at="epoch",
            emit_at="iteration",
            mode="train",
            classes=classes,
            tolerance=tolerance,
        )

    def __str__(self):
        return "Top1_RunningClassAcc_Epoch_Tol"


class ExperienceClassAccuracyWithTolerance(
    ClassAccuracyPluginMetricWithTolerance
):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience (separately
    for each class).
    This metric only works at eval time.
    """

    def __init__(self, classes=None, tolerance: int = 0):
        """
        Creates an instance of ExperienceClassAccuracyWithTolerance metric
        """
        super().__init__(
            reset_at="experience",
            emit_at="experience",
            mode="eval",
            classes=classes,
            tolerance=tolerance,
        )

    def __str__(self):
        return "Top1_ClassAcc_Exp_Tol"


class StreamClassAccuracyWithTolerance(
    ClassAccuracyPluginMetricWithTolerance
):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences
    (separately for each class).
    This metric only works at eval time.
    """

    def __init__(self, classes=None, tolerance: int = 0):
        """
        Creates an instance of StreamClassAccuracyWithTolerance metric
        """
        super().__init__(
            reset_at="stream",
            emit_at="stream",
            mode="eval",
            classes=classes,
            tolerance=tolerance,
        )

    def __str__(self):
        return "Top1_ClassAcc_Stream_Tol"


def class_accuracy_metrics_with_tolerance(
    *,
    tolerance: int = 0,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    classes=None,
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.
    :param tolerance: The tolerance to use for the class accuracy
    :param minibatch: If True, will return a metric able to log
        the per-class minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the per-class epoch accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the per-class  running epoch accuracy at training time.
    :param experience: If True, will return a metric able to log
        the per-class accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the per-class accuracy averaged over the entire evaluation stream of
        experiences.
    :param classes: The list of classes to track. See the corresponding
        parameter of :class:`ClassAccuracy` for a precise explanation.
    :return: A list of plugin metrics.
    """
    metrics = []
    if minibatch:
        metrics.append(
            MinibatchClassAccuracyWithTolerance(
                classes=classes, tolerance=tolerance
            )
        )

    if epoch:
        metrics.append(
            EpochClassAccuracyWithTolerance(
                classes=classes, tolerance=tolerance
            )
        )

    if epoch_running:
        metrics.append(
            RunningEpochClassAccuracyWithTolerance(
                classes=classes, tolerance=tolerance
            )
        )

    if experience:
        metrics.append(
            ExperienceClassAccuracyWithTolerance(
                classes=classes, tolerance=tolerance
            )
        )

    if stream:
        metrics.append(
            StreamClassAccuracyWithTolerance(
                classes=classes, tolerance=tolerance
            )
        )

    return metrics


__all__ = [
    "ClassAccuracyPluginMetricWithTolerance",
    "MinibatchClassAccuracyWithTolerance",
    "EpochClassAccuracyWithTolerance",
    "RunningEpochClassAccuracyWithTolerance",
    "ExperienceClassAccuracyWithTolerance",
    "StreamClassAccuracyWithTolerance",
    "class_accuracy_metrics_with_tolerance",
]
