from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Union

import torch
from torch import Tensor

from avalanche.evaluation import (
    Metric,
    PluginMetric,
    _ExtendedGenericPluginMetric,
    _ExtendedPluginMetricValue,
)
from avalanche.evaluation.metric_utils import (
    default_metric_name_template,
    generic_get_metric_name,
)
from avalanche.evaluation.metrics.class_accuracy import (
    TrackedClassesType,
)
from avalanche.training.templates import SupervisedTemplate


class ClassPredictionDiff(Metric[Dict[int, Dict[int, int]]]):
    """
    Modified from ClassAccuracy metric in Avalanche.
    """

    def __init__(
        self,
        classes: Optional[TrackedClassesType] = None,
    ):
        self.classes: Dict[int, Set[int]] = defaultdict(set)
        self.dynamic_classes = False
        self._class_diffs: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(0)
        )

        if classes is not None:
            if isinstance(classes, dict):
                # Task-id -> classes dict
                self.classes = {
                    task_id: self._ensure_int_classes(class_list)
                    for task_id, class_list in classes.items()
                }
            else:
                # Assume is a plain iterable
                self.classes = {0: self._ensure_int_classes(classes)}
        else:
            self.dynamic_classes = True

        self.__init_diffs_for_known_classes()

    @staticmethod
    def _ensure_int_classes(classes_iterable: Iterable[int]):
        return set(int(c) for c in classes_iterable)

    def __init_diffs_for_known_classes(self):
        for task_id, task_classes in self.classes.items():
            for c in task_classes:
                self._class_diffs[task_id][c] = 0

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        task_labels: Union[int, Tensor],
    ) -> None:
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
            task_labels = [task_labels] * len(true_y)

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        for pred, true, t in zip(predicted_y, true_y, task_labels):
            t = int(t)

            if self.dynamic_classes:
                self.classes[t].add(int(true))
            else:
                if t not in self.classes:
                    continue
                if int(true) not in self.classes[t]:
                    continue

            diff = true - pred

            self._class_diffs[t][int(diff.item())] += 1

    def result(self) -> Dict[int, Dict[int, int]]:
        running_class_accuracies = OrderedDict()
        for task_label in sorted(self._class_diffs.keys()):
            task_dict = self._class_diffs[task_label]
            running_class_accuracies[task_label] = OrderedDict()
            for class_id in sorted(task_dict.keys()):
                running_class_accuracies[task_label][
                    class_id
                ] = task_dict[class_id]

        return running_class_accuracies

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self._class_diffs = defaultdict(lambda: defaultdict(0))
        self.__init_diffs_for_known_classes()


class ClassPredictionDiffPluginMetric(_ExtendedGenericPluginMetric):
    """
    Base class for all class accuracy plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, classes=None):
        self._class_diff = ClassPredictionDiff(classes=classes)
        super(ClassPredictionDiffPluginMetric, self).__init__(
            self._class_diff,
            reset_at=reset_at,
            emit_at=emit_at,
            mode=mode,
        )

    def update(self, strategy: "SupervisedTemplate"):
        self._class_diff.update(
            strategy.mb_output, strategy.mb_y, strategy.mb_task_id
        )

    def result(
        self, strategy: "SupervisedTemplate"
    ) -> List[_ExtendedPluginMetricValue]:
        metric_values = []
        task_diffs = self._class_diff.result()
        phase_name = "train" if strategy.is_training else "eval"
        stream_name = strategy.experience.origin_stream.name
        experience_id = strategy.experience.current_experience

        for task_id, task_classes in task_diffs.items():
            for class_id, class_diff in task_classes.items():
                metric_values.append(
                    _ExtendedPluginMetricValue(
                        metric_name=str(self),
                        metric_value=class_diff,
                        phase_name=phase_name,
                        stream_name=stream_name,
                        task_label=task_id,
                        experience_id=experience_id,
                        class_id=class_id,
                    )
                )

        return metric_values

    def metric_value_name(
        self, m_value: _ExtendedPluginMetricValue
    ) -> str:
        m_value_values = vars(m_value)
        add_exp = self._emit_at == "experience"
        if not add_exp:
            del m_value_values["experience_id"]
        m_value_values["class_id"] = m_value.other_info["class_id"]

        return generic_get_metric_name(
            default_metric_name_template(m_value_values)
            + "/{class_id}",
            m_value_values,
        )


class EpochPredictionDiff(ClassPredictionDiffPluginMetric):
    def __init__(self, classes=None):
        """
        Creates an instance of the EpochClassAccuracy metric.
        """
        super().__init__(
            reset_at="stream",
            emit_at="stream",
            mode="eval",
            classes=classes,
        )

    def __str__(self):
        return "Top1_ClassDiff_Epoch"


def class_diff_metrics(
    *,
    stream=False,
    classes=None,
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param stream: If True, will return a metric able to log
        the per-class accuracy averaged over the entire evaluation stream of
        experiences.
    :param classes: The list of classes to track. See the corresponding
        parameter of :class:`ClassAccuracy` for a precise explanation.

    :return: A list of plugin metrics.
    """
    metrics = []

    if stream:
        metrics.append(ClassPredictionDiff(classes=classes))

    return metrics


__all__ = [
    "ClassPredictionDiff",
    "class_diff_metrics",
]
