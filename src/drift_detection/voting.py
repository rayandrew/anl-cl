from collections import deque
from collections.abc import Collection

from skmultiflow.drift_detection.base_drift_detector import (
    BaseDriftDetector,
)


class VotingDriftDetector:
    def __init__(
        self,
        window_size: int,
        threshold: int,
        verbose=True,
    ):
        self.methods: list[BaseDriftDetector] = []
        self.weights: list[float] = []
        self.drifts: list[deque[int]] = []
        self.vote_drifts: list[int] = []
        self.window_size = window_size
        self.threshold = threshold
        self.verbose = verbose

    def add_method(
        self, method: BaseDriftDetector, weight: float = 1.0
    ):
        self.methods.append(method)
        self.weights.append(weight)
        self.drifts.append(deque[int]())

    def _get_drift_point(self, data):
        for pos, ele in enumerate(data):
            for method_index, method in enumerate(self.methods):
                method.add_element(ele)
                if method.detected_change():
                    self.drifts[method_index].append(pos)

    def _vote_drift(self, data, window_size, threshold):
        vote_drifts = []
        for i in range(0, len(data), window_size):
            pos_sum = 0
            weight_sum = 0
            for method_index in range(len(self.methods)):
                while len(self.drifts[method_index]) != 0:
                    pos = self.drifts[method_index][0]
                    if pos >= (i + 1) * window_size:
                        break
                    else:
                        pos_sum += self.weights[method_index] * pos
                        weight_sum += self.weights[method_index]
                        self.drifts[method_index].popleft()
            if weight_sum != 0 and self.verbose:
                print(weight_sum)
            if weight_sum > threshold:
                mean_pos = int(pos_sum / weight_sum)
                vote_drifts.append(mean_pos)
        return vote_drifts

    def predict(self, data) -> Collection[int]:
        for method_idx in range(len(self.drifts)):
            self.drifts[method_idx] = deque()
        self._get_drift_point(data)
        return self._vote_drift(
            data, self.window_size, self.threshold
        )


def get_offline_voting_drift_detector(
    window_size: int = 75,
    threshold: int = 300,
    adwin: bool = True,
    ddm: bool = True,
    eddm: bool = True,
    hddm_a: bool = True,
    hddm_w: bool = True,
    page_hinkley: bool = True,
    kswin: bool = True,
    verbose: bool = False,
):
    dd = VotingDriftDetector(
        window_size=window_size,
        threshold=threshold,
        verbose=verbose,
    )
    if adwin:
        from skmultiflow.drift_detection.adwin import ADWIN

        dd.add_method(ADWIN(), 1)
    if ddm:
        from skmultiflow.drift_detection.ddm import DDM

        dd.add_method(DDM(), 1)
    if eddm:
        from skmultiflow.drift_detection.eddm import EDDM

        dd.add_method(EDDM(), 1)
    if hddm_a:
        from skmultiflow.drift_detection.hddm_a import HDDM_A

        dd.add_method(HDDM_A(), 1)
    if hddm_w:
        from skmultiflow.drift_detection.hddm_w import HDDM_W

        dd.add_method(HDDM_W(), 1)
    if page_hinkley:
        from skmultiflow.drift_detection.page_hinkley import (
            PageHinkley,
        )

        dd.add_method(PageHinkley(), 1)
    if kswin:
        from skmultiflow.drift_detection.kswin import KSWIN

        dd.add_method(KSWIN(), 1)
    return dd


__all__ = ["VotingDriftDetector", "get_offline_voting_drift_detector"]
