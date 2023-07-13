from collections import deque
from collections.abc import Sequence

from river.base import DriftDetector


class VotingDriftDetector:
    def __init__(
        self,
        window_size: int,
        threshold: int,
        verbose=True,
    ):
        self.methods: list[DriftDetector] = []
        self.weights: list[float] = []
        self.drifts: list[deque[int]] = []
        self.vote_drifts: list[int] = []
        self.window_size = window_size
        self.threshold = threshold
        self.verbose = verbose

    def add_method(self, method: DriftDetector, weight: float = 1.0):
        self.methods.append(method)
        self.weights.append(weight)
        self.drifts.append(deque[int]())

    def _get_drift_point(self, data):
        for i, val in enumerate(data):
            for method_index, method in enumerate(self.methods):
                _ = method.update(val)
                if method.drift_detected:
                    self.drifts[method_index].append(i)

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

    def predict(self, data) -> Sequence[int]:
        for method_idx in range(len(self.drifts)):
            self.drifts[method_idx] = deque()
        self._get_drift_point(data)
        return self._vote_drift(data, self.window_size, self.threshold)


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
        from river.drift import ADWIN

        dd.add_method(ADWIN(), 1)
    # if ddm:
    #     from river.drift.binary import DDM

    #     dd.add_method(DDM(), 1)
    # if eddm:
    #     from river.drift.binary import EDDM

    #     dd.add_method(EDDM(), 1)
    # if hddm_a:
    #     from river.drift.binary import HDDM_A

    #     dd.add_method(HDDM_A(), 1)
    # if hddm_w:
    #     from river.drift.binary import HDDM_W

    #     dd.add_method(HDDM_W(), 1)
    if page_hinkley:
        from river.drift import PageHinkley

        dd.add_method(PageHinkley(), 1)
    if kswin:
        from river.drift import KSWIN

        dd.add_method(KSWIN(), 1)
    return dd


__all__ = ["VotingDriftDetector", "get_offline_voting_drift_detector"]
