import ruptures as rpt


class RupturesDriftDetector:
    def __init__(
        self,
        kernel="linear",
        min_size=2,
        jump=1,
        penalty=100,
        params=None,
        **kwargs,
    ):
        self.detector = rpt.KernelCPD(
            kernel=kernel,
            min_size=min_size,
            jump=jump,
            params=params,
            **kwargs,
        )
        self.penalty = penalty

    def predict(self, data):
        return self.detector.fit_predict(data, pen=self.penalty)


def get_offline_ruptures_drift_detector(
    kernel="linear",
    min_size=2,
    jump=1,
    penalty=100,
    params=None,
    **kwargs,
):
    return RupturesDriftDetector(
        kernel=kernel,
        min_size=min_size,
        jump=jump,
        params=params,
        penalty=penalty,
        **kwargs,
    )


__all__ = [
    "get_offline_ruptures_drift_detector",
]
