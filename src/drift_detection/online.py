import numpy as np
from src.drift_detection.online_cp.detector import Detector
from src.drift_detection.online_cp.student_tmulti import StudentTMulti


class OnlineDriftDetector:
    def __init__(
        self,
        **kwargs,
    ):
        return

    def predict(self, data):
        # Get the length of the input data
        n = len(data)

        # Create an array of indices from 0 to n-1
        indices = np.arange(n).reshape((-1, 1))

        # Concatenate the indices and the original data along the second axis
        data_formatted = np.concatenate(
            (indices, data.reshape((-1, 1))), axis=1
        )

        detector = Detector(data_formatted.shape[0])
        # Number of cols
        observation_likelihood = StudentTMulti(
            data_formatted.shape[1]
        )

        R_mat = np.zeros(
            (data_formatted.shape[0], data_formatted.shape[0])
        )
        R_mat_cumfreq = np.zeros(
            (data_formatted.shape[0], data_formatted.shape[0])
        )
        R_mat.fill(np.nan)
        for t, x in enumerate(data_formatted[:, :]):
            print(t)
            # print(CP)
            detector.detect(
                x, observation_likelihood=observation_likelihood
            )
            _, CP, _, _ = detector.retrieve(observation_likelihood)

            R_old = detector.R_old

            try:
                R_mat[t, 0 : len(R_old)] = R_old
                R_mat_cumfreq[t, 0 : len(R_old)] = np.cumsum(R_old)
            except:
                R_mat[t, 0 : len(R_old)] = R_old[0:-1]
                R_mat_cumfreq[t, 0 : len(R_old)] = np.cumsum(
                    R_old[0:-1]
                )
        R_mat = R_mat.T
        R_mat_cumfreq = R_mat_cumfreq.T

        T = R_mat.shape[1]
        Mrun = np.zeros(T)

        for ii in range(T):
            try:
                Mrun[ii] = np.where(R_mat_cumfreq[:, ii] >= 0.5)[0][0]
            except:
                pass
        #########################################################################
        # Find the max value in Mrun sequentially
        # Check if the next value dropped a certain relative value
        # Check if that drop sustains for 10 points
        CP_CDF = [0]
        for i in range(len(Mrun) - 5):
            j = i + 1
            if (Mrun[i] - Mrun[j]) > 5:
                cnt = 0
                for k in range(1, 20):
                    if (i + k < len(Mrun)) and (
                        (Mrun[i] - Mrun[i + k]) > 10
                    ):
                        cnt = cnt + 1
                    else:
                        break
                if cnt > 10:
                    CP_CDF.append(i + 1)

        return CP_CDF


def get_online_drift_detector(
    **kwargs,
):
    return OnlineDriftDetector(
        **kwargs,
    )


__all__ = [
    "get_online_drift_detector",
]
