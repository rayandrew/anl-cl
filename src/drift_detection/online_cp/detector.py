from collections.abc import Callable
from functools import partial
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import ObservationLikelihood
from .hazard import constant_hazard

_default_hazard_func = partial(constant_hazard, lam=250)


class Detector:
    def __init__(self, dim: int):
        self.theta: Tuple[float, float] = (0.0, 0.0)
        self.CP: Any = np.zeros(1)
        self.R_old: npt.NDArray | List[int] = [1]
        self.maxes: npt.NDArray | List[int] = []
        self.curr_t = 0
        self.flag = False
        self.cnt = 0
        self.change = False
        self.prev_cp = 0
        self.last_cp = 0
        self.pred_save = 0

        # self.pred_save_mat = np.zeros((8000, 8000))
        # self.R_mat = np.zeros((1000, 1000))
        # self.R_mat_cumfreq = np.zeros((1000, 1000))
        # self.Mrun = np.zeros(1000)

        self.pred_save_mat = np.zeros((dim + 1, dim + 1))
        self.R_mat = np.zeros((dim + 1, dim + 1))
        self.R_mat_cumfreq = np.zeros((dim + 1, dim + 1))
        self.Mrun = np.zeros(dim + 1)

        self.trac = 0
        self.T = 1000

    def detect(
        self,
        x: pd.Series,
        observation_likelihood: ObservationLikelihood,
        hazard_func: Callable[
            [npt.NDArray], npt.NDArray
        ] = _default_hazard_func,
    ):
        # print("Detect")
        t = self.curr_t
        R = np.empty(t + 2)
        self.trac = self.trac + 1
        predprobs = observation_likelihood.pdf(x)
        self.predprobs = predprobs
        self.pred_save_mat[
            self.trac,
            int(self.last_cp) : int(self.last_cp) + len(predprobs),
        ] = predprobs

        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(self.curr_t + 1)))

        # Evaluate the growth probabilities
        R[1 : t + 2] = self.R_old[0 : t + 1] * predprobs * (1 - H)
        # np.dot(self.R_old[0:t+1],predprobs) * (1-H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0] = np.sum(self.R_old[0 : t + 1] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.

        self.R_old = R / np.sum(R)

        self.maxes = np.append(self.maxes, self.R_old.argmax())
        # print("   1")

        try:
            self.R_mat[self.trac, 0 : len(self.R_old)] = self.R_old
            self.R_mat_cumfreq[
                self.trac, 0 : len(self.R_old)
            ] = np.cumsum(self.R_old)
        except:
            self.R_mat[self.trac, 0 : len(self.R_old)] = self.R_old[
                0:-1
            ]
            self.R_mat_cumfreq[
                self.trac, 0 : len(self.R_old)
            ] = np.cumsum(self.R_old[0:-1])

        # print("   2")
        self.R_mat = self.R_mat.T

        self.R_mat_cumfreq = self.R_mat_cumfreq.T
        # self.R_mat_median = np.nanmedian(self.R_mat_cumfreq, axis=1)

        # print("   3")

        for ii in range(len(self.R_old)):
            try:
                self.Mrun[ii] = np.where(
                    self.R_mat_cumfreq[:, ii] >= 0.5
                )[0][0]
            except:
                pass

        # print("   4")

        data_history = "cumfreq"

        if data_history == "all":
            self.flag = False

        elif data_history == "argmax":
            if t > 0 and (self.maxes[-1] - self.maxes[-2]) < -10:
                self.flag = True
                observation_likelihood.curr_theta()
            elif self.flag == True and t > 0:
                if abs(self.maxes[-1] - self.maxes[-2]) < 5:
                    self.cnt += 1
                if self.cnt > 10:
                    self.change = True
                    self.flag = False
                    self.cnt = 0
                    self.CP = np.concatenate(
                        (
                            self.CP,
                            [self.last_cp + t - self.maxes[-1] + 1],
                        )
                    )
                    self.prev_cp = self.last_cp
                    self.last_cp = self.CP[-1]
                    self.curr_t = int(
                        t - (self.last_cp - self.prev_cp)
                    )
                    observation_likelihood.save_theta()
                else:
                    self.flag = False
                    self.cnt = 0

        elif data_history == "cumfreq":
            if (
                t > 0
                and (
                    self.Mrun[len(self.R_old) - 2]
                    - self.Mrun[len(self.R_old) - 1]
                )
                > 10
                and (self.flag == False)
            ):
                self.flag = True
                observation_likelihood.curr_theta()
            elif self.flag == True and t > 0:
                if (
                    abs(
                        self.Mrun[len(self.R_old) - 2]
                        - self.Mrun[len(self.R_old) - 1]
                    )
                    > 10
                ):
                    self.cnt += 1

                if self.cnt > 10:
                    self.change = True
                    self.flag = False
                    self.cnt = 0
                    self.CP = np.concatenate(
                        (self.CP, [self.last_cp + t - 11])
                    )
                    self.prev_cp = self.last_cp
                    self.last_cp = self.CP[-1]
                    self.curr_t = int(
                        t - 0.7 * (self.last_cp - self.prev_cp) + 1
                    )
                    observation_likelihood.save_theta()
                else:
                    self.flag = False
                    self.cnt = 0

        # print("End of Detect")
        if self.change == True:
            observation_likelihood.reset_theta(self.curr_t)
            self.pred_save = self.pred_save + 1
            self.change = False

        # print("   5")

        # Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)

        # print("   6")

        self.curr_t += 1

    def retrieve(self, observation_likelihood: ObservationLikelihood):
        observation_likelihood.curr_theta()
        observation_likelihood.save_theta()
        self.theta = observation_likelihood.retrieve_theta()
        return self.maxes, self.CP, self.theta, self.pred_save_mat

    def plot(self, x: List[int | float] | npt.NDArray):
        plt.scatter(len(self.maxes) * np.ones(len(x)), x)
        plt.plot([self.CP, self.CP], [np.min(x), np.max(x)], "r")
        plt.pause(0.0001)

    def plot_with_mean(
        self,
        x: List[int | float] | npt.NDArray,
        maxes: List[int | float] | npt.NDArray,
        CP: npt.NDArray,
    ):
        plt.scatter(len(maxes) * np.ones(len(x)), x)
        for num in range(len(CP)):
            plt.axvline(CP[num], color="m")
        plt.pause(0.0001)


__all__ = ["Detector"]
