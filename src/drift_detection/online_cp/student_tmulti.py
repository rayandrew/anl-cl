from typing import List

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from src.drift_detection.online_cp.base import ObservationLikelihood


class StudentTMulti(ObservationLikelihood["StudentTMulti"]):
    def __init__(self, dim: int):
        self.nu0 = self.nu = np.array(dim)
        self.Lambda0 = self.Lambda = np.eye(dim) * 0.001
        self.kappa0 = self.kappa = np.array(1)
        self.mu0 = self.mu = np.zeros(dim)
        self.dim = dim
        self.curr_mean = self.mu0
        self.curr_cov = self.Lambda0
        self.saved_mean: npt.NDArray | List[npt.NDArray] = []
        self.saved_cov: npt.NDArray | List[npt.NDArray] = []

    def pdf(self, data):
        df = self.nu
        dim = self.dim
        loc = self.mu
        lam = self.Lambda
        length = int(np.size(loc) / dim)
        mult: npt.NDArray | List[float] = np.zeros(length)
        scale = np.zeros((length, dim, dim))

        if length == 1:
            scale = np.linalg.inv(
                lam * (2 * (self.kappa + 1)) / (df * self.kappa)
            )
            mult = [
                np.matmul(
                    np.matmul(data - loc, scale),
                    np.transpose(data - loc),
                )
            ]
        else:
            scaling = np.transpose(
                [2 * (self.kappa + 1) / (df * self.kappa)]
            )
            for i in range(length):
                scale[i] = np.linalg.inv(lam[i] * scaling[i])
                mult[i] = np.dot(
                    np.dot((data - loc)[i], scale[i]),
                    np.transpose((data - loc)[i]),
                )
        _, logdet = np.linalg.slogdet(scale)
        logc = (
            gammaln(df / 2.0 + dim / 2.0)
            - gammaln(df / 2.0)
            + 0.5 * logdet
            - dim / 2.0 * np.log(df * np.pi)
        )
        logc = np.transpose(logc)
        return np.exp(
            logc - (df / 2.0 + dim / 2.0) * np.log1p(mult / df)
        )

    def update_theta(self, data):
        if np.size([self.mu], axis=-2) > 1:
            kappa = np.transpose([self.kappa])
            # print("before")
            # print(self.mu)
            muT0: npt.NDArray = np.concatenate(  # type: ignore
                ([self.mu0], (kappa * self.mu + data) / (kappa + 1))
            )
            # print("now")
            kappaT0: npt.NDArray = np.concatenate(  # type: ignore
                ([self.kappa0], (self.kappa + 1))
            )
            nuT0: npt.NDArray = np.concatenate(  # type: ignore
                ([self.nu0], self.nu + 1)
            )
            x_mu = data - self.mu
            length = np.size(kappa)
            Lambda = np.zeros((length, self.dim, self.dim))
            for i in range(length):
                Lambda[i] = self.Lambda[i] + kappa[i] * np.matmul(
                    np.transpose([x_mu[i]]), [x_mu[i]]
                ) / (2.0 * (kappa[i] + 1.0))
            LambdaT0: npt.NDArray = np.concatenate(  # type: ignore
                ([self.Lambda0], Lambda)  # type: ignore
            )
        else:
            muT0 = np.stack(
                (
                    self.mu0,
                    (self.kappa * self.mu + data) / (self.kappa + 1),
                )
            )
            kappaT0 = np.stack((self.kappa0, self.kappa + 1))
            nuT0 = np.stack((self.nu0, self.nu + 1))
            x_mu = data - self.mu
            LambdaT0 = np.stack(
                (
                    self.Lambda0,
                    self.Lambda
                    + self.kappa
                    * np.matmul(x_mu.transpose(), x_mu)
                    / (2.0 * (self.kappa + 1.0)),
                )
            )

        self.mu = muT0
        self.kappa = kappaT0
        self.nu = nuT0
        self.Lambda = LambdaT0
        return self

    def curr_theta(self):
        self.curr_mean = self.mu[-2]
        self.curr_cov = (
            self.Lambda[-2]
            * 2
            * (self.kappa[-2] + 1)
            / (self.nu[-2] * self.kappa[-2])
        )
        return self

    def save_theta(self):
        if np.size(self.saved_mean):
            self.saved_mean = np.concatenate(
                (self.saved_mean, [self.curr_mean])
            )
        else:
            self.saved_mean = [self.curr_mean]
        if np.size(self.saved_cov):
            self.saved_cov = np.concatenate(
                (self.saved_cov, [self.curr_cov])
            )
        else:
            self.saved_cov = [self.curr_cov]

        return self

    def reset_theta(self, t):
        self.mu = self.mu[0 : t + 1]
        self.kappa = self.kappa[0 : t + 1]
        self.nu = self.nu[0 : t + 1]
        self.Lambda = self.Lambda[0 : t + 1]
        return self

    def retrieve_theta(self):
        return (self.saved_mean, self.saved_cov)


__all__ = ["StudentTMulti"]
