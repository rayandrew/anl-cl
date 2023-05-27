from abc import ABC, ABCMeta, abstractmethod
from typing import Generic, Tuple, TypeVar

import numpy.typing as npt
import pandas as pd

# from typing import List
TObservationLikelihood = TypeVar(
    "TObservationLikelihood", bound="ObservationLikelihood"
)


class ObservationLikelihood(
    ABC, Generic[TObservationLikelihood], metaclass=ABCMeta
):
    @abstractmethod
    def pdf(self, data: pd.Series | npt.NDArray) -> npt.NDArray:
        pass

    @abstractmethod
    def reset_theta(self, theta: int) -> TObservationLikelihood:
        pass

    @abstractmethod
    def update_theta(
        self, data: pd.Series | npt.NDArray
    ) -> TObservationLikelihood:
        pass

    @abstractmethod
    def save_theta(self) -> TObservationLikelihood:
        pass

    @abstractmethod
    def curr_theta(self) -> TObservationLikelihood:
        pass

    @abstractmethod
    def retrieve_theta(self) -> Tuple[float, float]:
        pass


__all__ = ["ObservationLikelihood"]