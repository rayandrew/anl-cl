import numpy as np
import numpy.typing as npt


def constant_hazard(r: npt.NDArray, lam: float | int):
    return 1 / lam * np.ones(r.shape)


__all__ = ["constant_hazard"]