import gorilla
import numpy as np
from sklearn.neighbors import KDTree

np.float = float

from skmultiflow.lazy.base_neighbors import BaseNeighbors

from src.utils.patch import patch_filter


@gorilla.patches(
    BaseNeighbors,
    gorilla.Settings(allow_hit=True),
    filter=patch_filter,
)
class CustomBaseNeighbors:
    @staticmethod
    def valid_metrics():
        return KDTree.valid_metrics()


__all__ = ["CustomBaseNeighbors"]
