from abc import ABCMeta

from src.dataset.base import BaseDataset


class GoogleDataset(BaseDataset, metaclass=ABCMeta):
    pass


__all__ = ["GoogleDataset"]
