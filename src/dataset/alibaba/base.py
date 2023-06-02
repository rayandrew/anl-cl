from abc import ABCMeta

from src.dataset.base import BaseDataset


class AlibabaDataset(BaseDataset, metaclass=ABCMeta):
    pass


__all__ = ["AlibabaDataset"]
