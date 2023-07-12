from abc import ABCMeta, abstractmethod
from typing import Callable, TypeAlias

import pandas as pd

TData: TypeAlias = pd.DataFrame


class BaseTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, data: TData) -> TData:
        pass

    def transform(self, data: TData) -> TData:
        return self(data)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return self.__repr__()


def transform(fn: Callable[[TData], TData]):
    class Transform(BaseTransform):
        def __call__(self, df: TData) -> TData:
            return fn(df)

        def __repr__(self):
            return f"{self.__class__.__name__}({fn.__name__})"

    return Transform


def apply_transforms(
    data: TData,
    transforms: list[BaseTransform] | BaseTransform | None = None,
) -> TData:
    if transforms is None:
        return data

    if isinstance(transforms, list):
        for transform in transforms:
            data = transform(data)
        return data

    return transforms(data)


class BaseFeatureTransformSet(BaseTransform, metaclass=ABCMeta):
    @property
    @abstractmethod
    def target_name(self) -> str:
        raise NotImplementedError


__all__ = [
    "BaseTransform",
    "BaseFeatureTransformSet",
    "apply_transforms",
]
