from abc import ABCMeta, abstractmethod
from typing import Callable, TypeAlias, TypeVar

import pandas as pd

TData = TypeVar("TData", bound=pd.DataFrame)
TransformFn = Callable[[TData], TData]
TAcceptableTransform: TypeAlias = TransformFn | "BaseTransform"


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
    transforms: list[BaseTransform | TransformFn]
    | TransformFn
    | BaseTransform
    | None = None,
) -> TData:
    if transforms is None:
        return data

    if isinstance(transforms, list):
        for transform in transforms:
            data = transform(data)
        return data

    # if isinstance(transforms, Callable):
    #     return transforms(data)

    return transforms(data)


class BaseFeatureTransformSet(BaseTransform, metaclass=ABCMeta):
    @property
    @abstractmethod
    def transform_set(self) -> list[BaseTransform | TransformFn]:
        raise NotImplementedError

    @property
    @abstractmethod
    def target_name(self) -> str:
        raise NotImplementedError

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return apply_transforms(data, self.transform_set)


__all__ = [
    "BaseTransform",
    "BaseFeatureTransformSet",
    "TransformFn",
    "apply_transforms",
    "TAcceptableTransform",
]
