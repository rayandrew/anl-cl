from abc import ABCMeta, abstractmethod
from typing import Any, Protocol

import pandas as pd


# https://github.com/ContinualAI/avalanche/blob/2b7fa26f0ca98603b057a2eee992a4dc3a55abe1/avalanche/benchmarks/utils/transform_groups.py#L57
class ComposedTransformDef(Protocol):
    def __call__(self, *data: Any) -> Any:
        pass


class TransformDef(Protocol):
    def __call__(self, data: Any) -> Any:
        pass


Transform = TransformDef | ComposedTransformDef


class BaseTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self(data)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return self.__repr__()


def apply_transforms(
    data: Any,
    transforms: list[Transform] | Transform | None = None,
) -> Any:
    """Apply a list of transforms to a data object."""
    if transforms is None:
        return data

    if isinstance(transforms, list):
        for transform in transforms:
            data = transform(data)
        return data

    return transforms(data)


# @overload
# def apply_transforms(
#     data: pd.DataFrame,
#     transforms: list[Transform] | Transform | None = None,
# ) -> pd.DataFrame:
#     ...


class BaseFeatureEngineering(metaclass=ABCMeta):
    """Base class for feature engineering.

    This class is used to define the feature engineering process.

    The process is as follows:
    data
        -> preprocess_transform
        -> for chunk in chunk_transform:
            -> chunk_transform
        -> postprocess_transform
    -> final data
    """

    @property
    @abstractmethod
    def target_name(self) -> str:
        raise NotImplementedError

    @property
    def preprocess_transform_set(
        self,
    ) -> list[Transform] | None:
        return None

    def apply_preprocess_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return apply_transforms(data, self.preprocess_transform_set)

    @property
    def chunk_transform_set(
        self,
    ) -> list[Transform] | None:
        return None

    def apply_chunk_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return apply_transforms(data, self.chunk_transform_set)

    @property
    def postprocess_transform_set(
        self,
    ) -> list[Transform] | None:
        return None

    def apply_postprocess_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return apply_transforms(data, self.postprocess_transform_set)

    def __repr__(self) -> str:
        return (
            f"""{self.__class__.__name__}(
            target_name="{self.target_name}",
            preprocess_transform_set=[
        """
            + ",\n".join(
                [f"{transform}" for transform in self.preprocess_transform_set]
            )
            if self.preprocess_transform_set is not None
            else ""
            + """
            ],
            chunk_transform_set=[
        """
            + ",\n".join(
                [f"{transform}" for transform in self.chunk_transform_set]
            )
            if self.chunk_transform_set is not None
            else ""
            + """
            ],
            postprocess_transform_set=[
        """
            + ",\n".join(
                [f"{transform}" for transform in self.postprocess_transform_set]
            )
            if self.postprocess_transform_set is not None
            else ""
            + """
            ],
        )"""
        )


__all__ = [
    "apply_transforms",
    "ComposedTransformDef",
    "TransformDef",
    "Transform",
    "BaseFeatureEngineering",
    "BaseTransform",
]
