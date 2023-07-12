import numpy as np
import pandas as pd

from src.helpers.config import Config

from .base import BaseFeatureTransformSet, BaseTransform, apply_transforms
from .general import (
    ColumnsDropTransform,
    DiscretizeColumnTransform,
    OneHotColumnsTransform,
)


class BucketSubscriptionCPUPercentTransform(BaseTransform):
    def __init__(self, target_name: str):
        self.target_name = target_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        temp_column = f"{self.target_name}_percent"
        df[temp_column] = df[self.target_name].apply(np.ceil)
        df = df.astype({temp_column: int})
        percent_df = pd.get_dummies(
            df[temp_column], prefix="percent", dtype=int
        )
        df = pd.concat([df, percent_df], axis=1)
        percent_columns = [f"percent_{i}" for i in range(0, 101)]
        df[percent_columns] = df.groupby(by=["subscriptionid"])[
            percent_columns
        ].transform(lambda x: x.cumsum())
        df = df.drop(columns=[temp_column])
        return df

    def __repr__(self) -> str:
        return (
            "BucketSubscriptionCPUPercentTransform(target_name="
            + f"{self.target_name})"
        )


NON_FEATURE_COLUMNS = [
    "vmid",
    "subscriptionid",
    "deploymentid",
    "vmcreated",
    "vmdeleted",
    "maxcpu",
    "p95maxcpu",
    "avgcpu",
    "vmcategory",
    "vmcorecountbucket",
    "vmmemorybucket",
    "lifetime",
    "corehour",
]


class NoFeats(BaseFeatureTransformSet):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = list(
            filter(lambda x: x != config.dataset.target, NON_FEATURE_COLUMNS)
        )
        self._transforms: list[BaseTransform] = [
            ColumnsDropTransform(columns=self._non_feature_columns),
            DiscretizeColumnTransform(
                column=config.dataset.target,
                new_column=self._target_name,
                n_bins=4,
            ),
        ]

    @property
    def target_name(self) -> str:
        return self._target_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return apply_transforms(data, self._transforms)

    def __repr__(self) -> str:
        return "NoFeats()"


class FeatureA_TransformSet(BaseFeatureTransformSet):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = list(
            filter(lambda x: x != config.dataset.target, NON_FEATURE_COLUMNS)
        )
        self._transforms: list[BaseTransform] = [
            BucketSubscriptionCPUPercentTransform(config.dataset.target),
            OneHotColumnsTransform(
                columns=[
                    "vmcategory",
                    "vmcorecountbucket",
                    "vmmemorybucket",
                ],
            ),
            ColumnsDropTransform(columns=self._non_feature_columns),
            DiscretizeColumnTransform(
                column=config.dataset.target,
                new_column=self._target_name,
                n_bins=4,
            ),
        ]

    @property
    def target_name(self) -> str:
        return self._target_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return apply_transforms(data, self._transforms)

    def __repr__(self) -> str:
        return "FeatureA_TransformSet()"


__all__ = ["BucketSubscriptionCPUPercentTransform", "FeatureA_TransformSet"]
