import numpy as np
import pandas as pd

from src.helpers.config import Config
from src.transforms.base import BaseTransform, TransformFn
from src.utils.logging import logging

from .base import BaseFeatureTransformSet, BaseTransform, TransformFn
from .general import (
    ColumnsDropTransform,
    DiscretizeColumnTransform,
    OneHotColumnsTransform,
    PrintColumnsTransform,
)

log = logging.getLogger(__name__)


class BucketSubscriptionCPUPercentTransform(BaseTransform):
    def __init__(self, target_name: str, drop_percent: bool = True):
        self.target_name = target_name
        self.drop_percent = drop_percent

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
        if self.drop_percent:
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
        self._config = config
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = list(
            filter(lambda x: x != config.dataset.target, NON_FEATURE_COLUMNS)
        )

    @property
    def transform_set(self) -> list[BaseTransform | TransformFn]:
        return [
            ColumnsDropTransform(columns=self._non_feature_columns),
            DiscretizeColumnTransform(
                column=self._config.dataset.target,
                new_column=self._target_name,
                n_bins=self._config.num_classes,
            ),
        ]

    @property
    def target_name(self) -> str:
        return self._target_name

    def __repr__(self) -> str:
        return "NoFeats()"


class FeatureA_TransformSet(BaseFeatureTransformSet):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = list(
            filter(
                lambda x: x != config.dataset.target and x != "DIST_COL",
                NON_FEATURE_COLUMNS,
            )
        )

    @property
    def transform_set(self) -> list[BaseTransform | TransformFn]:
        return [
            PrintColumnsTransform("Original Columns"),
            BucketSubscriptionCPUPercentTransform(self._config.dataset.target),
            OneHotColumnsTransform(
                columns=[
                    "vmcategory",
                    "vmcorecountbucket",
                    "vmmemorybucket",
                ],
            ),
            ColumnsDropTransform(columns=self._non_feature_columns),
            DiscretizeColumnTransform(
                column=self._config.dataset.target,
                new_column=self.target_name,
                n_bins=self._config.num_classes,
                drop_original=True,
            ),
            PrintColumnsTransform("Final Columns"),
            lambda df: df.iloc[0:500_000],
        ]

    @property
    def target_name(self) -> str:
        return self._target_name

    def __repr__(self) -> str:
        return "FeatureA_TransformSet()"


__all__ = ["BucketSubscriptionCPUPercentTransform", "FeatureA_TransformSet"]
