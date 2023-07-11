import numpy as np
import pandas as pd

from src.helpers.config import Config

from .base import BaseTransform, apply_transforms
from .general import (
    ColumnsDropTransform,
    DiscretizeColumnTransform,
    OneHotColumnsTransform,
)


class BucketCPUPercentTransform(BaseTransform):
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
        return f"BucketCPUPercentTransform(target_name={self.target_name})"


class FeatureA_TransformSet(BaseTransform):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = list(
            filter(
                lambda x: x != config.dataset.target,
                [
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
                ],
            )
        )
        self._transforms: list[BaseTransform] = [
            BucketCPUPercentTransform(),
            ColumnsDropTransform(
                columns=[
                    "vmid",
                    "subscriptionid",
                    "deploymentid",
                    "vmcreated",
                    "vmdeleted",
                    "maxcpu",
                    "avgcpu",
                    "vmcategory",
                    "vmcorecountbucket",
                    "vmmemorybucket",
                    "lifetime",
                    "corehour",
                ]
            ),
            OneHotColumnsTransform(
                columns=[
                    "vmcategory",
                    "vmcorecountbucket",
                    "vmmemorybucket",
                ],
            ),
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


__all__ = ["BucketCPUPercentTransform", "FeatureA_TransformSet"]
