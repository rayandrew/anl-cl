import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.helpers.config import Config
from src.helpers.definitions import DD_ID
from src.utils.general import discretize_column
from src.utils.logging import logging

from .base import BaseFeatureEngineering, BaseTransform, Transform
from .general import (
    ColumnsDropTransform,
    DiscretizeColumnTransform,
    NamedInjectTransform,
    OneHotColumnsTransform,
    PrintColumnsTransform,
)

log = logging.getLogger(__name__)


class BucketSubscriptionCPUPercentTransform_V1(BaseTransform):
    def __init__(
        self,
        target_name: str,
        drop_percent: bool = True,
        n_bins: int | None = 100,
        drop_first: bool = False,
    ):
        self.target_name = target_name
        self.drop_percent = drop_percent
        self.n_bins = n_bins if n_bins is not None else 100
        self.drop_first = drop_first

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert an iterable of indices to one-hot encoded labels."""

        df = data.copy()
        temp_column = f"{self.target_name}_percent"
        df[temp_column] = df[self.target_name].apply(np.ceil)
        df = df.astype({temp_column: int})

        # one hot encoding but with number of class
        targets = np.array(df[temp_column].values).reshape(-1)
        ohe = np.eye(self.n_bins + 1)[targets]
        percent_columns = [
            f"{temp_column}_{i}" for i in range(0, self.n_bins + 1)
        ]
        percent_df = pd.DataFrame(
            ohe,
            columns=percent_columns,
        )
        if self.drop_first:
            percent_df = percent_df.drop(columns=[f"{temp_column}_0"])
            percent_columns = percent_columns[1:]

        # percent_df = pd.get_dummies(
        #     df[temp_column], prefix="percent", dtype=int
        # )
        df = pd.concat([df, percent_df], axis=1)
        df[percent_columns] = df.groupby(by=["subscriptionid"])[
            percent_columns
        ].transform(lambda x: x.cumsum())
        if self.drop_percent:
            df = df.drop(columns=[temp_column])
        return df

    def __repr__(self) -> str:
        return f"""BucketSubscriptionCPUPercentTransform(
                version=1,
                target_name="{self.target_name}",
                drop_percent={self.drop_percent},
                n_bins={self.n_bins},
                drop_first={self.drop_first}
            )"""


# https://stackoverflow.com/a/61117770
def find_first(x: npt.ArrayLike):
    idx = x.view(bool).argmax() // x.itemsize
    return idx if x[idx] else -1


def custom_transform(data: pd.DataFrame) -> pd.DataFrame:
    shifted = data.shift(1, fill_value=0)
    first_idx = find_first(shifted.values == 1)
    if first_idx != -1:
        shifted.iloc[first_idx:] = True
    return shifted


class BucketSubscriptionCPUPercentTransform_V2(BaseTransform):
    def __init__(
        self,
        target_name: str,
        n_bins: int | None = 100,
        drop_first: bool = False,
    ):
        self.target_name = target_name
        self.n_bins = n_bins if n_bins is not None else 100
        self.drop_first = drop_first

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert an iterable of indices to one-hot encoded labels."""

        df = data.copy()
        temp_column = f"{self.target_name}_percent"
        df[temp_column] = df[self.target_name].apply(np.ceil)
        df = df.astype({temp_column: int})

        # one hot encoding but with number of class
        targets = np.array(df[temp_column].values).reshape(-1)
        ohe = np.eye(self.n_bins + 1)[targets]
        percent_columns = [
            f"{temp_column}_{i}" for i in range(0, self.n_bins + 1)
        ]
        percent_df = pd.DataFrame(
            ohe,
            columns=percent_columns,
        )
        if self.drop_first:
            percent_df = percent_df.drop(columns=[f"{temp_column}_0"])
            percent_columns = percent_columns[1:]

        df = pd.concat([df, percent_df], axis=1)
        df = df.copy()
        df[percent_columns] = (
            df.groupby(by=["subscriptionid"])[percent_columns]
            .transform(custom_transform)
            .astype(int)
        )
        # df[percent_columns] = (
        #     df.groupby(by=["subscriptionid"])[percent_columns]
        #     .transform(lambda d: d.shift(fill_value=False))
        #     .astype(bool)
        # )
        # df[percent_columns] = (
        #     df.groupby(by=["subscriptionid"])[percent_columns]
        #     .transform(lambda d: d | d.shift())
        #     .astype(int)
        # )
        df = df.drop(columns=[temp_column])
        return df

    def __repr__(self) -> str:
        return f"""BucketSubscriptionCPUPercentTransform(
                version=2,
                target_name="{self.target_name}",
                drop_first={self.drop_first},
                n_bins={self.n_bins},
                drop_first={self.drop_first}
            )"""


class BucketSubscriptionCPUTransform(BaseTransform):
    def __init__(
        self,
        target_name: str,
        n_bins: int | None = 4,
        drop_first: bool = False,
    ):
        self.target_name = target_name
        self.n_bins = n_bins if n_bins is not None else 4
        self.drop_first = drop_first

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert an iterable of indices to one-hot encoded labels."""

        df = data.copy()
        temp_column = f"{self.target_name}_temp"
        df[temp_column] = discretize_column(df[self.target_name], self.n_bins)

        # one hot encoding but with number of class
        targets = np.array(df[temp_column].values).reshape(-1)
        ohe = np.eye(self.n_bins)[targets]
        percent_columns = [f"{temp_column}_{i}" for i in range(0, self.n_bins)]
        percent_df = pd.DataFrame(
            ohe,
            columns=percent_columns,
        )
        if self.drop_first:
            percent_df = percent_df.drop(columns=[f"{temp_column}_0"])
            percent_columns = percent_columns[1:]

        df = pd.concat([df, percent_df], axis=1)
        df[percent_columns] = (
            df.groupby(by=["subscriptionid"])[percent_columns]
            .transform(custom_transform)
            .astype(int)
        )
        df = df.drop(columns=[temp_column])
        return df

    def __repr__(self) -> str:
        return f"""BucketSubscriptionCPUTransform(
                target_name="{self.target_name}",
                drop_first={self.drop_first},
                n_bins={self.n_bins},
                drop_first={self.drop_first}
            )"""


class HistoricalBucketCPUTransform(BaseTransform):
    def __init__(
        self,
        target_name: str,
        n_bins: int | None = 4,
    ):
        self.target_name = target_name
        self.n_bins = n_bins if n_bins is not None else 4

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert an iterable of indices to one-hot encoded labels."""
        data = data.copy()
        targets = np.array(data[self.target_name].values).reshape(-1)
        ohe = np.eye(self.n_bins)[targets]
        percent_columns = [
            f"{self.target_name}_{i}" for i in range(0, self.n_bins)
        ]
        percent_df = pd.DataFrame(
            ohe,
            columns=percent_columns,
        )
        data = pd.concat([data, percent_df], axis=1)
        data[percent_columns] = data[percent_columns].astype(int)
        data[percent_columns] = data[percent_columns].cumsum()
        data[f"{self.target_name}_total"] = data[percent_columns].sum(axis=1)
        data[percent_columns] = data[percent_columns].div(
            data[f"{self.target_name}_total"], axis=0
        )
        data = data.drop(columns=[f"{self.target_name}_total"])
        return data

    def __repr__(self) -> str:
        return f"""HistoricalBucketCPUTransform(
                target_name="{self.target_name}",
                n_bins={self.n_bins},
            )"""


class HistoricalBucketCPUEachSubscriptionTransform(BaseTransform):
    def __init__(
        self,
        target_name: str,
        n_bins: int | None = 4,
    ):
        self.target_name = target_name
        self.n_bins = n_bins if n_bins is not None else 4

    def custom_apply_fn(self, percent_columns: list[str]):
        def fun(df: pd.DataFrame) -> pd.DataFrame:
            df[percent_columns] = df[percent_columns].cumsum()
            df["percent_total"] = df[percent_columns].sum(axis=1)
            df[percent_columns] = df[percent_columns].div(
                df["percent_total"], axis=0
            )
            df = df.drop(columns=["percent_total"])
            return df

        return fun

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert an iterable of indices to one-hot encoded labels."""
        data = data.copy()
        targets = np.array(data[self.target_name].values).reshape(-1)
        ohe = np.eye(self.n_bins)[targets]
        percent_columns = [
            f"{self.target_name}_{i}" for i in range(0, self.n_bins)
        ]
        percent_df = pd.DataFrame(
            ohe,
            columns=percent_columns,
        )
        data = pd.concat([data, percent_df], axis=1)
        data[percent_columns] = data[percent_columns].astype(int)
        data = (
            data.groupby(by=["subscriptionid"])
            .apply(self.custom_apply_fn(percent_columns))
            .reset_index(drop=True)
        )
        return data

    def __repr__(self) -> str:
        return f"""HistoricalBucketCPUEachSubscriptionTransform(
                target_name="{self.target_name}",
                n_bins={self.n_bins},
            )"""


class GroupByDayTransform(BaseTransform):
    def __init__(
        self,
        timestamp_col: str = "vmcreated",
        day_col: str = "day",
        sort: bool = False,
    ) -> None:
        super().__init__()
        self.timestamp_col = timestamp_col
        self.day_col = day_col
        self.sort = sort

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data[self.day_col] = np.floor(data[self.timestamp_col] / (24 * 60 * 60))
        data[self.day_col] = data[self.day_col].astype(int)
        if self.sort:
            data = data.sort_values(by=[self.day_col]).reset_index(drop=True)
        return data

    def __repr__(self) -> str:
        return f"GroupByDayTransform(sort={self.sort})"


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


class NoFeats(BaseFeatureEngineering):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = list(
            filter(lambda x: x != config.dataset.target, NON_FEATURE_COLUMNS)
        )

    @property
    def preprocess_transform_set(self) -> list[Transform] | None:
        return [
            ColumnsDropTransform(columns=self._non_feature_columns),
            DiscretizeColumnTransform(
                column=self._config.dataset.target,
                new_column=self._target_name,
                n_bins=self._config.dataset.num_classes,
            ),
        ]

    @property
    def target_name(self) -> str:
        return self._target_name

    def __repr__(self) -> str:
        return "NoFeats()"


class FeatureEngineering_A(BaseFeatureEngineering):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = list(
            filter(
                # lambda x: x != config.dataset.target and x != "DIST_COL",
                lambda x: x != config.dataset.target,
                NON_FEATURE_COLUMNS,
            )
        )

    @property
    def preprocess_transform_set(self) -> list[Transform] | None:
        def scaler_transform(data: pd.DataFrame) -> pd.DataFrame:
            scaler = StandardScaler()
            # Select the columns to be scaled (excluding "bucket_util_cpu")
            columns_to_scale = [
                col for col in data.columns if col != self.target_name
            ]

            # Scale the selected columns
            data[columns_to_scale] = scaler.fit_transform(
                data[columns_to_scale]
            )
            return data

        return [
            PrintColumnsTransform("Original Columns"),
            # lambda data: data.copy(),
            NamedInjectTransform(DD_ID),
            BucketSubscriptionCPUPercentTransform_V1(
                self._config.dataset.target
            ),
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
                n_bins=self._config.dataset.num_classes,
                drop_original=True,
            ),
            scaler_transform,
            PrintColumnsTransform("Final Columns"),
        ]

    @property
    def target_name(self) -> str:
        return self._target_name


class FeatureEngineering_B(BaseFeatureEngineering):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = list(
            filter(
                # lambda x: x != config.dataset.target and x != "DIST_COL",
                lambda x: x != config.dataset.target,
                NON_FEATURE_COLUMNS,
            )
        )

    @property
    def preprocess_transform_set(self) -> list[Transform] | None:
        def scaler_transform(data: pd.DataFrame) -> pd.DataFrame:
            scaler = StandardScaler()
            # Select the columns to be scaled (excluding "bucket_util_cpu")
            columns_to_scale = [
                col for col in data.columns if col != self.target_name
            ]

            # Scale the selected columns
            data[columns_to_scale] = scaler.fit_transform(
                data[columns_to_scale]
            )
            return data

        return [
            PrintColumnsTransform("Original Columns"),
            # lambda data: data.copy(),
            NamedInjectTransform(DD_ID),
            BucketSubscriptionCPUPercentTransform_V2(
                self._config.dataset.target,
                n_bins=100,
                drop_first=False,
            ),
            # BucketSubscriptionCPUTransform(
            #     self._config.dataset.target,
            #     n_bins=self._config.dataset.num_classes,
            #     drop_first=False,
            # ),
            OneHotColumnsTransform(
                columns=[
                    "vmcategory",
                    "vmcorecountbucket",
                    "vmmemorybucket",
                ],
                drop_first=True,
            ),
            NamedInjectTransform("day"),
            ColumnsDropTransform(columns=self._non_feature_columns),
            DiscretizeColumnTransform(
                column=self._config.dataset.target,
                new_column=self.target_name,
                n_bins=self._config.dataset.num_classes,
                drop_original=True,
            ),
            # scaler_transform,
            PrintColumnsTransform("Final Columns"),
        ]

    @property
    def target_name(self) -> str:
        return self._target_name


class FeatureEngineering_C(BaseFeatureEngineering):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = list(
            filter(
                # lambda x: x != config.dataset.target and x != "DIST_COL",
                lambda x: x != config.dataset.target,
                NON_FEATURE_COLUMNS,
            )
        )

    @property
    def preprocess_transform_set(self) -> list[Transform] | None:
        def scaler_transform(data: pd.DataFrame) -> pd.DataFrame:
            scaler = StandardScaler()
            # Select the columns to be scaled (excluding "bucket_util_cpu")
            columns_to_scale = [
                col for col in data.columns if col != self.target_name
            ]

            # Scale the selected columns
            data[columns_to_scale] = scaler.fit_transform(
                data[columns_to_scale]
            )
            return data

        def ceil_percent(data: pd.DataFrame) -> pd.DataFrame:
            data[self.target_name] = data[self._config.dataset.target].apply(
                np.ceil
            )
            data = data.astype({self.target_name: int})
            return data

        def clean_df(data: pd.DataFrame) -> pd.DataFrame:
            df = data.copy()
            first_bucket = df[df.p95maxcpu <= 20]
            second_bucket = df[(df.p95maxcpu > 27) & (df.p95maxcpu <= 47)]
            third_bucket = df[(df.p95maxcpu >= 50) & (df.p95maxcpu <= 73)]
            fourth_bucket = df[df.p95maxcpu >= 78]
            # display(fourth_bucket)
            return pd.concat(
                [first_bucket, second_bucket, third_bucket, fourth_bucket]
            ).reset_index(drop=True)

        def group_by_deployment(data: pd.DataFrame) -> pd.DataFrame:
            dfs = []
            count = 0
            grouped_df = data.groupby(by=["deploymentid"])
            for i, g in grouped_df:
                if count == 1000:
                    break
                if len(g) < 50:
                    continue
                test_df = g.copy().reset_index(drop=True)
                dfs.append(test_df)
                count += 1
            return pd.concat(dfs).reset_index(drop=True)

        return [
            PrintColumnsTransform("Original Columns"),
            lambda data: clean_df(data),
            lambda data: ceil_percent(data),
            lambda data: group_by_deployment(data),
            lambda data: data.copy()
            .sort_values(
                by=["vmcreated", "vmdeleted", "deploymentid", "subscriptionid"]
            )
            .reset_index(drop=True),
            NamedInjectTransform(DD_ID),
            DiscretizeColumnTransform(
                column=self._config.dataset.target,
                new_column=self.target_name,
                n_bins=self._config.dataset.num_classes,
                drop_original=True,
            ),
            # lambda data: data.copy(),
            HistoricalBucketCPUEachSubscriptionTransform(
                self.target_name,
                n_bins=4,
            ),
            # BucketSubscriptionCPUTransform(
            #     self._config.dataset.target,
            #     n_bins=self._config.dataset.num_classes,
            #     drop_first=False,
            # ),
            OneHotColumnsTransform(
                columns=[
                    "vmcategory",
                    "vmcorecountbucket",
                    "vmmemorybucket",
                ],
                drop_first=True,
            ),
            ColumnsDropTransform(columns=self._non_feature_columns),
            # NamedInjectTransform("day"),
            # scaler_transform,
            PrintColumnsTransform("Final Columns"),
            # lambda data: data.iloc[0:300_000],
        ]

    @property
    def target_name(self) -> str:
        return self._target_name


__all__ = [
    "BucketSubscriptionCPUPercentTransform_V1",
    "BucketSubscriptionCPUPercentTransform_V2",
    "GroupByDayTransform",
    "FeatureEngineering_A",
    "FeatureEngineering_B",
    "FeatureEngineering_C",
]
