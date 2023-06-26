import pandas as pd
from sklearn.preprocessing import minmax_scale

from src.utils.general import append_prev_feature, discretize_column

from .base import BaseTransform


class CleanDataTransform(BaseTransform):
    def __init__(self, exclude: str | list[str] = []):
        if isinstance(exclude, str):
            exclude = [exclude]
        self.excludes = exclude

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        # non_feature_columns = filter(
        #     lambda x: x in data.columns and x not in self.excludes,
        #     self.NON_FEATURE_COLUMNS,
        # )
        data = data[
            data.plan_cpu.notna()
            & data.plan_mem.notna()
            & data.plan_disk.notna()
            & (data.cpu_util_percent > 0)
            & (data.cpu_util_percent <= 100)
        ]
        data = data.dropna()
        # data = data[(data.plan_cpu > 0) & (data.plan_mem > 0)]
        data = data.sort_values(by=["time_stamp"])
        data = data.drop(columns=self.excludes)
        data = data.reset_index(drop=True)
        return data

    def __repr__(self) -> str:
        return f"CleanDataTransform()"


class ColumnsDropTransform(BaseTransform):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop(columns=self.columns)
        data = data.reset_index(drop=True)
        return data

    def __repr__(self) -> str:
        return f"ColumnsDropTransform(columns={self.columns})"


class DiscretizeColumnTransform(BaseTransform):
    def __init__(
        self,
        column: str,
        new_column: str | None = None,
        n_bins: int = 4,
    ):
        self.column = column
        self.n_bins = n_bins
        self.new_column = column if new_column is None else new_column

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        column_data = minmax_scale(data[self.column])
        data[self.new_column] = discretize_column(
            column_data, self.n_bins
        )
        return data

    def __repr__(self) -> str:
        return f"DiscretizeColumnTransform(n_bins={self.n_bins})"


class EnumColumnTransform(BaseTransform):
    def __init__(self, column: str, new_column: str | None = None):
        self.column = column
        self.new_column = column if new_column is None else new_column

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.new_column] = (
            data[self.column].astype("category").cat.codes
        )
        return data


class AppendPrevFeatureTransform(BaseTransform):
    def __init__(self, columns: list[str], n_historical: int = 4):
        self.n_historical = n_historical
        self.columns = columns

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            append_prev_feature(data, self.n_historical, column)
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data

    def __repr__(self) -> str:
        return f"Feats_A_Transform(n_historical={self.n_historical})"


class StrCountTransform(BaseTransform):
    def __init__(
        self,
        column: str,
        sep: str = ",",
        new_column: str | None = None,
    ):
        self.column = column
        self.sep = sep
        self.new_column = column if new_column is None else new_column

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        # data[f'count_{self.column}'] = data[self.column].str.split(self.sep).str.len()
        data[self.new_column] = data[self.column].apply(
            lambda x: len(x.split(","))
        )
        return data

    def __repr__(self) -> str:
        return f"StrCountTransform(column={self.column}, sep={self.sep}, new_column={self.new_column})"


NON_FEATURE_COLUMNS = [
    "time_stamp",
    # usage
    "instance_id",
    "cpu_util_percent",
    "mem_util_percent",
    "disk_util_percent",
    "avg_cpu_1_min",
    "avg_cpu_5_min",
    "avg_cpu_15_min",
    "avg_cpi",
    "avg_cache_miss",
    "max_cpi",
    "max_cache_miss",
    # event
    "event_type",
    # "machine_id",
    # "plan_cpu",
    # "plan_mem",
    # "plan_disk",
    # "cpu_set",
]


__all__ = [
    "CleanDataTransform",
    "DiscretizeOutputTransform",
    "AppendPrevFeatureTransform",
    "ColumnDropperTransform",
    "StrCountTransform",
    "EnumColumnTransform",
    "NON_FEATURE_COLUMNS",
]
