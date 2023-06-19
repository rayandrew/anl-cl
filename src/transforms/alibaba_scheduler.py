import pandas as pd
from sklearn.preprocessing import minmax_scale

from src.utils.general import append_prev_feature, discretize_column

from .base import BaseTransform


class CleanDataTransform(BaseTransform):
    NON_FEATURE_COLUMNS = [
        "name",
        # "task_type",
        "status",
        "start_time",
        "end_time",
        # "instance_num",
        # "plan_cpu",
        # "plan_mem",
        "instance_name",
        # "instance_name.1",
        "instance_start_time",
        "instance_end_time",
        "machine_id",
        "seq_no",
        "total_seq_no",
        # "instance_name",
        "cpu_avg",
        "cpu_max",
        "mem_avg",
        "mem_max",
    ]

    def __init__(self, target: str):
        self.target = target

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        non_feature_columns = filter(
            lambda x: x in data.columns and x != self.target,
            self.NON_FEATURE_COLUMNS,
        )
        data = data.fillna(0)
        data = data[(data.plan_cpu > 0) & (data.plan_mem > 0)]
        data = data.sort_values(by=["start_time"])
        data = data.drop(columns=non_feature_columns)
        return data

    def __repr__(self) -> str:
        return (
            f"CleanDataTransform(output_column={self.output_column})"
        )


class DiscretizeOutputTransform(BaseTransform):
    def __init__(self, target: str, n_bins: int = 4):
        self.target = target
        self.n_bins = n_bins

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        target_data = minmax_scale(data[self.target])
        data[self.target] = discretize_column(
            target_data, self.n_bins
        )
        return data

    def __repr__(self) -> str:
        return f"DiscretizeOutputTransform(n_bins={self.n_bins})"


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


__all__ = [
    "CleanDataTransform",
    "DiscretizeOutputTransform",
    "AppendPrevFeatureTransform",
]
