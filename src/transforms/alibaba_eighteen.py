import pandas as pd

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

    def __init__(self, exclude: str | list[str]):
        if isinstance(exclude, str):
            exclude = [exclude]
        self.excludes = exclude

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        non_feature_columns = filter(
            lambda x: x in data.columns and x not in self.excludes,
            self.NON_FEATURE_COLUMNS,
        )
        data = data.dropna()
        data = data.reset_index(drop=True)
        # data = data.fillna(0)
        data = data[(data.plan_cpu > 0) & (data.plan_mem > 0)]
        data = data.sort_values(by=["start_time"])
        data = data.drop(columns=non_feature_columns)
        return data

    def __repr__(self) -> str:
        return "CleanDataTransform()"


__all__ = [
    "CleanDataTransform",
]
