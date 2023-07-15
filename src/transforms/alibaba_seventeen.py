import pandas as pd

from src.helpers.config import Config

from .base import BaseFeatureEngineering, BaseTransform, Transform
from .general import ColumnsDropTransform, DiscretizeColumnTransform


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
        return "CleanDataTransform()"


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
        # data[f'count_{self.column}'] =
        #   data[self.column].str.split(self.sep).str.len()
        data[self.new_column] = data[self.column].apply(
            lambda x: len(x.split(","))
        )
        return data

    def __repr__(self) -> str:
        return (
            f"StrCountTransform(column={self.column}"
            + f" sep={self.sep}, new_column={self.new_column})"
        )


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


class FeatureEngineering_A(BaseFeatureEngineering):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = list(
            filter(
                lambda x: x != config.dataset.target,
                NON_FEATURE_COLUMNS,
            )
        )

    @property
    def preprocess_transform_set(self) -> list[Transform] | None:
        return [
            CleanDataTransform(),
            ColumnsDropTransform(
                columns=self._non_feature_columns + ["cpu_set"]
            ),
            DiscretizeColumnTransform(
                column=self._config.dataset.target,
                new_column=self._target_name,
            ),
        ]

    @property
    def target_name(self) -> str:
        return self._target_name


__all__ = [
    "CleanDataTransform",
    "StrCountTransform",
    "NON_FEATURE_COLUMNS",
    "FeatureEngineering_A",
]
