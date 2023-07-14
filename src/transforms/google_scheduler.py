from typing import Any

import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

from src.helpers.config import Config

from .base import BaseTransform, Transform, apply_transforms
from .general import ColumnsDropTransform


class CleanDataTransform(BaseTransform):
    def __init__(self, exclude: str | list[str] = []):
        if isinstance(exclude, str):
            exclude = [exclude]
        self.excludes = exclude

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.dropna(subset=["util_cpu", "cpu_95"])
        data = data.sort_values(by=["start_time"])
        data = data.drop(columns=self.excludes)
        data = data.head(200000)
        data = data.reset_index(drop=True)
        return data

    def __repr__(self) -> str:
        return "CleanDataTransform()"


class ClassifyThrottleTransform(BaseTransform):
    def __init__(
        self, exclude: str | list[str] = [], new_column: str = "bucket_util_cpu"
    ):
        if isinstance(exclude, str):
            exclude = [exclude]
        self.excludes = exclude
        self.target_name_new = new_column

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        print(len(data))
        bin_edges = [
            -float("inf"),
            1,
            float("inf"),
        ]
        data[self.target_name_new] = pd.cut(
            data["util_cpu"], bins=bin_edges, labels=False
        )
        return data

    def __repr__(self) -> str:
        return "ClassifyThrottleTransform()"


class DurationHistoryTransform(BaseTransform):
    def __init__(
        self,
        exclude: str | list[str] = [],
        colname: str = "",
        dur_cutoff: int = 412000000,
    ):
        if isinstance(exclude, str):
            exclude = [exclude]
        self.colname = colname
        self.dur_cutoff = dur_cutoff
        self.excludes = exclude

    def __call__(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """colname = column to group the history on
        This function will map jobs with
            duration >= ~6 minutes as 1, otherwise 0.
        Used to add duration information to a
            job indirectly through historical data.
        """

        long_duration_name = str(self.colname) + "_history_duration_long"
        short_duration_name = str(self.colname) + "_history_duration_short"
        data[long_duration_name] = 0
        data[short_duration_name] = 0

        histogram_map: dict[str, Any] = {}

        for index, row in data.iterrows():
            group_name = row[self.colname]
            duration_classification = (
                1
                if (row["end_time"] - row["start_time"]) >= self.dur_cutoff
                else 0
            )

            group_hist = histogram_map.get(group_name, {"long": 0, "short": 0})

            total_rows = max(
                group_hist["long"] + group_hist["short"],
                1,
            )

            data.at[index, long_duration_name] = group_hist["long"] / total_rows
            data.at[index, short_duration_name] = (
                group_hist["short"] / total_rows
            )

            if duration_classification == 1:
                group_hist["long"] += 1
            else:
                group_hist["short"] += 1

            # Update the dictionary with the modified histogram
            # for the collection_logical_name
            histogram_map[group_name] = group_hist
        return data

    def __repr__(self) -> str:
        return "DurationHistoryTransform()"


class ThrottleHistoryTransform(BaseTransform):
    def __init__(self, exclude: str | list[str] = [], colname: str = ""):
        if isinstance(exclude, str):
            exclude = [exclude]
        self.excludes = exclude
        self.colname = colname

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        colname = column name to group on
        Will make a history of throttled/non-throttled jobs based on colname
        """
        throttle_name = str(self.colname) + "_history_throttle"
        non_throttle_name = str(self.colname) + "_history_non_throttle"
        print(len(data))

        data[throttle_name] = 0
        data[non_throttle_name] = 0

        histogram_map: dict[str, Any] = {}

        for index, row in data.iterrows():
            collection_name = row[self.colname]
            cpu_classification = 1 if row["util_cpu"] > 1 else 0

            collection_hist = histogram_map.get(
                collection_name, {"throttle": 0, "non_throttle": 0}
            )

            total_rows = max(
                collection_hist["throttle"] + collection_hist["non_throttle"],
                1,
            )
            data.at[index, throttle_name] = (
                collection_hist["throttle"] / total_rows
            )
            data.at[index, non_throttle_name] = (
                collection_hist["non_throttle"] / total_rows
            )

            if cpu_classification == 1:
                collection_hist["throttle"] += 1
            else:
                collection_hist["non_throttle"] += 1

            # Update the dictionary with the modified histogram
            # for the collection_logical_name
            histogram_map[collection_name] = collection_hist
        return data

    def __repr__(self) -> str:
        return "ThrottleHistoryTransform()"


FEATURE_COLUMNS = [
    "sched_class",
    # "collection_max_per_machine",
    # "collection_max_per_switch",
    "collection_vertical_scaling",
    "collection_scheduler",
    "priority",
    "req_cpu",
    "req_mem",
    "constraint_mapped",
    "collection_logical_name_mapped",
    "constraint_mapped_history_throttle",
    "collection_logical_name_mapped_history_throttle",
    "constraint_mapped_history_non_throttle",
    "collection_logical_name_mapped_history_non_throttle",
    "constraint_mapped_history_duration_long",
    "collection_logical_name_mapped_history_duration_long",
    "constraint_mapped_history_duration_short",
    "collection_logical_name_mapped_history_duration_short",
]

ALL_FEATURE_COLUMNS = [
    "machine_id",
    "start_time",
    "end_time",
    "collection_id",
    "instance_index",
    "alloc_collection_id",
    "cpu_95",
    "sched_class",
    "priority",
    "req_cpu",
    "req_mem",
    "collection_logical_name",
    "collection_max_per_machine",
    "collection_max_per_switch",
    "collection_vertical_scaling",
    "collection_scheduler",
    "req_constraint",
    "machine_cpu_cap",
    "machine_mem_cap",
    "util_cpu",
    "collection_logical_name_mapped",
    "constraint_str",
    "constraint_mapped",
]


class Baseline_TransformSet(BaseTransform):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = [
            feature
            for feature in ALL_FEATURE_COLUMNS
            if feature not in FEATURE_COLUMNS
        ]
        self._transforms: list[Transform] = [
            CleanDataTransform(),
            ClassifyThrottleTransform(new_column=self._target_name),
            ColumnsDropTransform(columns=self._non_feature_columns),
        ]

    @property
    def target_name(self) -> str:
        return self._target_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return apply_transforms(data, self._transforms)

    def __repr__(self) -> str:
        return "Baseline_TransformSet()"


class FeatureA_TransformSet(BaseTransform):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = [
            feature
            for feature in ALL_FEATURE_COLUMNS
            if feature not in FEATURE_COLUMNS
        ]
        self._transforms: list[Transform] = [
            CleanDataTransform(),
            ClassifyThrottleTransform(new_column=self._target_name),
            ThrottleHistoryTransform(colname="collection_logical_name_mapped"),
            ThrottleHistoryTransform(colname="constraint_mapped"),
            ColumnsDropTransform(columns=self._non_feature_columns),
        ]

    @property
    def target_name(self) -> str:
        return self._target_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        print("disini jg")
        data = apply_transforms(data, self._transforms)
        print(data.columns)
        scaler = StandardScaler()
        # Select the columns to be scaled (excluding "bucket_util_cpu")
        columns_to_scale = [
            col for col in data.columns if col != self._target_name
        ]

        # Scale the selected columns
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
        return data

    def __repr__(self) -> str:
        return "FeatureA_TransformSet()"


class FeatureB_TransformSet(BaseTransform):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._target_name = f"bucket_{config.dataset.target}"
        self._non_feature_columns = [
            feature
            for feature in ALL_FEATURE_COLUMNS
            if feature not in FEATURE_COLUMNS
        ]
        self._transforms: list[Transform] = [
            CleanDataTransform(),
            ClassifyThrottleTransform(new_column=self._target_name),
            ThrottleHistoryTransform(colname="collection_logical_name_mapped"),
            ThrottleHistoryTransform(colname="scheduler_mapped"),
            DurationHistoryTransform(colname="collection_logical_name_mapped"),
            DurationHistoryTransform(colname="scheduler_mapped"),
            ColumnsDropTransform(columns=self._non_feature_columns),
        ]

    @property
    def target_name(self) -> str:
        return self._target_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return apply_transforms(data, self._transforms)

    def __repr__(self) -> str:
        return "FeatureB_TransformSet()"


__all__ = [
    "ClassifyThrottleTransform",
    "CleanDataTransform",
    "DurationHistoryTransform",
    "ThrottleHistoryTransform",
    "Baseline_TransformSet",
    "FeatureA_TransformSet",
    "FeatureB_TransformSet",
]
