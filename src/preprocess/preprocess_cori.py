import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Union

if TYPE_CHECKING:
    snakemake: Any = None


import joblib
import pandas as pd
import semver
import sklearn
from sklearn.preprocessing import OneHotEncoder


class DatasetPreprocessing(object):
    START_TIME_START = "2017-03-24 00:00:00"
    START_TIME_END = "2017-08-10 00:00:00"

    # APP_NAMES = ["dbscan", "hacc", "ior", "vpicio"]

    # ACTION_NAMES = ["read", "write"]

    # FPP_SH_NAMES = ["fpp", "shared"]

    ONE_HOT_COLUMNS = [
        "APP_name",
        "darshan_read_or_write_job",
        "darshan_fpp_or_ssf_job",
    ]

    def __init__(
        self,
        directory: Union[Path, str],
        encode_collectively: bool = False,
    ) -> None:
        self._directory = Path(directory).absolute()
        self._data: dict[str, pd.DataFrame] = {}
        self._encode_collectively = encode_collectively

        if self._encode_collectively:
            # collective
            self._encoder = OneHotEncoder()
        else:
            # separate
            self._encoder = {
                col: OneHotEncoder() for col in self.ONE_HOT_COLUMNS
            }

        self._load_data()
        self._agg_data()
        self._encode_data()

    def _load_data(self):
        for csv_file in glob.glob(
            str(self._directory.joinpath("*.csv"))
        ):
            df = pd.read_csv(csv_file)
            assert (
                "_datetime_start" in df and "_datetime_end" in df
            ), f"no `_datetime_start` and `_datetime_end` columns found in: {csv_file}"

            key = Path(csv_file).stem

            # (
            #     _,
            #     _,
            #     _,
            #     _,
            #     _,
            #     _,
            #     _system,
            #     app_name,
            #     rw,
            #     sh_fpp,
            # ) = key.split("_")

            df._datetime_start = pd.to_datetime(df._datetime_start)
            df._datetime_end = pd.to_datetime(df._datetime_end)

            self._data[key] = df

    def _agg_data(self):
        self._all_data = pd.concat(
            [self._data[key] for key in self._data]
        )

        self._all_data._datetime_start = pd.to_datetime(
            self._all_data._datetime_start
        )
        self._all_data._datetime_end = pd.to_datetime(
            self._all_data._datetime_end
        )

        # before upgrade maintenance
        self._all_data.loc[
            (
                self._all_data["_datetime_start"]
                < self.START_TIME_START
            ),
            "label",
        ] = 1

        # first upgrade maintenance
        self._all_data.loc[
            (
                self._all_data["_datetime_start"]
                >= self.START_TIME_START
            )
            & (
                self._all_data["_datetime_start"]
                < self.START_TIME_END
            ),
            "label",
        ] = 2

        # second upgrade maintenance
        self._all_data.loc[
            (
                self._all_data["_datetime_start"]
                >= self.START_TIME_END
            ),
            "label",
        ] = 3

        # self._all_data = self._all_data[
        #     (
        #         self._all_data["_datetime_start"]
        #         <= "2017-03-24 00:00:00"
        #     )
        # ]

        self._all_data["label"] = self._all_data["label"].astype(
            "int"
        )

    @staticmethod
    def one_hot_encode(
        encoder: OneHotEncoder,
        data: pd.DataFrame,
        columns: Union[str, Sequence[str]],
    ):
        if isinstance(columns, str):
            columns = [columns]

        sk_ver = semver.VersionInfo.parse(sklearn.__version__)
        encoder.fit(data[columns])
        app_name_columns = (
            encoder.get_feature_names(columns)
            if sk_ver.compare("1.0.0") < 0
            else encoder.get_feature_names_out(columns)
        )
        app_name_transformed = encoder.transform(
            data[columns].to_numpy().reshape(-1, len(columns))
        )
        return pd.DataFrame(
            app_name_transformed.toarray(),
            columns=app_name_columns,
        ).astype("int")

    def _encode_data(self):
        # we need to one hot encode 3 columns: app_name, read/write, fpp_shared
        sk_ver = semver.VersionInfo.parse(sklearn.__version__)

        # TODO: ask Sandeep, better to create collective one hot encode or not

        if self._encode_collectively:
            # collective
            encode_df = DatasetPreprocessing.one_hot_encode(
                self._encoder,
                self._all_data,
                DatasetPreprocessing.ONE_HOT_COLUMNS,
            )
        # seperate
        else:
            encode_df = [
                DatasetPreprocessing.one_hot_encode(
                    self._encoder[col],
                    self._all_data,
                    col,
                )
                for col in self._encoder
            ]

        # concat all data
        self._all_data = self._all_data.reset_index()
        self._all_data = pd.concat(
            [self._all_data]
            + (
                [encode_df]
                if self._encode_collectively
                else encode_df
            ),
            axis=1,
        )

    def save(self, path: Union[Path, str]):
        assert not self._all_data.empty, "data is not loaded"
        # assert path.is_dir(), "path must be directory"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._all_data.to_csv(path.joinpath("data.csv"))
        joblib.dump(self._encoder, path.joinpath("encoder.joblib"))

    @property
    def data(self):
        assert not self._all_data.empty, "data is not loaded"
        return self._all_data


if __name__ == "__main__":
    d = DatasetPreprocessing(snakemake.config)
    d.save("data/full-data-grouped-3-labels")

    pd.set_option("display.max_columns", None)
    # print (df.columns)
    print(
        d.data[
            [
                "_benchmark_id",
                "APP_name_dbscan",
                "APP_name_hacc",
                "APP_name_ior",
                "APP_name_vpicio",
                "darshan_read_or_write_job_read",
                "darshan_read_or_write_job_write",
                "darshan_fpp_or_ssf_job_fpp",
                "darshan_fpp_or_ssf_job_shared",
            ]
        ].head()
    )

    # print(1, len(d.data[d.data["label"] == 1]))
    # print(2, len(d.data[d.data["label"] == 2]))

    print(d.data.groupby(["label"]).size())
