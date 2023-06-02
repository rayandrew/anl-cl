from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.utils.general import custom_round


def main(args):
    mu_path = Path(args.mu_data)
    bi_path = Path(args.bi_data)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    bi = pd.read_csv(bi_path)
    bi = bi[bi["cpu_avg"].notna() & bi["cpu_max"].notna()]
    bi["start_time_ins"] = bi["start_time_ins"].apply(
        lambda x: custom_round(x, 10)
    )
    bi["end_time_ins"] = bi["end_time_ins"].apply(
        lambda x: custom_round(x, 10)
    )
    bi = bi.sort_values("start_time_ins")

    mu = pd.read_csv(
        mu_path,
        names=[
            "machine_id",
            "time_stamp",
            "cpu_util_percent",
            "mem_util_percent",
            "mem_gps",
            "mpki",
            "net_in",
            "net_out",
            "disk_io_percent",
        ],
    )

    mrg = pd.merge_asof(
        mu,
        bi,
        left_on="time_stamp",
        right_on="start_time_ins",
        direction="backward",
    ).query("time_stamp <= end_time_ins")

    grouped = (
        mrg[
            [
                # "machine_id_x",
                # "instance_name",
                # "task_name",
                "time_stamp",
                "cpu_avg",
                "cpu_max",
                "mem_avg",
                "mem_max",
                "plan_cpu",
                "plan_mem",
                "cpu_util_percent",
                "mem_util_percent",
                "disk_io_percent",
                # "mpki",
            ]
        ]
        .groupby(by=["time_stamp"])
        # .agg(["sum", "count"])
        .sum()
        .reset_index()
        .sort_values("time_stamp")
    )

    grouped.to_csv(output_path / f"{mu_path.stem}.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="data",
        description="",
        epilog="",
    )

    parser.add_argument("--mu-data", required=True, type=str)
    parser.add_argument("--bi-data", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    main(args)
