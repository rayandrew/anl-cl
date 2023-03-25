from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

MACHINE_DISTS = {
    "m_25": [219_200],
    "m_881": [462_500, 522_000, 585_000],
}

def main(input_path, machine_id):
    input_path = Path(input_path)
    filename = input_path.stem

    df = pd.read_csv(input_path, names=[
        "machine_id",
        "time_stamp",
        "cpu_util_percent",
        "mem_util_percent",
        "mem_gps",
        "mkpi",
        "net_in",
        "net_out",
        "disk_util_percent",
    ])
    distributions = MACHINE_DISTS[machine_id]
    dist_obs_length = len(distributions)
    if dist_obs_length >= 2:
        for (i, ts) in enumerate(reversed(distributions)):
            if i == 0:
                df["dist_label"] = dist_obs_length - i - 1
                continue
            df.loc[df["time_stamp"] <= ts, "dist_label"] = (
                dist_obs_length - i - 1
            )
        df["dist_label"] = df["dist_label"].astype(int)
    elif dist_obs_length == 1:
        df["dist_label"] = 1
        df.loc[df["time_stamp"] <= distributions[0], "dist_label"] = 0

    df.to_csv(f"{filename}.csv", index=False, header=False)

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="data",
        description="",
        epilog="",
    )

    parser.add_argument("input_path")
    parser.add_argument("-m", "--machine", required=True)
    args = parser.parse_args()

    main(args.input_path, args.machine)