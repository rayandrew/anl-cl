from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    snakemake: Any = None


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.general import head, set_seed


def plot(stream_window: Sequence[int], change: Sequence[int], path: Path, config: dict = {}):
    y = stream_window
    x = [i for i in range(len(y))]

    plt.figure(figsize=(30, 6))

    # Plot the data
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title(f"Stream Data Plot {config}")

    plt.tight_layout()

    # Add vertical lines
    index = [i for i in change]
    for i in index:
        plt.axvline(x=i, color="red", linestyle="--")

    # Show the plot
    plt.savefig(f"{path}")


def add_dist_label(data: pd.DataFrame, dist: Sequence[int], start_from=0):
    distributions = np.zeros(len(data), dtype=int)

    print(f"Dist from 0 to {dist[0]}: 0")
    distributions[: dist[0]] = start_from
    for i in range(len(dist) - 1):
        print(f"Dist from {dist[i]} to {dist[i+1]}: {i+1}")
        distributions[dist[i] : dist[i + 1]] = i + 1 + start_from
    # print(
    #     f"Dist from {dist[len(dist) - 1]} to {dist[-1]}: {len(dist) - 1}"
    # )
    distributions[dist[-1] :] = len(dist)  + start_from 

    new_data = data.copy()
    new_data["dist_label"] = pd.Series(distributions)

    return new_data

def get_specialized_dd_config(cfg: dict, method: str):
    if "drift_detection" in cfg:
        # TODO: specialized file in single dataset
        # placeholder

        # specialized in single dataset
        if method in cfg["drift_detection"]:
            return cfg["drift_detection"][method]

    return None


def main():
    set_seed(snakemake.config.get("seed", 0))
    input_path = Path(str(snakemake.input))
    orig_data = pd.read_csv(input_path)
    dataset_config = snakemake.params.dataset_config
    dd_config = get_specialized_dd_config(dataset_config, snakemake.params.method)

    output_path = Path(str(snakemake.output))
    output_path.mkdir(parents=True, exist_ok=True)

    base_output_filename = f"{input_path.stem}_{dataset_config['y']}"

    data = orig_data.copy()

    plot_config = {}
    if dd_config is not None:
        plot_config = dd_config

    match snakemake.params.method:
        case "voting":
            from src.drift_detection.voting import (
                get_offline_voting_drift_detector,
            ) 

            plot_config = dd_config
            if dd_config is None:
                dd = get_offline_voting_drift_detector(
                    window_size=head(snakemake.params.window_size),
                    threshold=head(snakemake.params.threshold),
                    verbose=False,
                )
                plot_config = snakemake.params
            else:
                dd = get_offline_voting_drift_detector(
                    window_size=dd_config["window_size"],
                    threshold=dd_config["threshold"],
                    verbose=False,
                )

        case "ruptures":
            from src.drift_detection.ruptures import (
                get_offline_ruptures_drift_detector,
            )
            if dd_config is None:
                dd = get_offline_ruptures_drift_detector(
                    kernel=snakemake.params.kernel,
                    min_size=snakemake.params.min_size,
                    jump=snakemake.params.jump,
                    penalty=snakemake.params.penalty,
                )
            else:
                dd = get_offline_ruptures_drift_detector(
                    kernel=dd_config["kernel"],
                    min_size=dd_config["min_size"],
                    jump=dd_config["jump"],
                    penalty=dd_config["penalty"],
                ) 
        case _:
            raise ValueError("Method not found")

    match snakemake.params.dataset:
        case "alibaba":
            data = data[dataset_config["y"]]
        case "google":
            data = data[dataset_config["y"]]
        # case "cori":
        #     data = data[dataset_config["y"]]
        case _:
            raise ValueError("Dataset name not found")

    change_list = dd.predict(data.values)

    if snakemake.params.method == "ruptures":
        change_list = change_list[:-1]

    n_dist = len(change_list)
    if n_dist == 0:
        print("No change detected")
        return

    print(f"Number of distributions: {n_dist}")
    plot(
        data,
        change_list,
        path=output_path / f"{base_output_filename}.png",
        config=plot_config
    )

    data = add_dist_label(orig_data, change_list, start_from=0)
    data.to_csv(output_path / f"{base_output_filename}.csv", index=False)

    changes = np.zeros((len(change_list), 2))
    changes[:, 0] = change_list

    for i in range(len(change_list)):
        if snakemake.params.dataset == "alibaba":
            changes[i, 1] = orig_data.iloc[change_list[i]]["time_stamp"]
        elif snakemake.params.dataset == "google":
            changes[i, 1] = orig_data.iloc[change_list[i]]["start_time"]

    np.savetxt(
        output_path / f"{base_output_filename}_cp.csv",
        changes,
        fmt="%d",
        delimiter=",",
    )


if __name__ == "__main__":
    main()
