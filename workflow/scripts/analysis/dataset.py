from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.transforms.alibaba_eighteen as transforms

# from src.dataset.alibaba import (
#     AlibabaSchedulerDataAccessor,
#     AlibabaSchedulerDatasetChunkGenerator,
#     AlibabaSchedulerDatasetGenerator,
# )
from src.helpers.definitions import Snakemake
from src.utils.logging import logging, setup_logging

if TYPE_CHECKING:
    snakemake: Snakemake

setup_logging(snakemake.log[0])
log = logging.getLogger(__name__)
sns.color_palette("tab10")


def cpu_util_percent(target: str):
    def _cpu_util_percent(data: pd.DataFrame):
        data[f"percent_{target}"] = (data[target] / data["plan_cpu"]) * 100
        return data

    return _cpu_util_percent


def get_dataset(input_path: Path, n_labels: int, targets: list[str]):
    data = pd.read_parquet(input_path)

    _transforms = [
        transforms.CleanDataTransform(exclude=targets),
        # lambda df: df.iloc[:100_000],
        lambda df: df[df[df.columns] > 0],
        # transforms.AppendPrevFeatureTransform(
        #     columns=["plan_cpu", "plan_mem", "instance_num"]
        # ),
    ]

    for target in targets:
        # _transforms.append(cpu_util_percent(target=target))
        _transforms.append(
            transforms.DiscretizeOutputTransform(
                target=f"percent_{target}",
                rename_target=f"pred_percent_{target}",
                n_bins=n_labels,
            )
        )

    for transform in _transforms:
        data = transform(data)

    return data


def plot_hist(
    dataset: pd.DataFrame,
    target: str,
    output_path: Path,
    title: str = "Dataset",
):
    # plot histogram of target
    fig, ax = plt.subplots()
    sns.histplot(
        data=dataset[target],
        bins=4,
        stat="probability",
        discrete=True,
        kde=True,
        color="tab:blue",
        ax=ax,
    )
    ax.set_title(f"{title}")
    ax.set_xlabel(target.upper())
    ax.set_ylabel("Probability")
    fig.savefig(output_path)
    plt.close(fig)

    # fig, ax = plt.subplots()
    # sns.histplot(
    #     data=dataset[target],
    #     bins=4,
    #     stat="probability",
    #     discrete=True,
    #     kde=True,
    #     color="tab:blue",
    #     ax=ax,
    # )
    # ax.set_title(f"Test subset")
    # ax.set_xlabel("CPU avg")
    # ax.set_ylabel("Probability")
    # fig.savefig(output_path / "test.png")
    # plt.close(fig)


def plot_hist_bar(dataset: pd.DataFrame, target: str, output_path: Path):
    fig, ax = plt.subplots()
    sns.countplot(
        x=target,
        data=dataset,
        ax=ax,
    )
    ax.bar_label(ax.containers[0])
    ax.set_title(f"Train subset")
    ax.set_xlabel("CPU avg")
    ax.set_ylabel("Probability")
    fig.savefig(output_path)
    plt.close(fig)


def plot_cdf(
    dataset: pd.DataFrame,
    target: str,
    output_path: Path,
    title: str = "Dataset",
):
    fig, ax = plt.subplots()
    sns.ecdfplot(
        data=dataset.filter(regex="^percent_", axis="columns"),
        # x=target,
        ax=ax,
    )
    # sns.ecdfplot(
    #     data=dataset,
    #     x="cpu_max",
    #     ax=ax,
    # )
    ax.set_title(f"{title}")
    ax.set_xlabel(target.upper())
    ax.set_ylabel("CDF")
    fig.savefig(output_path / "cdf_raw.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.ecdfplot(
        data=dataset.filter(regex="^pred_", axis="columns"),
        # x=target,
        ax=ax,
    )
    ax.set_title(f"{title}")
    ax.set_xlabel(target.upper())
    ax.set_ylabel("CDF")
    fig.savefig(output_path / "cdf_pred.png")
    plt.close(fig)


def main():
    input_path = Path(snakemake.input[0])
    output_path = Path(snakemake.output[0])

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist")

    targets = ["cpu_avg", "cpu_max"]
    dataset = get_dataset(
        input_path=input_path,
        n_labels=4,
        targets=targets,
    )

    output_path.mkdir(parents=True, exist_ok=True)

    # for i, subset in enumerate(dataset):
    for target in targets:
        # _output_path = (output_path / target).mkdir(parents=True, exist_ok=True)
        plot_hist(
            dataset,
            f"percent_{target}",
            output_path=output_path / f"{target}_density.png",
        )
        plot_hist(
            dataset,
            f"pred_percent_{target}",
            output_path=output_path / f"pred_{target}_density.png",
        )
        plot_hist_bar(
            dataset,
            f"pred_percent_{target}",
            output_path=output_path / f"pred_{target}_bar.png",
        )
    plot_cdf(dataset, target, output_path)


if __name__ == "__main__":
    main()
