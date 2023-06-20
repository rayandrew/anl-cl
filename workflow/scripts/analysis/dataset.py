from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.transforms.alibaba_scheduler as transforms

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


def get_dataset(input_path: Path, n_labels: int, target: str):
    data = pd.read_parquet(input_path)

    _transforms = [
        transforms.CleanDataTransform(exclude=[target, "cpu_max"]),
        # transforms.AppendPrevFeatureTransform(
        #     columns=["plan_cpu", "plan_mem", "instance_num"]
        # ),
        transforms.DiscretizeOutputTransform(
            target=target, n_bins=n_labels
        ),
        transforms.DiscretizeOutputTransform(
            target="cpu_max", n_bins=n_labels
        ),
    ]

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
    fig.savefig(output_path / "train.png")
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


def plot_hist_bar(
    dataset: pd.DataFrame, target: str, output_path: Path
):
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
    fig.savefig(output_path / "train_bar.png")
    plt.close(fig)


def plot_cdf(
    dataset: pd.DataFrame,
    target: str,
    output_path: Path,
    title: str = "Dataset",
):
    fig, ax = plt.subplots()
    sns.ecdfplot(
        data=dataset.filter(like="cpu_", axis="columns"),
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
    fig.savefig(output_path / "train_cdf.png")
    plt.close(fig)


def main():
    input_path = Path(snakemake.input[0])
    output_path = Path(snakemake.output[0])

    target = "cpu_avg"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path {input_path} does not exist"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(
        input_path=input_path,
        n_labels=4,
        target=target,
    )

    # for i, subset in enumerate(dataset):
    plot_hist(dataset, target, output_path)
    plot_cdf(dataset, target, output_path)
    plot_hist_bar(dataset, target, output_path)


if __name__ == "__main__":
    main()
