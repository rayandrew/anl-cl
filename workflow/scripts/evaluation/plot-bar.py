from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.helpers.definitions import Snakemake
from src.utils.general import set_seed
from src.utils.logging import logging, setup_logging
from src.utils.summary import TrainingSummary, generate_summary

if TYPE_CHECKING:
    snakemake: Snakemake = Snakemake()

setup_logging(snakemake.log[0])
log = logging.getLogger(__name__)
sns.color_palette("tab10")


def append_if_enum_exists_in_str(
    EnumType, input: str, output: MutableSequence[str]
):
    for e in EnumType:
        if e.value in input:
            output.append(e.value)
            return


def label_from_path(path: Path, sep: str = "_"):
    path_str = str(path)
    path_str = path_str.replace("/train_results.json", "")
    paths = path_str.split("/")
    # return sep.join(paths[-3:])  # training_scenario_strategy
    return sep.join(paths[6:])  # training_scenario_strategy


@dataclass
class Result:
    # avg_precisions:  Sequence[float]
    # avg_recalls:  Sequence[float]
    # avg_f1s:  Sequence[float]
    # avg_aurocs:  Sequence[float]
    precision: pd.DataFrame
    recall: pd.DataFrame
    f1: pd.DataFrame
    auroc: pd.DataFrame
    acc: pd.DataFrame
    forgetting: pd.DataFrame
    # label: str
    # f1: float
    # precision: float
    # recall: float
    # auroc: float


def get_metrics(
    summaries: Sequence[TrainingSummary], labels: Sequence[str]
) -> Result:
    avg_f1s: list[Tuple[str, float]] = []
    avg_precisions: list[Tuple[str, float]] = []
    avg_recalls: list[Tuple[str, float]] = []
    avg_aurocs: list[Tuple[str, float]] = []
    avg_accs: list[Tuple[str, float]] = []
    avg_forgettings: list[Tuple[str, float]] = []
    for summary, label in zip(summaries, labels):
        log.info(f"Label: {label}, summaries: {len(summaries)}")
        log.info(summaries)
        for i in range(len(summary.avg_f1)):
            avg_f1s.append((label, np.mean(summary.avg_f1[i]).item()))
            avg_precisions.append(
                (label, np.mean(summary.avg_precision[i]).item())
            )
            avg_recalls.append((label, np.mean(summary.avg_recall[i]).item()))
            avg_aurocs.append((label, np.mean(summary.avg_auroc[i]).item()))
            avg_accs.append((label, np.mean(summary.avg_acc[i]).item()))
            avg_forgettings.append((label, summary.ovr_avg_forgetting))

    return Result(
        f1=pd.DataFrame(avg_f1s, columns=["label", "value"]),
        precision=pd.DataFrame(avg_precisions, columns=["label", "value"]),
        recall=pd.DataFrame(avg_recalls, columns=["label", "value"]),
        auroc=pd.DataFrame(avg_aurocs, columns=["label", "value"]),
        acc=pd.DataFrame(avg_accs, columns=["label", "value"]),
        forgetting=pd.DataFrame(avg_forgettings, columns=["label", "value"]),
    )


def plot_bar(data: pd.DataFrame, label: str):
    label = label.upper()
    fig, ax = plt.subplots(figsize=(0.2 * len(data), 5))
    sns.barplot(x="label", y="value", data=data, ax=ax)
    # sns.barplot(x="label", y="value", data=df, ax=ax)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=30,
        horizontalalignment="right",
    )
    # ax.set_xticklabels(bars, rotation=90)
    ax.set_xlabel("")
    fig.tight_layout()
    return fig


def plot_line(data: pd.DataFrame, label_title: str):
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(15, 5))

    # Transforming the DataFrame
    unique_labels = data["label"].unique()
    results = pd.DataFrame()

    max_length = data.groupby("label")["value"].transform("count").max()

    for label in unique_labels:
        values = data[data["label"] == label]["value"].tolist()
        # Ignore if 1 chunk
        if len(values) == 1:
            continue
        elif len(values) < max_length:
            # Pad with None for consistency
            # if the plotted data has different chunks
            values.extend([None] * (max_length - len(values)))
        results[label] = values

    # Plot the data
    results.plot(ax=ax, marker="o", linestyle="-")
    # legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Chunk")
    plt.ylabel("Value")
    plt.title(label_title.upper())  # Add title to the plot
    plt.xticks(range(len(results)))  # Make x-values discrete
    ax.set_xticklabels(
        results.index
    )  # Set x-tick labels based on DataFrame index

    # Adjust the layout to make room for the legend
    plt.subplots_adjust(right=0.8)

    fig.tight_layout()
    return fig


def main():
    config = snakemake.config

    set_seed(config.get("seed", 0))
    print(str(snakemake.input))
    input_paths = Path(str(snakemake.input)).glob("**/train_results.json")
    print(input_paths)
    output_folder = Path(str(snakemake.output))
    output_folder.mkdir(parents=True, exist_ok=True)

    labels = []
    summaries = []

    for input in input_paths:
        print(input)
        label = label_from_path(input)
        labels.append(label)
        summary = generate_summary(input)
        summaries.append(summary)
    result = get_metrics(summaries, labels)

    for metric in result.__annotations__.keys():
        data: pd.DataFrame = getattr(result, metric)

        if len(data) == 0:
            continue

        metric_name = metric.replace("avg_", "")

        fig = plot_bar(data=data, label=metric_name)
        fig.savefig(output_folder / f"{metric}.png", dpi=300)
        plt.close(fig)

        fig_line = plot_line(data, metric_name)
        fig_line.savefig(output_folder / f"{metric}_line.png", dpi=300)
        plt.close(fig_line)


if __name__ == "__main__":
    main()
