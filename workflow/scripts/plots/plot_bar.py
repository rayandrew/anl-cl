from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.general import set_seed
from src.utils.logging import logging, setup_logging
from src.utils.summary import TrainingSummary, generate_summary

if TYPE_CHECKING:
    snakemake: Any

setup_logging(snakemake.log[0])
log = logging.getLogger(__name__)
sns.color_palette("tab10")


def plot_bar(data: Sequence[float], labels: Sequence[str]):
    fig, ax = plt.subplots()
    x = range(len(data))
    ax.bar(x, data)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ybound(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title(f"AUROC")
    return fig


def label_from_path(path: Path):
    if "offline-classification-no-retrain" in str(path):
        return "OFF_FROM_SCRATCH"
    elif "offline-classification-retrain-chunks-naive" in str(path):
        return "OFF_RETRAIN_CHUNKS_NAIVE"
    elif "offline-classification-retrain-chunks-from-scratch" in str(
        path
    ):
        return "OFF_RETRAIN_CHUNKS_FROM_SCRATCH"
    raise ValueError(f"Unknown scenario for path: {path}")


def get_auroc(
    summaries: Sequence[TrainingSummary], labels: Sequence[str]
):
    avg_aurocs = []
    for summary, label in zip(summaries, labels):
        for i in range(len(summary.avg_auroc)):
            avg_aurocs.append(
                [label, np.mean(summary.avg_auroc[i]).item()]
            )
    return avg_aurocs


def get_recall(
    summaries: Sequence[TrainingSummary], labels: Sequence[str]
):
    avg_recalls = []
    for summary, label in zip(summaries, labels):
        for i in range(len(summary.avg_recall)):
            avg_recalls.append(
                [label, np.mean(summary.avg_recall[i]).item()]
            )
    return avg_recalls


def get_precision(
    summaries: Sequence[TrainingSummary], labels: Sequence[str]
):
    avg_precisions = []
    for summary, label in zip(summaries, labels):
        for i in range(len(summary.avg_precision)):
            avg_precisions.append(
                [label, np.mean(summary.avg_precision[i]).item()]
            )
    return avg_precisions


def get_f1(
    summaries: Sequence[TrainingSummary], labels: Sequence[str]
):
    avg_f1s = []
    for summary, label in zip(summaries, labels):
        for i in range(len(summary.avg_f1)):
            avg_f1s.append([label, np.mean(summary.avg_f1[i]).item()])
    return avg_f1s


def main():
    config = snakemake.config

    set_seed(config.get("seed", 0))
    input_paths = [Path(str(input)) for input in snakemake.input]
    output_folder = Path(str(snakemake.output))
    output_folder.mkdir(parents=True, exist_ok=True)

    labels = []
    summaries = []

    for input in input_paths:
        label = label_from_path(input)
        labels.append(label)

        summary = generate_summary(input / "train_results.json")
        summaries.append(summary)

    dfs = {}
    dfs["auroc"] = pd.DataFrame(
        get_auroc(summaries, labels), columns=["label", "value"]
    )
    dfs["precision"] = pd.DataFrame(
        get_precision(summaries, labels), columns=["label", "value"]
    )
    dfs["recall"] = pd.DataFrame(
        get_recall(summaries, labels), columns=["label", "value"]
    )
    dfs["f1"] = pd.DataFrame(
        get_f1(summaries, labels), columns=["label", "value"]
    )

    log.info(f"labels: {label}")
    log.info(f"avg_aurocs: {avg_aurocs}")

    # for i in avg_aurocs:
    #     print(i, len(avg_aurocs[i]), labels)
    #     fig = plot_bar(i, avg_aurocs[i], labels)
    #     fig.savefig(output_folder / f"auroc_{i}.png")
    # fig = plot_bar(avg_aurocs, labels)

    for metric in dfs:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(x="label", y="value", data=dfs[metric])
        # sns.barplot(x="label", y="value", data=df, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ybound(0, 1)
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=30,
            horizontalalignment="right",
        )
        ax.set_xlabel("")
        # ax.set_xticklabels(bars, rotation=90)
        fig.tight_layout()
        fig.savefig(output_folder / f"{metric}.png", dpi=300)
        # sns.despine(left=True, bottom=True)
        plt.close(fig)
        dfs[metric].to_csv(
            output_folder / f"{metric}.csv", index=False
        )


if __name__ == "__main__":
    main()
