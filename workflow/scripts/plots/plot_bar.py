from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Tuple

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


def append_if_enum_exists_in_str(
    EnumType, input: str, output: Sequence[str]
):
    for e in EnumType:
        if e.value in input:
            output.append(e.value)
            return


def label_from_path(path: Path, sep: str = "_"):
    path = str(path)
    path = path.replace("/train_results.json", "")
    paths = path.split("/")
    return sep.join(paths[-3:]) # training_scenario_strategy


@dataclass
class Result:
    # avg_precisions: Sequence[float]
    # avg_recalls: Sequence[float]
    # avg_f1s: Sequence[float]
    # avg_aurocs: Sequence[float]
    avg_precisions: pd.DataFrame
    avg_recalls: pd.DataFrame
    avg_f1s: pd.DataFrame
    avg_aurocs: pd.DataFrame
    # label: str
    # f1: float
    # precision: float
    # recall: float
    # auroc: float


def get_metrics(
    summaries: Sequence[TrainingSummary], labels: Sequence[str]
) -> Sequence[Result]:
    avg_f1s: Sequence[Tuple[str, float]] = []
    avg_precisions: Sequence[Tuple[str, float]] = []
    avg_recalls: Sequence[Tuple[str, float]] = []
    avg_aurocs: Sequence[Tuple[str, float]] = []
    for summary, label in zip(summaries, labels):
        for i in range(len(summary.avg_f1)):
            avg_f1s.append((label, np.mean(summary.avg_f1[i]).item()))
            avg_precisions.append(
                (label, np.mean(summary.avg_precision[i]).item())
            )
            avg_recalls.append(
                (label, np.mean(summary.avg_recall[i]).item())
            )
            avg_aurocs.append(
                (label, np.mean(summary.avg_auroc[i]).item())
            )

    return Result(
        avg_f1s=pd.DataFrame(avg_f1s, columns=["label", "value"]),
        avg_precisions=pd.DataFrame(
            avg_precisions, columns=["label", "value"]
        ),
        avg_recalls=pd.DataFrame(
            avg_recalls, columns=["label", "value"]
        ),
        avg_aurocs=pd.DataFrame(
            avg_aurocs, columns=["label", "value"]
        ),
    )


def plot_bar(data: pd.DataFrame, label: str):
    label = label.upper()
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(x="label", y="value", data=data)
    # sns.barplot(x="label", y="value", data=df, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ybound(0, 1)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=30,
        horizontalalignment="right",
    )
    ax.set_xlabel("")
    # ax.set_xticklabels(bars, rotation=90)
    fig.tight_layout()
    return fig


def main():
    config = snakemake.config

    set_seed(config.get("seed", 0))
    input_paths = Path(str(snakemake.input)).glob(
        "**/train_results.json"
    )
    output_folder = Path(str(snakemake.output))
    output_folder.mkdir(parents=True, exist_ok=True)

    labels = []
    summaries = []

    for input in input_paths:
        label = label_from_path(input)
        labels.append(label)
        summary = generate_summary(input)
        summaries.append(summary)

    result = get_metrics(summaries, labels)

    for metric in ["auroc", "precision", "recall", "f1"]:
        data: pd.DataFrame = getattr(result, f"avg_{metric}s")
        fig = plot_bar(data, metric)
        data.to_csv(output_folder / f"{metric}.csv", index=False)
        fig.savefig(output_folder / f"{metric}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
