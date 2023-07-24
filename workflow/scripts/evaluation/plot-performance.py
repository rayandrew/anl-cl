# flake8: noqa: E501
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

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
    last_acc: pd.DataFrame
    last_auroc: pd.DataFrame
    last_forgetting: pd.DataFrame
    last_f1: pd.DataFrame
    last_recall: pd.DataFrame
    last_precision: pd.DataFrame
    chunk_acc: pd.DataFrame
    chunk_auroc: pd.DataFrame
    chunk_forgetting: pd.DataFrame
    chunk_f1: pd.DataFrame
    chunk_recall: pd.DataFrame
    chunk_precision: pd.DataFrame
    class_accuracy: pd.DataFrame
    # task_summary: dict[str, dict[int, TaskSummary]]
    # label: str
    # f1: float
    # precision: float
    # recall: float
    # auroc: float


RESULT_COLUMNS = ["label", "value"]


# Updated LABELS dictionary
LABELS = {
    "f1": "F1",
    "precision": "PRECISION",
    "recall": "RECALL",
    "auroc": "AUROC",
    "acc": "ACCURACY",
    "forgetting": "FORGETTING",
    "last_acc": "Last Model ACCURACY",
    "last_auroc": "Last Model AUROC",
    "last_f1": "Last Model F1",
    "last_recall": "Last Model RECALL",
    "last_precision": "Last Model PRECISION",
    "last_forgetting": "Last Model FORGETTING",
    "chunk_acc": "Model ACCURACY evaluated on Chunk N only",
    "chunk_auroc": "Model AUROC evaluated on Chunk N only",
    "chunk_f1": "Model F1 evaluated on Chunk N only only",
    "chunk_recall": "Model RECALL evaluated on Chunk N only",
    "chunk_precision": "Model PRECISION evaluated on Chunk N only",
    "chunk_forgetting": "Model FORGETTING evaluated on Chunk N only",
    "class_accuracy": "Classification Accuracy",
}


PATH_LABELS = {
    "drift-detection": "DD",
    "feature": "FEAT",
    "model": "MODEL",
    "no-retrain": "NR",
    "from-scratch": "FS",
}


def get_label(label: str):
    # log.info(f"get_label: {label}")
    for key in LABELS.keys():
        # log.info(f"key: {key}")
        if label in key:
            # log.info(f"label {label} in key: {key}")
            return LABELS[key]
    return "unknown"


def change_path_label(label: str):
    label_temp = label
    for key in PATH_LABELS.keys():
        if key in label_temp:
            label_temp = label_temp.replace(key, PATH_LABELS[key])
    return label_temp


def get_metrics(
    summaries: Sequence[TrainingSummary], labels: Sequence[str]
) -> Result:
    avg_f1s: list[Tuple[str, float]] = []
    avg_precisions: list[Tuple[str, float]] = []
    avg_recalls: list[Tuple[str, float]] = []
    avg_aurocs: list[Tuple[str, float]] = []
    avg_accs: list[Tuple[str, float]] = []
    avg_forgettings: list[Tuple[str, float]] = []

    # last model metrics
    last_accs: List[Tuple[str, float]] = []
    last_aurocs: List[Tuple[str, float]] = []
    last_f1s: List[Tuple[str, float]] = []
    last_forgettings: List[Tuple[str, float]] = []
    last_recalls: List[Tuple[str, float]] = []
    last_precisions: List[Tuple[str, float]] = []

    chunk_accs: List[Tuple[str, float]] = []
    chunk_aurocs: List[Tuple[str, float]] = []
    chunk_f1s: List[Tuple[str, float]] = []
    chunk_forgettings: List[Tuple[str, float]] = []
    chunk_recalls: List[Tuple[str, float]] = []
    chunk_precisions: List[Tuple[str, float]] = []

    avg_class_acc_data: List[tuple[str, float, int, str]] = []

    for summary, label in zip(summaries, labels):
        log.info(f"Label: {label}, summaries: {len(summaries)}")

        label_path = change_path_label(label)
        for class_id, acc_values in summary.avg_class_acc.items():
            class_label = f"{label_path}_Class_{class_id}"
            avg_class_acc_data.extend(
                [
                    (class_label, idx, value, class_id)
                    for value, idx in enumerate(acc_values)
                ]
            )

        for i in range(len(summary.avg_acc)):
            avg_f1s.append((label_path, np.mean(summary.avg_f1[i]).item()))
            avg_precisions.append(
                (label_path, np.mean(summary.avg_precision[i]).item())
            )
            avg_recalls.append(
                (label_path, np.mean(summary.avg_recall[i]).item())
            )
            avg_aurocs.append(
                (label_path, np.mean(summary.avg_auroc[i]).item())
            )
            avg_forgettings.append((label_path, summary.ovr_avg_forgetting))
            avg_accs.append((label_path, np.mean(summary.avg_acc[i]).item()))

        # Return evaluations only at chunk being trained on.
        if "NR" not in label_path:
            for idx, (key, task) in enumerate(summary.task_data.items()):
                chunk_accs.append((label_path, task.acc[idx]))
                # See if new version, aka with auroc or not
                if len(task.auroc) != 0:
                    chunk_aurocs.append((label_path, task.auroc[idx]))
                    chunk_f1s.append((label_path, task.f1[idx]))
                    # chunk_forgettings.append((label_path, task.forgetting[-1]))
                    chunk_recalls.append((label_path, task.recall[idx]))
                    chunk_precisions.append((label_path, task.precision[idx]))

        # Get Last Model
        for key, task in summary.task_data.items():
            last_accs.append((label_path, task.acc[-1]))
            if len(task.auroc) != 0:
                last_aurocs.append((label_path, task.auroc[-1]))
                last_f1s.append((label_path, task.f1[-1]))
                # last_forgettings.append((label_path, task.forgetting[-1]))
                last_recalls.append((label_path, task.recall[-1]))
                last_precisions.append((label_path, task.precision[-1]))

    return Result(
        f1=pd.DataFrame(avg_f1s, columns=RESULT_COLUMNS),
        precision=pd.DataFrame(avg_precisions, columns=RESULT_COLUMNS),
        recall=pd.DataFrame(avg_recalls, columns=RESULT_COLUMNS),
        auroc=pd.DataFrame(avg_aurocs, columns=RESULT_COLUMNS),
        acc=pd.DataFrame(avg_accs, columns=RESULT_COLUMNS),
        forgetting=pd.DataFrame(avg_forgettings, columns=RESULT_COLUMNS),
        last_acc=pd.DataFrame(last_accs, columns=RESULT_COLUMNS),
        last_auroc=pd.DataFrame(last_aurocs, columns=RESULT_COLUMNS),
        last_f1=pd.DataFrame(last_f1s, columns=RESULT_COLUMNS),
        last_recall=pd.DataFrame(last_recalls, columns=RESULT_COLUMNS),
        last_precision=pd.DataFrame(last_precisions, columns=RESULT_COLUMNS),
        last_forgetting=pd.DataFrame(last_forgettings, columns=RESULT_COLUMNS),
        chunk_acc=pd.DataFrame(chunk_accs, columns=RESULT_COLUMNS),
        chunk_auroc=pd.DataFrame(chunk_aurocs, columns=RESULT_COLUMNS),
        chunk_f1=pd.DataFrame(chunk_f1s, columns=RESULT_COLUMNS),
        chunk_recall=pd.DataFrame(chunk_recalls, columns=RESULT_COLUMNS),
        chunk_precision=pd.DataFrame(chunk_precisions, columns=RESULT_COLUMNS),
        chunk_forgetting=pd.DataFrame(
            chunk_forgettings, columns=RESULT_COLUMNS
        ),
        class_accuracy=pd.DataFrame(
            avg_class_acc_data, columns=(RESULT_COLUMNS + ["chunk", "class"])
        ),
    )


def plot_bar(data: pd.DataFrame, label: str, get_last: bool = False):
    # label = label.upper()
    fig, ax = plt.subplots(figsize=(0.15 * len(data) + 2, 5))
    if get_last:
        temp_data = (
            data.groupby("label")
            .tail(1)
            .sort_values(by="value", ascending=False)
            .reset_index(drop=True)
        )
        sns.barplot(
            x="label",
            y="value",
            data=temp_data,
            ax=ax,
        )
    else:
        sns.barplot(
            x="label",
            y="value",
            data=data,
            ax=ax,
            order=data.groupby("label")["value"]
            .mean()
            .sort_values(ascending=False)
            .index,
        )
    # sns.barplot(x="label", y="value", data=df, ax=ax)
    label_title = get_label(label)
    if "acc" in label:
        ax.set_ylabel("ACCURACY")
    else:
        ax.set_ylabel(label_title)
    ax.set_title(label_title)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=30,
        horizontalalignment="right",
    )
    # ax.set_xticklabels(bars, rotation=90)
    ax.set_xlabel("")
    fig.tight_layout()
    return fig


def plot_line(data: pd.DataFrame, label: str):
    fig, ax = plt.subplots(figsize=(15, 5))

    # Transforming the DataFrame
    unique_labels = data["label"].unique()
    results = pd.DataFrame()

    max_length = data.groupby("label")["value"].transform("count").max()

    for _label in unique_labels:
        values = data[data["label"] == _label]["value"].tolist()
        if len(values) < max_length:
            # Pad with None for consistency
            # if the plotted data has different chunks
            values.extend([None] * (max_length - len(values)))
        results[_label] = values

    # Plot the data
    results.plot(ax=ax, marker="o", linestyle="-")
    # legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Chunk")
    label_title = get_label(label)

    if "acc" in label:
        ax.set_ylabel("ACCURACY")
    else:
        ax.set_ylabel(label_title)

    ax.set_title(label_title)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results.index)

    fig.subplots_adjust(right=0.8)

    fig.tight_layout()
    return fig


def plot_class_bar(data, task):
    plt.figure(figsize=(10, 6))
    plt.title(f"Bar Plot for Chunk {task}", fontsize=16)

    # Filter the data for the current task
    task_data = data[data["chunk"] == task]
    task_data["label"] = task_data["label"].apply(
        lambda label: label.split("_Class")[0].strip()
    )

    # Check if the filtered data is empty
    if task_data.empty:
        print(f"Warning: Data for task {task} is empty. Skipping plot.")
        return None

    # Create the figure and axis objects
    fig, ax = plt.subplots()

    # Use the order parameter to sort the x-axis labels in alphabetical order
    sns.barplot(
        data=task_data,
        x="label",
        y="value",
        hue="class",
        order=sorted(task_data["label"].unique()),
        ax=ax,
    )
    plt.xlabel("Label", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.xticks(rotation=45, ha="right")

    # Ensure tight layout
    plt.tight_layout()

    return fig


def main():
    config = snakemake.config

    set_seed(config.get("seed", 0))
    log.info("Input path %s", str(snakemake.input))
    input_paths = Path(str(snakemake.input)).glob("**/train_results.json")
    output_folder = Path(str(snakemake.output))
    output_folder.mkdir(parents=True, exist_ok=True)
    class_metrics_folder = output_folder / "class_metrics"
    class_metrics_folder.mkdir(parents=True, exist_ok=True)

    labels = []
    summaries = []

    for input in input_paths:
        log.info("Processing input: %s", input)
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
        log.info("Plotting %s", metric_name)

        fig_bar_stddev = plot_bar(data=data, label=metric_name, get_last=False)
        fig_bar_stddev.savefig(output_folder / f"{metric}_bar_std.png", dpi=300)
        plt.close(fig_bar_stddev)

        if "last" not in metric_name:
            fig_bar_last = plot_bar(data=data, label=metric_name, get_last=True)
            fig_bar_last.savefig(
                output_folder / f"{metric}_bar_last.png", dpi=300
            )
            plt.close(fig_bar_last)

        fig_line = plot_line(data, label=metric_name)
        fig_line.savefig(output_folder / f"{metric}_line.png", dpi=300)
        plt.close(fig_line)

        if "class_accuracy" in metric_name:
            num_tasks = data["chunk"].nunique()
            # Create a bar plot for each Chunk and save it as an image
            for task in range(0, num_tasks):
                fig = plot_class_bar(data, task)
                if fig is not None:
                    # Save the figure as an image in the specified output folder
                    fig.savefig(
                        class_metrics_folder
                        / f"classification_accuracy_bar_chunk_{task}.png",
                        dpi=300,
                    )
                    # Close the figure to release resources
                    plt.close(fig)

        data.to_csv(output_folder / f"{metric}.csv", index=False)


if __name__ == "__main__":
    main()
