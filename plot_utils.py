from typing import Sequence, Dict, Literal, Any, Optional
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, auc
from cycler import cycler

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from parse_v2 import compute_perf


def label_binarizer(y: Sequence[int], n_labels: int) -> np.ndarray:
    y_onehot = np.zeros((len(y), n_labels))
    for i, label in enumerate(y):
        y_onehot[i][label] = 1
    return y_onehot


def get_y_label(y_var: Literal["cpu", "mem", "disk"]):
    if y_var == "cpu":
        return "CPU Utilization"
    elif y_var == "mem":
        return "Memory Utilization"
    elif y_var == "disk":
        return "Disk IO Utilization"
    else:
        raise ValueError(f"Unknown y_var: {y_var}")


@dataclass
class TrainResult:
    avg_acc: Sequence[Sequence[float]] = field(default_factory=list)
    avg_forgetting: Sequence[Sequence[float]] = field(default_factory=list)
    ovr_avg_acc: Optional[float] = None
    ovr_avg_forgetting: Optional[float] = None


@dataclass
class EvalResult:
    name: str
    diffs: np.ndarray
    diffs_dict: Dict[int, int]
    y_origs: np.ndarray
    y_preds: np.ndarray
    predict_proba: np.ndarray
    train_results: Optional[TrainResult] = None


def auc_roc(result: EvalResult, n_labels: int = 10):
    fpr, tpr, roc_auc = dict(), dict(), dict()

    y_onehot = label_binarizer(result.y_origs, n_labels)

    for i in range(n_labels):
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], result.predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # calculate micro avg
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_onehot.ravel(), result.predict_proba.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # calculate macro avg
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_labels)]))
    # all_fpr = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_labels):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_labels

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


COLORS = ["blue", "orange", "green", "black", "maroon", "purple", "brown"]
CYCLE_COLORS = cycle(COLORS)
CYCLER_COLORS = cycler(color=COLORS)


def plot_roc_curve(ax, res: EvalResult, title: str, n_labels: int = 10):
    fpr, tpr, roc_auc = auc_roc(res, n_labels)

    for class_id, color in zip(range(n_labels), CYCLE_COLORS):
        if class_id not in roc_auc:
            continue
        ax.plot(
            fpr[class_id],
            tpr[class_id],
            label=f"ROC region #{class_id} (area = {roc_auc[class_id]:.2f})",
            linestyle="-",
            color=color,
        )

    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"ROC micro avg (area = {roc_auc['micro']:.2f})",
        linestyle=":",
        color="deeppink",
        linewidth=4,
    )
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"ROC macro avg (area = {roc_auc['macro']:.2f})",
        linestyle=":",
        color="navy",
        linewidth=4,
    )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize="small")


def plot_diff(
    results: Sequence[EvalResult],
    output_folder: Path,
    args: Any,
    title: Optional[str] = None,
):
    assert len(results) >= 2
    print("Plotting diffs...")

    n_results = len(results)

    fig, ax = plt.subplots(figsize=(4 * n_results, 5))
    width = 0.4

    # Add counts above the two bar graphs
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    diffs = np.array(list(results[0].diffs_dict.keys()))

    for i, res in enumerate(results):
        ax.bar(
            diffs + width * (i - n_results / 2),
            res.diffs_dict.values(),
            width=width,
            label=res.name,
        )
        # autolabel(rect)

    ax.set_xlabel("Diff")
    ax.set_ylabel("Frequency")
    ax.set_title("Diff between actual and predicted label")
    # ax.set_xlim(-args.n_labels, args.n_labels)
    ax.set_xticks(diffs)
    ax.set_xticklabels(diffs)
    ax.legend([res.name for res in results])

    suptitle = "Prediction diff"
    if title:
        suptitle += f" of {title}"
    fig.suptitle(suptitle)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()

    fig.savefig(output_folder / "diff.png", dpi=100)
    plt.close(fig)


def plot_prediction(
    results: Sequence[EvalResult],
    output_folder: Path,
    changepoints: np.ndarray,
    args: Any,
    title: Optional[str] = None,
):
    assert len(results) >= 1
    print("Plotting prediction...")

    def format_ax(ax, min_x: int = 0, n_scale: int = 1, format: bool = True):
        ax.set_ylabel("Regions")
        if format:
            ax.set_ylim(min_x, args.n_labels)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_major_locator(ticker.MultipleLocator(n_scale))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    n_results = len(results)
    fig, axs = plt.subplots(n_results, 1, figsize=(16, 4 * n_results))

    if n_results == 1:
        axs = [axs]

    for i, res in enumerate(results):
        axs[i].plot(
            res.y_origs,
            label="Original",
            alpha=0.5,
            linestyle="--",
        )
        axs[i].plot(
            res.y_preds,
            label="Predictions",
            alpha=0.5,
            linestyle="-",
        )

        axs[i].set_title(res.name)
        axs[i].legend()
        format_ax(axs[i])

    if len(changepoints) > 0:
        for cp in changepoints:
            for ax in axs:
                ax.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

    suptitle = "Prediction"
    if title:
        suptitle += f" of {title}"
    fig.suptitle(suptitle)

    fig.tight_layout()
    fig.savefig(output_folder / "prediction.png", dpi=100)
    plt.close(fig)


def plot_auc_roc(
    results: Sequence[EvalResult],
    output_folder: Path,
    args: Any,
    title: Optional[str] = None,
):
    assert len(results) >= 1
    print("Plotting AUC ROC...")

    n_results = len(results)
    fig, axs = plt.subplots(1, n_results, figsize=(6 * n_results, 6))

    if n_results == 1:
        axs = [axs]

    for i, res in enumerate(results):
        plot_roc_curve(axs[i], res, res.name)

    suptitle = f"ROC Plot"
    if title:
        suptitle = f"{suptitle} of {title}"

    fig.suptitle(suptitle)

    fig.tight_layout()
    fig.savefig(output_folder / "auc_roc.png", dpi=100)
    plt.close(fig)


MARKERS = ["D", "P", "s", "v", "o", "*", "X"]
CYCLE_MARKERS = cycle(MARKERS)
CYCLER_MARKERS = cycler(marker=MARKERS)


def plot_avg_acc(
    results: Sequence[EvalResult],
    output_folder: Path,
    n_exp: int,
    args: Any,
    title: Optional[str] = None,
):
    assert len(results) >= 1
    print("Plotting average accuracy...")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.arange(1, n_exp + 1)

    ax.set_prop_cycle(CYCLER_COLORS + CYCLER_MARKERS)

    for res in results:
        if res.train_results is None or res.train_results.avg_forgetting is None:
            continue
        ax.plot(
            x,
            res.train_results.avg_acc,
            label=res.name,
            alpha=0.5,
            linestyle="-",
        )

    ax.set_xticks(x)
    ax.set_xlim([0.7, float(n_exp) + 0.3])
    ax.legend()
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Distribution Shift Window (Task)")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, symbol="", decimals=0))

    suptitle = f"Average accuracy"
    if title:
        suptitle = f"{suptitle} of {title}"

    fig.suptitle(suptitle)

    fig.tight_layout()
    fig.savefig(output_folder / "avg_acc.png", dpi=100)
    plt.close(fig)


def plot_avg_forgetting(
    results: Sequence[EvalResult],
    output_folder: Path,
    n_exp: int,
    args: Any,
    title: Optional[str] = None,
):
    assert len(results) >= 1
    print("Plotting average forgetting...")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.arange(1, n_exp + 1)

    ax.set_prop_cycle(CYCLER_COLORS + CYCLER_MARKERS)

    for res in results:
        if res.train_results is None or res.train_results.avg_forgetting is None:
            continue
        ax.plot(
            x,
            res.train_results.avg_acc,
            label=res.name,
            alpha=0.5,
            linestyle="-",
        )

    ax.set_xticks(x)
    ax.set_xlim([0.7, float(n_exp) + 0.3])
    ax.legend()
    ax.set_ylabel("Forgetting (%)")
    ax.set_xlabel("Distribution Shift Window (Task)")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, symbol="", decimals=0))

    suptitle = f"Average forgetting"
    if title:
        suptitle = f"{suptitle} of {title}"

    fig.suptitle(suptitle)

    fig.tight_layout()
    fig.savefig(output_folder / "avg_forgetting.png", dpi=100)
    plt.close(fig)


__all__ = [
    "label_binarizer",
    "TrainResult",
    "EvalResult",
    "auc_roc",
    "plot_roc_curve",
    "COLORS",
    "MARKERS",
    "get_y_label",
    "plot_diff",
    "plot_prediction",
    "plot_auc_roc",
    "plot_avg_acc",
    "plot_avg_forgetting",
]
