from typing import Sequence, Dict, Literal, Any
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
class EvalResult:
    name: str
    diffs: np.ndarray
    diffs_dict: Dict[int, int]
    y_origs: np.ndarray
    y_preds: np.ndarray
    predict_proba: np.ndarray


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


COLORS = cycle(
    [
        "aqua",
        "darkorange",
        "cornflowerblue",
        "red",
        "green",
        "blue",
        "yellow",
        "black",
        "darkblue",
        "darkgreen",
    ]
)


def plot_roc_curve(ax, res: EvalResult, title: str, n_labels: int = 10):
    fpr, tpr, roc_auc = auc_roc(res, n_labels)

    for class_id, color in zip(range(n_labels), COLORS):
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
    results: Sequence[EvalResult], output_folder: Path, args: Any
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

    # ax.bar(
    #     diffs - width / 2, non_seq_res.diffs_dict.values(), color="blue", width=width
    # )
    # ax.bar(diffs + width / 2, seq_res.diffs_dict.values(), color="red", width=width)

    # autolabel(rects1)
    # autolabel(rects2)

    ax.set_xlabel("Diff")
    ax.set_ylabel("Frequency")
    ax.set_title("Diff between actual and predicted label")
    # ax.set_xlim(-args.n_labels, args.n_labels)
    ax.set_xticks(diffs)
    ax.set_xticklabels(diffs)
    ax.legend([res.name for res in results])

    fig.suptitle(f"Prediction of {get_y_label(args.y)}")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()

    fig.savefig(output_folder / "diff.png", dpi=100)
    plt.close(fig)

__all__ = [
    "label_binarizer",
    "EvalResult",
    "auc_roc",
    "plot_roc_curve",
    "COLORS",
    "get_y_label",
]
