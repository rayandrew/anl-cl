from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Any, Sequence
from itertools import cycle
import math

import torch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


from dataset import AlibabaMachineDataset

TRes = Dict[str, Dict[int, int] | Sequence[int]]

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


def label_binarizer(y: Sequence[int], n_labels: int) -> np.ndarray:
    y_onehot = np.zeros((len(y), n_labels))
    for i, label in enumerate(y):
        y_onehot[i][label] = 1
    return y_onehot


def plot_auc_roc(non_seq_res: TRes, seq_res: TRes, output_folder: Path, args: Any):
    def plot_roc_curve(ax, res: TRes, title: str):
        fpr, tpr, roc_auc = dict(), dict(), dict()
        y_onehot = label_binarizer(res["y_origs"], args.n_labels)
        for i in range(args.n_labels):
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], res["predict_proba"][:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            # check if nan
            # if math.isnan(roc_auc[i]):
            #     roc_auc[i] = 0
            # del fpr[i]
            # del tpr[i]
            # del roc_auc[i]

        # calculate micro avg
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_onehot.ravel(), res["predict_proba"].ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # calculate macro avg
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.n_labels)]))
        # all_fpr = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(args.n_labels):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= args.n_labels

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        for class_id, color in zip(range(args.n_labels), COLORS):
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

    print("Plotting AUC ROC...")
    fig, (ax_non_seq, ax_seq) = plt.subplots(1, 2, figsize=(12, 6))

    plot_roc_curve(ax_non_seq, non_seq_res, "Non Sequential")
    plot_roc_curve(ax_seq, seq_res, "Sequential")

    fig.tight_layout()
    fig.savefig(output_folder / "auc_roc.png", dpi=100)


def plot_prediction(
    non_seq_res: pd.DataFrame, seq_res: pd.DataFrame, output_folder: Path, args: Any
):
    print("Plotting prediction...")

    def format_ax(ax, min_x: int = 0, n_scale: int = 1, format: bool = True):
        # ax.set_xlabel("Time")
        ax.set_ylabel("Regions")
        if format:
            ax.set_ylim(min_x, args.n_labels)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_major_locator(ticker.MultipleLocator(n_scale))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig, (ax_non_seq, ax_seq) = plt.subplots(2, 1, figsize=(16, 7))

    # ax_non_seq.plot(non_seq_res["y_origs"], label="Original", alpha=0.9, linestyle="-")
    # ax_non_seq.plot(
    #     non_seq_res["y_preds"],
    #     label="Predictions",
    #     alpha=0.5,
    #     color="maroon",
    #     linestyle="--",
    # )

    # x = list(range(len(non_seq_res["y_origs"])))
    sns.lineplot(
        data=non_seq_res,
        x=non_seq_res.index,
        y="y_origs",
        label="Original",
        ax=ax_non_seq,
        alpha=0.5,
        linestyle="--",
    )
    sns.lineplot(
        data=non_seq_res,
        x=non_seq_res.index,
        y="y_preds",
        label="Predictions",
        ax=ax_non_seq,
        alpha=0.5,
        linestyle="-",
    )

    # ax_non_seq.scatter(
    #     x, non_seq_res["y_origs"], label="Original", alpha=0.9, linestyle="-"
    # )
    # ax_non_seq.scatter(
    #     x,
    #     non_seq_res["y_preds"],
    #     label="Predictions",
    #     alpha=0.5,
    #     color="maroon",
    # )
    ax_non_seq.set_title("Non Sequential")
    ax_non_seq.legend()
    format_ax(ax_non_seq)

    sns.lineplot(
        data=seq_res,
        x=seq_res.index,
        y="y_origs",
        label="Original",
        ax=ax_seq,
        alpha=0.5,
        linestyle="--",
    )
    sns.lineplot(
        data=seq_res,
        x=seq_res.index,
        y="y_preds",
        label="Predictions",
        ax=ax_seq,
        alpha=0.5,
        linestyle="-",
    )
    # sns.lineplot(x, seq_res["y_origs"], label="Original", ax=ax_seq)
    # sns.lineplot(x, seq_res["y_preds"], label="Predictions", ax=ax_seq)

    # ax_seq.plot(seq_res["y_origs"], label="Original", alpha=0.9, linestyle="-")
    # ax_seq.plot(
    #     seq_res["y_preds"],
    #     label="Predictions",
    #     alpha=0.5,
    #     color="maroon",
    #     linestyle="--",
    # )
    # x = list(range(len(seq_res["y_origs"])))
    # ax_seq.scatter(x, seq_res["y_origs"], label="Original", alpha=0.9)
    # ax_seq.scatter(
    #     x,
    #     seq_res["y_preds"],
    #     label="Predictions",
    #     alpha=0.5,
    #     color="maroon",
    # )
    ax_seq.set_title("Sequential")
    ax_seq.legend()
    format_ax(ax_seq)

    prefix_label = "[Local]" if args.local else "[Global]"
    fig.suptitle(f"{prefix_label} Sequential vs Non-Sequential ({args.strategy})")

    fig.tight_layout()
    fig.savefig(output_folder / "plot_vs.png", dpi=100)


def main(args):
    data_path = Path(args.data)
    output_folder = Path(args.output_folder) / data_path.stem / args.strategy
    output_folder.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    raw_seq_data = AlibabaMachineDataset(
        filename=data_path,
        n_labels=args.n_labels,
        mode="predict",
        y=args.y,
        seq=True,
        seq_len=args.seq_len,
    )
    seq_data = torch.utils.data.DataLoader(
        raw_seq_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
    )
    raw_non_seq_data = AlibabaMachineDataset(
        filename=data_path,
        n_labels=args.n_labels,
        mode="predict",
        y=args.y,
        seq=False,
        seq_len=args.seq_len,
    )
    non_seq_data = torch.utils.data.DataLoader(
        raw_non_seq_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
    )

    seq_model = torch.jit.load(args.seq)
    seq_model.to(device)
    seq_model.eval()

    non_seq_model = torch.jit.load(args.non_seq)
    non_seq_model.to(device)
    non_seq_model.eval()

    non_seq_res = {
        "diffs_dict": {},
        "diffs": [],
        "y_origs": [],
        "y_preds": [],
        "predict_proba": [],
    }

    seq_res = {
        "diffs_dict": {},
        "diffs": [],
        "y_origs": [],
        "y_preds": [],
        "predict_proba": [],
    }

    for i in range(-args.n_labels, args.n_labels + 1, 1):
        non_seq_res["diffs_dict"][i] = 0
        seq_res["diffs_dict"][i] = 0

    print("Predicting using non-sequential model...")
    for i, (x, _dist, y) in enumerate(non_seq_data):
        x = x.to(device)
        y = y.to(device)
        y_pred = non_seq_model(x)
        pred_label = torch.argmax(y_pred, dim=1)
        diffs = y - pred_label
        for diff in diffs:
            non_seq_res["diffs_dict"][diff.item()] += 1
        non_seq_res["diffs"] += diffs.tolist()
        non_seq_res["y_origs"] += y.tolist()
        non_seq_res["y_preds"] += pred_label.tolist()
        non_seq_res["predict_proba"] += y_pred.tolist()
        # print(y_pred.tolist())
        # print(non_seq_res["predict_proba"])
        # print(np.array(non_seq_res["predict_proba"]))
        # return
        # break

    print("Predicting using sequential model...")
    for i, (x, _dist, y) in enumerate(seq_data):
        x = x.to(device)
        y = y.to(device)
        y_pred = seq_model(x)
        pred_label = torch.argmax(y_pred, dim=1)
        diffs = y - pred_label
        for diff in diffs:
            seq_res["diffs_dict"][diff.item()] += 1
        seq_res["diffs"] += diffs.tolist()
        seq_res["y_origs"] += y.tolist()
        seq_res["y_preds"] += pred_label.tolist()
        seq_res["predict_proba"] += y_pred.tolist()
        # break

    non_seq_res["predict_proba"] = np.array(non_seq_res["predict_proba"])
    seq_res["predict_proba"] = np.array(seq_res["predict_proba"])

    df_non_seq = pd.DataFrame(
        {
            "y_origs": non_seq_res["y_origs"],
            "y_preds": non_seq_res["y_preds"],
            "diffs": non_seq_res["diffs"],
        }
    )
    df_seq = pd.DataFrame(
        {
            "y_origs": seq_res["y_origs"],
            "y_preds": seq_res["y_preds"],
            "diffs": seq_res["diffs"],
        }
    )

    plot_prediction(df_non_seq, df_seq, output_folder, args=args)
    plot_auc_roc(non_seq_res, seq_res, output_folder, args=args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="Data file")
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("-o", "--output_folder", type=str, default="out")
    parser.add_argument("-nl", "--n_labels", type=int, default=10)
    parser.add_argument(
        "-y",
        type=str,
        choices=["cpu", "mem", "disk"],
        default="cpu",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        help="Strategy to use",
        type=str,
        choices=["gss", "agem", "naive", "lwf", "ewc", "gdumb"],
        default="naive",
    )
    parser.add_argument("--seq", help="Path to sequential model", type=str)
    parser.add_argument("--non_seq", help="Path to non-sequential model", type=str)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--local", action="store_true")

    args = parser.parse_args()
    main(args)
