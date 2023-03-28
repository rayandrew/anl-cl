from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from dataset import AlibabaMachineDataset

from plot_utils import EvalResult, plot_roc_curve, get_y_label


def plot_diff(
    non_seq_res: EvalResult, seq_res: EvalResult, output_folder: Path, args: Any
):
    print("Plotting diffs...")

    fig, ax = plt.subplots(figsize=(8, 5))
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

    diffs = np.array(list(non_seq_res.diffs_dict.keys()))

    ax.bar(
        diffs - width / 2, non_seq_res.diffs_dict.values(), color="blue", width=width
    )
    ax.bar(diffs + width / 2, seq_res.diffs_dict.values(), color="red", width=width)

    # autolabel(rects1)
    # autolabel(rects2)

    ax.set_xlabel("Diff")
    ax.set_ylabel("Frequency")
    ax.set_title("Diff between actual and predicted label")
    # ax.set_xlim(-args.n_labels, args.n_labels)
    ax.set_xticks(diffs)
    ax.set_xticklabels(diffs)
    ax.legend(["Non-Sequence", "Sequence"])

    fig.suptitle(f"Prediction of {get_y_label(args.y)}")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()

    fig.savefig(output_folder / "diff.png", dpi=100)
    plt.close(fig)


def plot_auc_roc(
    non_seq_res: EvalResult, seq_res: EvalResult, output_folder: Path, args: Any
):
    print("Plotting AUC ROC...")
    fig, (ax_non_seq, ax_seq) = plt.subplots(1, 2, figsize=(12, 6))

    plot_roc_curve(ax_non_seq, non_seq_res, "Non-Sequential")
    plot_roc_curve(ax_seq, seq_res, "Sequential")

    fig.tight_layout()
    fig.savefig(output_folder / "auc_roc.png", dpi=100)
    plt.close(fig)


def plot_prediction(
    non_seq_res: EvalResult,
    seq_res: EvalResult,
    output_folder: Path,
    changepoints: np.ndarray,
    args: Any,
):
    print("Plotting prediction...")

    def format_ax(ax, min_x: int = 0, n_scale: int = 1, format: bool = True):
        ax.set_ylabel("Regions")
        if format:
            ax.set_ylim(min_x, args.n_labels)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_major_locator(ticker.MultipleLocator(n_scale))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig, (ax_non_seq, ax_seq) = plt.subplots(2, 1, figsize=(16, 7))

    ax_non_seq.plot(
        non_seq_res.y_origs,
        label="Original",
        alpha=0.5,
        # color="#1f77b4",
        linestyle="--",
    )
    ax_non_seq.plot(
        non_seq_res.y_preds,
        label="Predictions",
        alpha=0.5,
        # color="#ff7f0e",
        linestyle="-",
    )
    ax_non_seq.set_title("Non Sequential")
    ax_non_seq.legend()
    format_ax(ax_non_seq)

    ax_seq.plot(
        seq_res.y_origs,
        label="Original",
        alpha=0.5,
        # color="#1f77b4",
        linestyle="--",
    )
    ax_seq.plot(
        seq_res.y_preds,
        label="Predictions",
        alpha=0.5,
        # color="#ff7f0e",
        linestyle="-",
    )
    ax_seq.set_title("Sequential")
    ax_seq.legend()
    format_ax(ax_seq)

    if len(changepoints) > 0:
        for cp in changepoints:
            ax_non_seq.axvline(x=cp, color="r", linestyle="--", alpha=0.7)
            ax_seq.axvline(x=cp, color="r", linestyle="--", alpha=0.7)

    prefix_label = "[Local]" if args.local else "[Global]"
    fig.suptitle(f"{prefix_label} Sequential vs Non-Sequential ({args.strategy})")

    fig.tight_layout()
    fig.savefig(output_folder / "prediction.png", dpi=100)
    plt.close(fig)


def main(args):
    data_path = Path(args.data)
    changepoints_path = data_path.parent / f"{data_path.stem}_change.csv"

    changepoints = np.array([])
    if changepoints_path.exists():
        # changepoints file contains of two columns: # rows, timestamp
        changepoints = np.loadtxt(changepoints_path, delimiter=",")
        changepoints = changepoints.astype(int)[:, 0]  # pick # rows

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

    non_seq_eval_res = EvalResult(**non_seq_res)
    seq_eval_res = EvalResult(**seq_res)

    plot_prediction(
        non_seq_eval_res,
        seq_eval_res,
        output_folder,
        changepoints=changepoints,
        args=args,
    )
    plot_auc_roc(non_seq_eval_res, seq_eval_res, output_folder, args=args)
    plot_diff(non_seq_eval_res, seq_eval_res, output_folder, args=args)


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
    parser.add_argument(
        "--type",
        type=str,
        choices=[
            "seq-non-seq",
            "local-global",
        ],
        default="test",
    )

    args = parser.parse_args()
    main(args)
