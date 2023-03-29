from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Sequence, Literal

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import AlibabaSchedulerDataset, AlibabaMachineDataset
from plot_utils import get_y_label


def plot_diff(
    y_var: Literal["cpu", "mem", "disk"],
    diffs: Dict[int, int],
    output_folder: Path,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for diff in diffs.keys():
        ax.bar(diff, diffs[diff], color="maroon", width=0.4)

    ax.set_xlabel("Diff")
    ax.set_ylabel("Frequency")
    ax.set_title("Diff between actual and predicted label")

    fig.suptitle(f"Prediction of {get_y_label(y_var)}")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.savefig(output_folder / "diff.png")
    plt.close(fig)


def plot_inference(
    args,
    y_origs: Sequence[int],
    y_regions: Sequence[int],
    y_preds: Sequence[int],
    tss: Sequence[int],
    y_var: Literal["cpu", "mem", "disk"],
    output_folder: Path,
):
    def format_ax(
        ax, min_x: int = 0, max_x: int = 10, n_scale: int = 1, format: bool = True
    ):
        ax.set_xlabel("Time")
        if format:
            ax.set_ylim(min_x, max_x)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(n_scale))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig, (ax_region, ax_pred) = plt.subplots(2, 1, figsize=(25, 10))

    tss = list(range(len(y_regions)))
    # ax_orig.scatter(tss, y_origs)
    # ax_region.plot(tss, y_regions)
    ax_region.scatter(tss, y_regions)
    # ax_pred.plot(tss, y_preds, color="maroon")
    ax_pred.scatter(tss, y_preds, color="maroon")

    # format_ax(ax_orig, format=False)
    # ax_orig.set_ylabel("Percentage (%)")
    format_ax(ax_region, max_x=args.n_labels)
    ax_region.set_ylabel("Region")
    format_ax(ax_pred, max_x=args.n_labels)
    ax_pred.set_ylabel("Region")

    fig.suptitle(f"Prediction of {get_y_label(y_var)}")
    fig.tight_layout()

    fig.savefig(output_folder / "inference.png")
    plt.close(fig)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model_path)
    model.to(device)
    model.eval()

    dataset = AlibabaMachineDataset(
        filename=args.filename,
        # train=False,
        n_labels=args.n_labels,
        mode="predict",
        y=args.y,
        seq=args.seq,
        seq_len=args.seq_len,
        univariate=args.univariate,
    )

    diffs = {}
    y_origs = []
    y_regions = []
    y_preds = []
    tss = []

    for i in range(-args.n_labels, args.n_labels + 1, 1):
        diffs[i] = 0

    for i, (x, _, y) in enumerate(dataset):
        # x = torch.from_numpy(x.reshape(1, -1)).to(device).float()
        x = x.to(device).reshape(1, -1)

        y_pred = model(x)
        pred_label = torch.argmax(y_pred, dim=1).item()
        diff = (y - pred_label).item()
        diffs[diff] += 1
        y_preds.append(pred_label)
        y_regions.append(y)
        # y_origs.append(y_orig)
        # tss.append(ts)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    if args.plot:
        plot_diff(y_var=args.y, diffs=diffs, output_folder=output_folder)
        plot_inference(
            args,
            y_origs=y_origs,
            y_regions=y_regions,
            y_preds=y_preds,
            tss=tss,
            y_var=args.y,
            output_folder=output_folder,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True)
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, default="out")
    parser.add_argument("-nl", "--n_labels", type=int, default=10)
    parser.add_argument(
        "-y",
        type=str,
        choices=["cpu", "mem", "disk"],
        default="cpu",
    )
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("--seq", action="store_true")
    parser.add_argument("--seq_len", type=int, default=3)
    parser.add_argument("--univariate", action="store_true")
    args = parser.parse_args()

    main(args)
