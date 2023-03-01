from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Sequence, Literal

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import AlibabaDataset


def get_y_label(
    y_var: Literal["cpu_util_percent", "mem_util_percent", "disk_io_percent"]
):
    if y_var == "cpu_util_percent":
        return "CPU Utilization"
    elif y_var == "mem_util_percent":
        return "Memory Utilization"
    elif y_var == "disk_io_percent":
        return "Disk IO Utilization"
    else:
        raise ValueError(f"Unknown y_var: {y_var}")


def plot_diff(
    y_var: Literal["cpu_util_percent", "mem_util_percent", "disk_io_percent"],
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
    y_var: Literal["cpu_util_percent", "mem_util_percent", "disk_io_percent"],
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

    fig, (ax_orig, ax_region, ax_pred) = plt.subplots(3, 1, figsize=(25, 15))

    ax_orig.scatter(tss, y_origs)
    ax_region.scatter(tss, y_regions)
    ax_pred.scatter(tss, y_preds)

    format_ax(ax_orig, format=False)
    ax_orig.set_ylabel("Percentage (%)")
    format_ax(ax_region, max_x=args.n_labels)
    ax_region.set_ylabel("Region")
    format_ax(ax_pred, max_x=args.n_labels)
    ax_pred.set_ylabel("Region")

    fig.suptitle(f"Prediction of {get_y_label(y_var)}")

    fig.savefig(output_folder / "inference.png")
    plt.close(fig)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model_path)
    model.to(device)
    model.eval()

    dataset = AlibabaDataset(
        filename=args.filename, train=False, n_labels=args.n_labels, eval=True, y=args.y
    )

    diffs = {}
    y_origs = []
    y_regions = []
    y_preds = []
    tss = []

    for i in range(-args.n_labels, args.n_labels + 1, 1):
        diffs[i] = 0

    for i, (x, y, y_orig, ts) in enumerate(dataset):
        x = torch.from_numpy(x.reshape(1, -1)).to(device).float()

        y_pred = model(x)
        pred_label = torch.argmax(y_pred, dim=1).item()
        diff = y - pred_label
        diffs[diff] += 1
        y_preds.append(pred_label)
        y_regions.append(y)
        y_origs.append(y_orig)
        tss.append(ts)

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
        choices=["cpu_util_percent", "mem_util_percent", "disk_io_percent"],
        default="cpu_util_percent",
    )
    parser.add_argument("-p", "--plot", action="store_true")
    args = parser.parse_args()

    main(args)
