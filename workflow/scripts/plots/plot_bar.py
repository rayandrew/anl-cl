from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt

from src.utils.general import set_seed
from src.utils.logging import logging, setup_logging
from src.utils.summary import generate_summary

if TYPE_CHECKING:
    snakemake: Any = None

setup_logging(snakemake.log[0])
log = logging.getLogger(__name__)


def plot_bar(
    scenario: int, data: Sequence[float], labels: Sequence[str]
):
    fig, ax = plt.subplots()
    x = range(len(data))
    ax.bar(x, data)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ybound(0, 1)
    # ax.bar(labels, data)
    ax.set_ylabel("AUROC")
    ax.set_title(f"Accuracy per scenario {scenario}")
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


def main():
    config = snakemake.config
    # scenario = snakemake.params.scenario
    # dataset = snakemake.params.dataset

    set_seed(config.get("seed", 0))
    input_paths = [Path(str(input)) for input in snakemake.input]
    output_folder = Path(str(snakemake.output))
    output_folder.mkdir(parents=True, exist_ok=True)

    avg_aurocs = defaultdict(list)
    labels = []
    for input in input_paths:
        label = label_from_path(input)
        summary = generate_summary(input / "train_results.json")
        print(summary.avg_auroc)
        for i in range(len(summary.avg_auroc)):
            avg_aurocs[i].append(summary.avg_auroc[i])
        labels.append(label)

    for i in avg_aurocs:
        print(i, len(avg_aurocs[i]), labels)
        fig = plot_bar(i, avg_aurocs[i], labels)
        fig.savefig(output_folder / f"auroc_{i}.png")


if __name__ == "__main__":
    main()
