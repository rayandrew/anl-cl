from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from scipy.stats import gaussian_kde

from src.utils.dataset import get_label_idx_from_data


def plot_density(input: Path, y: str, dataset_name: str):
    label_index = get_label_idx_from_data(dataset_name, y)
    df = pd.read_csv(input)
    g = df.groupby(df.columns[-1])

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_title(f"Drift Detection on {dataset_name} Dataset")
    ax.set_xlabel("Data")
    ax.set_ylabel("Density")

    # count = 0
    for label, group in g:
        # if count == 5:
        #     break

        # print(group.values[:, label_index])
        ax = sns.kdeplot(
            # group.values[:, -1],
            group.values[:, label_index],
            bw=0.5,
            label=label,
            ax=ax,
        )
        # count += 1

    return fig, ax


@hydra.main(
    config_path="../../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    outfile_path = Path(cfg.filename)
    output_path = outfile_path.parent
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = Path(cfg.dataset.raw_path)

    # NOTE: this path is duplicated from drift_detection.py
    changepoints = np.genfromtxt(
        output_path / f"{input_path.stem}_{cfg.y}_cp.csv",
        delimiter=",",
    )

    if len(changepoints) == 0:
        print("No changepoints found.")
        return

    # NOTE: this path is duplicated from drift_detection.py
    # orig_data = np.genfromtxt(outfile_path, delimiter=",")

    fig, _ = plot_density(
        input=outfile_path,
        y=cfg.y,
        dataset_name=cfg.dataset.name,
    )

    fig.savefig(
        output_path / f"{input_path.stem}_{cfg.y}_density.png",
        dpi=300,
    )

    # for cp in changepoints:


if __name__ == "__main__":
    main()

__all__ = ["plot_density"]
