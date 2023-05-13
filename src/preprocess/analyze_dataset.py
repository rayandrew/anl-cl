import argparse
from collections import deque
from pathlib import Path
from typing import Sequence

import hydra
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.decomposition import PCA


@hydra.main(
    config_path="../../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    data = pd.read_csv(cfg.dataset.path)
    data = data.drop(columns=data.columns[[0, 1]])
    output = data.pop(cfg.y)

    n_chunks = 10
    # n_chunks = int(len(data) / n_chunks)
    data_chunks = np.array_split(data, n_chunks)
    # data_chunks = [
    #     data[i : i + n_chunks]
    #     for i in range(0, len(data), int(len(data) / (n_chunks - 1)))
    # ]

    print(len(data_chunks))

    print("FEATURES", data.columns)

    # pca = PCA(n_components=2)
    # pca.fit(data)
    # transformed_data = pca.transform(data)

    # plot 2-dimension data
    n_row = 2
    # fig, ax = plt.subplots(
    #     n_row, int(n_chunks / n_row), figsize=(30, 10)
    # )
    fig, ax = plt.subplots(figsize=(30, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(data_chunks)))
    for i, chunk in enumerate(data_chunks):
        row = i % n_row
        col = i // n_row
        print(i, row, col, len(chunk))
        pca = PCA(n_components=2)
        pca.fit(chunk)
        transformed_data = pca.transform(chunk)
        # ax[row][col].scatter(
        #     transformed_data[:, 0], transformed_data[:, 1]
        # )
        ax.scatter(
            transformed_data[:, 0],
            transformed_data[:, 1],
            color=colors[i],
            # c=[i for _ in range(len(chunk))],
            # cmap="Greens",
        )

    fig.tight_layout()
    # fig.show()
    fig.savefig("pca.png")
    plt.close(fig)

    # ax.scatter(transformed_data[:, 0], transformed_data[:, 1])

    # transformed_data.plot(
    #     kind="scatter", x=0, y=1, c=output, cmap="Spectral"
    # )
    plt.show()

    # print(data)
    # print(output)


if __name__ == "__main__":
    main()
