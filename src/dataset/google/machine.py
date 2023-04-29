from pathlib import Path
from typing import Literal, Union

import torch

from avalanche.benchmarks.utils import make_classification_dataset

import numpy as np

from src.dataset.base import TDatasetSubset
from src.utils.general import split_evenly_by_classes

from .base import GoogleDataset

TYVariable = Literal["cpu", "mem"]


class BaseGoogleMachineDataset(GoogleDataset):
    def __init__(
        self,
        filename: Union[str, Path],
        n_labels: int,
        train_ratio: float = GoogleDataset.TRAIN_RATIO,
        y: TYVariable = "cpu",
        subset: TDatasetSubset = "training",
    ):
        """Dataset for Google Machine dataset

        Args:
            filename (Union[str, Path]): Path to the dataset file
            n_labels (int): Number of labels to use
            train_ratio (float, optional): Ratio of training data. Defaults to GoogleDataset.TRAIN_RATIO.
            y (Literal["cpu", "mem"], optional): Variable to predict. Defaults to "cpu".
            subset (Literal["training", "testing", "all"], optional): Subset of the dataset. Defaults to "all".
        """
        assert subset in ["training", "testing", "all"]
        assert y in ["cpu", "mem"]
        self.filename = filename
        self.train_ratio = train_ratio
        self.n_labels = n_labels
        self.y_var = y
        self.subset = subset
        self._n_experiences = None
        self._load_data()

    def _clean_data(self, data):
        # do not need code below as the data should come without header
        data = data[1:]
        # ts = data[:, 1] # timestamp
        # TODO: this is hacky solution, need to fix
        # X need to accomodate "data" and dist_labels together
        # such that we can use `train_test_split` to split the data
        # in the future, we should not use these two variables together
        if self.y_var == "cpu":
            label_index = 2
        elif self.y_var == "mem":
            label_index = 3

        dist_labels = data[:, -1]
        labels = data[:, label_index]
        # Normalize labels from 0 to 10
        maxi = np.max(labels)
        mini = np.min(labels)
        labels -= mini
        labels /= (maxi - mini)
        # Now label is from 0 to 1

        # Prevent label with value 10, because we wanna cast to Int
        labels *= 9.98

        # ROund labels before casting to int
        labels = labels.astype(int)
        unique_labels, counts = np.unique(labels, return_counts=True)

        min_count_idx = np.argmin(counts)
        least_common_value = unique_labels[min_count_idx]
        least_common_count = counts[min_count_idx]
        print("The least common value in labels is",
              least_common_value, "with a count of", least_common_count)
        data = np.delete(data, label_index, axis=1)
        data = data[
            :, 2:-1
        ]  # remove start_time + end_time + dist_label

        return data, labels, dist_labels

    def _process_data(self, data, labels, dist_labels):
        Xs = []
        Dists = []
        for i, d in enumerate(data):
            x = d.flatten()
            y = labels[i]
            dist = int(dist_labels[i])
            Xs.append((x, y))
            Dists.append(dist)
        return Xs, Dists

    def _load_data(self):
        assert self.subset in ["training", "testing", "all"]
        data = np.genfromtxt(self.filename, delimiter=",")
        data = self._process_nan(data)
        data, labels, dist_labels = self._clean_data(data)

        unique_labels, counts = np.unique(dist_labels, return_counts=True)
        min_count_idx = np.argmin(counts)

        least_common_value = unique_labels[min_count_idx]
        least_common_count = counts[min_count_idx]

        print("The least common value in labels is",
              least_common_value, "with a count of", least_common_count)
        Data, Dists = self._process_data(data, labels, dist_labels)

        if self.subset == "all":
            X = [d[0] for d in Data]
            y = [d[1] for d in Data]
            self.data = X
            self.dist_labels = Dists
            self.outputs = y
            return
        (
            Data_train,
            Data_test,
            Dist_train,
            Dist_test,
        ) = split_evenly_by_classes(
            Data, Dists, train_ratio=self.train_ratio
        )

        X_train = [d[0] for d in Data_train]
        y_train = [d[1] for d in Data_train]
        dist_labels_train = Dist_train

        X_test = [d[0] for d in Data_test]
        y_test = [d[1] for d in Data_train]
        dist_labels_test = Dist_test

        if self.subset == "training":
            self.data = X_train
            self.dist_labels = dist_labels_train
            self.outputs = y_train
        elif self.subset == "testing":
            self.data = X_test
            self.dist_labels = dist_labels_test
            self.outputs = y_test

    def input_size(self) -> int:
        if self.data is None:
            raise ValueError("Dataset not loaded yet")
        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
        return len(self.data[0])

    def n_experiences(self) -> int:
        if self._n_experiences is None:
            self._n_experiences = len(np.unique(self.dist_labels))
        return self._n_experiences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        dist_label = self.dist_labels[index]
        label = self.outputs[index]
        data_tensor = torch.tensor(data, dtype=torch.float32)
        return data_tensor, label, dist_label


def GoogleMachineDataset(
    filename: str,
    univariate: str,
    n_labels: int = 10,
    subset: TDatasetSubset = "train",
    y: TYVariable = "cpu",
    seq_len: int = 0,
):
    dataset = BaseGoogleMachineDataset(
        filename=filename,
        n_labels=n_labels,
        subset=subset,
        y=y,
    )
    # NOTE: might be slow in the future
    dist_labels = [datapoint[2] for datapoint in dataset]
    return (
        make_classification_dataset(
            dataset,
            targets=dist_labels,
        ),
        dataset,
    )


__all__ = [
    "BaseGoogleMachineDataset",
    "GoogleMachineDataset",
]
