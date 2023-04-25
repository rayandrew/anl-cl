from pathlib import Path
from typing import Literal, Union

import torch

from avalanche.benchmarks.utils import make_classification_dataset

import numpy as np

from src.utils.general import split_evenly_by_classes

from .base import AlibabaDataset


class BaseAlibabaMachineDataset(AlibabaDataset):
    def __init__(
        self,
        filename: Union[str, Path],
        n_labels: int,
        train_ratio: float = AlibabaDataset.TRAIN_RATIO,
        y: Literal["cpu", "mem", "disk"] = "cpu",
        subset: Literal["training", "testing", "all"] = "training",
    ):
        """Dataset for Alibaba Machine dataset

        Args:
            filename (Union[str, Path]): Path to the dataset file
            n_labels (int): Number of labels to use
            train_ratio (float, optional): Ratio of training data. Defaults to AlibabaDataset.TRAIN_RATIO.
            y (Literal["cpu", "mem", "disk"], optional): Variable to predict. Defaults to "cpu".
            subset (Literal["training", "testing", "all"], optional): Subset of the dataset. Defaults to "all".
        """
        assert subset in ["training", "testing", "all"]
        assert y in ["cpu", "mem", "disk"]
        self.filename = filename
        self.train_ratio = train_ratio
        self.n_labels = n_labels
        self.y_var = y
        self.subset = subset
        self._n_experiences = None
        self._load_data()

    def _clean_data(self, data):
        # do not need code below as the data should come without header
        # data = data[1:]
        # ts = data[:, 1] # timestamp
        # TODO: this is hacky solution, need to fix
        # X need to accomodate "data" and dist_labels together
        # such that we can use `train_test_split` to split the data
        # in the future, we should not use these two variables together
        label_index = 8
        if self.y_var == "cpu":
            label_index = 2
        elif self.y_var == "mem":
            label_index = 3

        # filter -1 and 101
        lower_bound = 0
        upper_bound = 100
        bad_mask = (
            np.isnan(data[:, label_index])
            | ~np.isfinite(data[:, label_index])
            | (data[:, label_index] < lower_bound)
            | (data[:, label_index] > upper_bound)
        )
        data = data[~bad_mask]

        dist_labels = data[:, -1]
        labels = data[:, label_index]
        labels = labels.astype(int)
        # normalize labels
        labels = np.maximum(labels, 0)
        labels = np.minimum(labels, 99)
        labels = labels // self.n_labels

        data = np.delete(data, label_index, axis=1)
        data = data[
            :, 2:-1
        ]  # remove machine id + timestamp + dist_label

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


class BaseAlibabaMachineSequenceDataset(BaseAlibabaMachineDataset):
    def __init__(
        self,
        seq_len: int = 5,
        univariate: bool = False,
        flatten: bool = False,
        *args,
        **kwargs,
    ):
        self.seq_len = seq_len
        self.univariate = univariate
        self.flatten = flatten
        super(BaseAlibabaMachineSequenceDataset, self).__init__(
            *args, **kwargs
        )

    # def _process_data(self, data, labels, dist_labels):
    #     Xs = []
    #     Dists = []
    #     i = 0
    #     if self.univariate:
    #         while i + self.seq_len <= data.shape[0]:
    #             ys = labels[i : i + self.seq_len]
    #             Xs.append((ys[:-1], ys[-1]))
    #             dists = dist_labels[i : i + self.seq_len]
    #             dist = int(dists[-1])
    #             Dists.append(dist)
    #             i += 1
    #     else:
    #         # multivariate
    #         while i + self.seq_len <= data.shape[0]:
    #             mat = data[i : i + self.seq_len, :]
    #             x = mat.flatten()
    #             ys = labels[i : i + self.seq_len]
    #             x = np.append(x, ys[:-1])
    #             y = ys[-1]
    #             dists = dist_labels[i : i + self.seq_len]
    #             dist = int(dists[-1])
    #             Xs.append((x, y))
    #             Dists.append(dist)
    #             i += 1
    #     return Xs, Dists

    def _process_data(self, data, labels, dist_labels):
        Xs = []
        Dists = []
        i = 0
        if self.univariate:
            # TODO: add non-flatten options
            while i + self.seq_len <= data.shape[0]:
                ys = labels[i : i + self.seq_len]
                Xs.append((ys[:-1], ys[-1]))
                dists = dist_labels[i : i + self.seq_len]
                dist = int(dists[-1])
                Dists.append(dist)
                i += 1
        else:
            # multivariate
            if self.flatten:
                while i + self.seq_len <= data.shape[0]:
                    mat = data[i : i + self.seq_len, :]
                    x = mat.flatten()
                    ys = labels[i : i + self.seq_len]
                    x = np.append(x, ys[:-1])
                    y = ys[-1]
                    dists = dist_labels[i : i + self.seq_len]
                    dist = int(dists[-1])
                    Xs.append((x, y))
                    Dists.append(dist)
                    i += 1
            else:
                # non-flatten
                while i + self.seq_len <= data.shape[0]:
                    mat = data[i : i + self.seq_len, :]
                    x = mat
                    # x = mat.flatten()
                    ys = labels[i : i + self.seq_len]
                    y = ys[-1]
                    ydata = ys[:-1].reshape(-1, 1)
                    num_missing = self.seq_len - len(ydata)
                    zeros = np.zeros((num_missing, 1))
                    ydata = np.concatenate([ydata, zeros], axis=0)
                    x = np.concatenate([x, ydata], axis=1)
                    dists = dist_labels[i : i + self.seq_len]
                    dist = int(dists[-1])
                    Xs.append((x, y))
                    Dists.append(dist)
                    i += 1
        return Xs, Dists


def alibaba_machine_sequence_collate(batch):
    tensors, targets, t_labels = [], [], []
    for x, region, _dist_label, t_label in batch:
        tensors += [x.t()]
        targets += [torch.tensor(region)]
        t_labels += [torch.tensor(t_label)]
    # tensors = [item.t() for item in tensors]
    tensors = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0.0
    )
    targets = torch.stack(targets)
    t_labels = torch.stack(t_labels)
    return tensors, targets, t_labels


def AlibabaMachineDataset(
    filename: str,
    n_labels: int = 10,
    subset="train",
    y: Literal["cpu", "mem", "disk"] = "cpu",
):
    dataset = BaseAlibabaMachineDataset(
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


def AlibabaMachineSequenceDataset(
    filename: str,
    n_labels: int = 10,
    subset="train",
    y: Literal["cpu", "mem", "disk"] = "cpu",
    seq_len: int = 5,
    univariate: bool = False,
):
    dataset = BaseAlibabaMachineSequenceDataset(
        filename=filename,
        n_labels=n_labels,
        subset=subset,
        y=y,
        seq_len=seq_len,
        univariate=univariate,
    )

    # NOTE: might be slow in the future
    dist_labels = [datapoint[2] for datapoint in dataset]
    return (
        make_classification_dataset(
            dataset,
            collate_fn=alibaba_machine_sequence_collate,
            targets=dist_labels,
        ),
        dataset,
    )


__all__ = [
    "BaseAlibabaMachineDataset",
    "BaseAlibabaMachineSequenceDataset",
    "AlibabaMachineDataset",
    "AlibabaMachineSequenceDataset",
    "alibaba_machine_sequence_collate",
]

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # parser.add_argument("-d", "--data", type=str, default="data/mu_dist/m_25.csv")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="out_preprocess/m_25/m_25.csv",
    )
    parser.add_argument("-n", "--n_labels", type=int, default=10)
    parser.add_argument(
        "-m",
        "--subset",
        type=str,
        choices=["training", "testing", "all"],
        default="training",
    )
    parser.add_argument(
        "-y",
        type=str,
        choices=["cpu", "mem", "disk"],
        default="cpu",
    )
    parser.add_argument(
        "-s",
        "--seq",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--seq_len",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--univariate",
        action="store_true",
    )
    args = parser.parse_args()

    dataset, n_exp, input_size = AlibabaMachineSequenceDataset(
        filename=args.data,
        n_labels=args.n_labels,
        y=args.y,
        subset=args.subset,
        seq_len=args.seq_len,
        univariate=args.univariate,
    )
    print("INPUT SIZE", input_size)
    print("N EXPERIENCES", n_exp)
    # print("TARGETS", np.unique(dataset.targets))
    # print("OUTPUTS", np.unique(dataset.outputs))
    print("LENGTH", len(dataset))
    for d in dataset:
        print(d)
        break

    # print(dataset[3911])
    # print(dataset[3912])
    # print(dataset[3913])
    # print(dataset[3914])
    # print(dataset[3915])
    # print(dataset[3916])
    # print(dataset[3917])
