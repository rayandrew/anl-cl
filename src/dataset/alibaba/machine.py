from pathlib import Path
from typing import Literal, Union

import torch

from avalanche.benchmarks.utils import make_classification_dataset

import numpy as np

from src.utils.general import split_evenly_by_classes

from .base import AlibabaDataset


class AlibabaMachineDataset(AlibabaDataset):
    CACHE = {}

    def __init__(
        self,
        filename: Union[str, Path],
        n_labels: int,
        train_ratio: float = AlibabaDataset.TRAIN_RATIO,
        y: Literal["cpu", "mem", "disk"] = "cpu",
        subset: Literal["training", "testing", "all"] = "training",
        seq: bool = False,
        seq_len: int = 2,
        univariate: bool = False,
    ):
        """Dataset for Alibaba Machine dataset

        Args:
            filename (Union[str, Path]): Path to the dataset file
            n_labels (int): Number of labels to use
            train_ratio (float, optional): Ratio of training data. Defaults to AlibabaDataset.TRAIN_RATIO.
            y (Literal["cpu", "mem", "disk"], optional): Variable to predict. Defaults to "cpu".
            subset (Literal["training", "testing", "all"], optional): Subset of the dataset. Defaults to "all".
            seq (bool, optional): Whether to use sequence data. Defaults to False.
            seq_len (int, optional): Sequence length for sequence data. Defaults to 2.
        """
        assert subset in ["training", "testing", "all"]
        assert y in ["cpu", "mem", "disk"]
        assert seq_len >= 2
        if univariate:
            assert seq, "Univariate data must be sequence data"
        self.filename = filename
        self.train_ratio = train_ratio
        self.n_labels = n_labels
        self.y_var = y
        self.subset = subset
        self.seq = seq
        self.seq_len = seq_len
        self.univariate = univariate
        self._n_experiences = None
        self._load_data()

    def _augment_data(self, data):
        # do not need code below as the data should come without header
        # data = data[1:]
        # ts = data[:, 1] # timestamp
        # TODO: this is hacky solution, need to fix
        # X need to accomodate "data" and dist_labels together
        # such that we can use `train_test_split` to split the data
        # in the future, we should not use these two variables together
        Xs = []
        Dists = []

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

        if self.seq:
            i = 0
            if self.univariate:
                while i + self.seq_len <= data.shape[0]:
                    ys = labels[i : i + self.seq_len]
                    Xs.append((ys[:-1], ys[-1]))
                    dists = dist_labels[i : i + self.seq_len]
                    dist = int(dists[-1])
                    Dists.append(dist)
                    i += 1
            else:
                # multivariate
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
            for i, d in enumerate(data):
                x = d.flatten()
                y = labels[i]
                dist = int(dist_labels[i])
                Xs.append((x, y))
                Dists.append(dist)

        return Xs, Dists

    def _load_data(self):
        assert self.subset in ["training", "testing", "all"]

        additional_key = "non-seq"
        if self.seq:
            additional_key = "seq"
            if self.univariate:
                additional_key += "_uni"
            else:
                additional_key += "_mult"

        key = (
            self.filename,
            self.train_ratio,
            self.subset,
            additional_key,
        )
        if key in self.CACHE:
            self.data, self.targets, self.outputs = self.CACHE[key]

        data = np.genfromtxt(self.filename, delimiter=",")
        data = self._process_nan(data)
        Data, Dists = self._augment_data(data)

        if self.subset == "all":
            X = [d[0] for d in Data]
            y = [d[1] for d in Data]
            self.data = X
            self.targets = Dists
            self.outputs = y
        else:
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

            self.CACHE[(self.filename, self.train_ratio, "train")] = (
                X_train,
                dist_labels_train,
                y_train,
            )
            self.CACHE[(self.filename, self.train_ratio, "test")] = (
                X_test,
                dist_labels_test,
                y_test,
            )

            if self.subset == "training":
                self.data = X_train
                self.targets = dist_labels_train
                self.outputs = y_train
            elif self.subset == "testing":
                self.data = X_test
                self.targets = dist_labels_test
                self.outputs = y_test

    def input_size(self) -> int:
        if self.data is None:
            raise ValueError("Dataset not loaded yet")
        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
        return len(self.data[0])

    def n_experiences(self) -> int:
        if self._n_experiences is None:
            self._n_experiences = len(np.unique(self.targets))
        return self._n_experiences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        dist_label = self.targets[index]
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.from_numpy(np.array(self.outputs[index]))
        dist_label_tensor = torch.from_numpy(np.array(dist_label))
        return data_tensor, label_tensor, dist_label_tensor


def alibaba_machine_collate(batch):
    tensors, targets, t_labels = [], [], []
    for x, region, dist_label, _ in batch:
        tensors += [x]
        targets += [torch.tensor(region)]
        t_labels += [torch.tensor(dist_label)]
    tensors = [item.t() for item in tensors]
    tensors = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0.0
    )
    targets = torch.stack(targets)
    t_labels = torch.stack(t_labels)
    return tensors, targets, t_labels


# TODO: remove this later
def AlibabaMachineDatasetF(
    filename: str,
    n_labels: int = 10,
    subset="train",
    y: Literal["cpu", "mem", "disk"] = "cpu",
    seq: bool = False,
    seq_len: int = 5,
    univariate: bool = False,
):
    dataset = AlibabaMachineDataset(
        filename=filename,
        n_labels=n_labels,
        subset=subset,
        y=y,
        seq=False,
        seq_len=seq_len,
        univariate=univariate,
    )
    return dataset, dataset.n_experiences(), dataset.input_size()


# TODO: simplify this later
def AlibabaMachineSequence(
    filename: str,
    n_labels: int = 10,
    subset="train",
    y: Literal["cpu", "mem", "disk"] = "cpu",
    seq: bool = False,
    seq_len: int = 5,
    univariate: bool = False,
):
    dataset = AlibabaMachineDataset(
        filename=filename,
        n_labels=n_labels,
        subset=subset,
        y=y,
        seq=False,
        seq_len=seq_len,
        univariate=univariate,
    )
    # return dataset, dataset.n_experiences(), dataset.input_size()

    dist_labels = [datapoint[2] for datapoint in dataset]
    return (
        make_classification_dataset(
            dataset,
            collate_fn=alibaba_machine_collate,
            # targets=dist_labels,
        ),
        dataset.n_experiences(),
        dataset.input_size(),
    )


__all__ = [
    "AlibabaMachineDataset",
    "AlibabaMachineSequence",
    "AlibabaMachineDatasetF",
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
        choices=["train", "test", "predict"],
        default="train",
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
        default=2,
    )
    parser.add_argument(
        "--univariate",
        action="store_true",
    )
    args = parser.parse_args()

    dataset = AlibabaMachineDataset(
        filename=args.data,
        n_labels=args.n_labels,
        y=args.y,
        subset=args.subset,
        seq=args.seq,
        seq_len=args.seq_len,
        univariate=args.univariate,
    )
    print("INPUT SIZE", dataset.input_size())
    print("N EXPERIENCES", dataset.n_experiences())
    print("TARGETS", np.unique(dataset.targets))
    print("OUTPUTS", np.unique(dataset.outputs))
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
