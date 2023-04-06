import random
from pathlib import Path
from typing import Literal, Union

import torch

import numpy as np

from .base import AlibabaDataset


class AlibabaSchedulerDataset(AlibabaDataset):
    FEATURE_COLUMNS = [
        # "time_stamp",
        "cpu_avg",
        "cpu_max",
        "mem_avg",
        "mem_max",
        "plan_cpu",
        "plan_mem",
        # "cpu_util_percent",
        # "mem_util_percent",
        # "disk_io_percent"
    ]

    def __init__(
        self,
        filename: Union[str, Path],
        n_labels: int,
        train_ratio: float = AlibabaDataset.TRAIN_RATIO,
        y: Literal["cpu", "mem", "disk"] = "cpu",
        mode: Literal["train", "test", "predict"] = "train",
        seq: bool = False,
        seq_len: int = 2,
        univariate: bool = False,
    ):
        """Dataset for Alibaba Machine Scheduler

        Args:
            filename (Union[str, Path]): path to the dataset
            n_labels (int): number of labels to use
            train_ratio (float, optional): ratio of training data. Defaults to AlibabaDataset.TRAIN_RATIO.
            y (Literal["cpu", "mem", "disk"], optional): which variable to predict. Defaults to "cpu".
            mode (Literal["train", "test", "predict"], optional): which mode to use. Defaults to "train".
            seq (bool, optional): whether to use sequence data. Defaults to False.
            seq_len (int, optional): length of sequence. Defaults to 2.
            univariate (bool, optional): whether to use univariate data. Defaults to False.
        """
        assert mode in ["train", "test", "predict"]
        assert y in ["cpu", "mem", "disk"]
        assert seq_len >= 2
        if univariate:
            assert seq, "Univariate data must be sequence data"
        self.filename = filename
        self.train_ratio = train_ratio
        self.n_labels = n_labels
        self.y_var = y
        self.mode = mode
        self.seq = seq
        self.seq_len = seq_len
        self.univariate = univariate
        self._n_experiences = None
        self._load_data()

    def _augment_data(self, data):
        data = data[1:]  # skip header
        data = data[:, 1:]  # skip timestamp
        ts = data[:, 0]
        new_data = []
        label_index = -2
        if self.y_var == "cpu":
            label_index = -4
        elif self.y_var == "mem":
            label_index = -3

        if self.seq_len > 1:
            i = 0
            while i + self.seq_len <= data.shape[0]:
                mat = data[i : i + self.seq_len, :]
                item = mat.flatten()
                data = item[:-3]
                label = item[label_index]
                dist_label = item[-1]
                i += 1
                new_data.append((data, label, int(dist_label), ts[i]))
        else:
            for i, data in enumerate(data):
                item = data.flatten()
                data = item[:-3]
                label = item[label_index]
                dist_label = item[-1]
                new_data.append((data, label, int(dist_label), ts[i]))
        return new_data

    def _prepare_targets(self, data):
        targets = []
        for i in range(len(data)):
            targets.append(int(data[i][2]))
        return targets

    def input_size(self) -> int:
        if self.data is None:
            raise ValueError("Dataset is not loaded yet")
        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
        return len(self.data[0][0])

    def _load_data(self):
        data = np.genfromtxt(self.filename, delimiter=",")
        data = self._process_nan(data)
        new_data = self._augment_data(data)

        train_size = int(len(new_data) * self.train_ratio)
        assert self.mode in ["train", "test", "predict"]
        if self.mode == "train":
            self.data = new_data[:train_size]
            random.shuffle(self.data)
            self.targets = self._prepare_targets(self.data)
        elif self.mode == "test":
            self.data = new_data[train_size:]
            random.shuffle(self.data)
            self.targets = self._prepare_targets(self.data)
        else:
            self.data = new_data
            self.targets = self._prepare_targets(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index][0]
        label = np.array(int(min(self.data[index][1], 99) / 10))
        dist_label = self.data[index][2]
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.from_numpy(label)
        dist_label_tensor = torch.from_numpy(np.array(dist_label))
        return data_tensor, dist_label_tensor, label_tensor


__all__ = ["AlibabaSchedulerDataset"]
