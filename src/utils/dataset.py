from typing import Literal

import numpy as np
import numpy.typing as npt

TDatasetName = Literal["alibaba", "google"]
TOutputVar = Literal["cpu", "mem", "disk"]


def get_alibaba_output_idx(y: TOutputVar):
    label_index = 8
    if y == "cpu":
        label_index = 2
    elif y == "mem":
        label_index = 3
    return label_index


def get_alibaba_output(data: npt.ArrayLike, y: TOutputVar):
    label_index = get_alibaba_output_idx(y)
    data = data[:, label_index]
    return data


def get_google_output_idx(y: TOutputVar):
    if cfg.y == "cpu":
        label_index = 2
    elif cfg.y == "mem":
        label_index = 3
    else:
        raise ValueError("Invalid y value")

    return label_index


def get_google_output(data: npt.ArrayLike, y: TOutputVar):
    label_index = get_google_output_idx(y)
    data = data[1:]  # google contains csv headers
    data = data[:, label_index]
    return data


__all__ = [
    "TDatasetName",
    "TOutputVar",
    "get_alibaba_output",
    "get_alibaba_output_idx",
    "get_google_output",
    "get_google_output_idx",
]
