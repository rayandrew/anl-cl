import argparse
import time
from pathlib import Path
from typing import Sequence
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.hddm_w import HDDM_W
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection import KSWIN


class DriftDetection:
    def __init__(self, data_stream, verbose=True):
        self.methods = []
        self.weights = []
        self.drifts = []
        self.data_stream = data_stream
        self.verbose = verbose

    def add_method(self, method, weight=1):
        self.methods.append(method)
        self.weights.append(weight)
        self.drifts.append(deque())

    def _get_drift_point(self):
        for pos, ele in enumerate(self.data_stream):
            for method_index, method in enumerate(self.methods):
                method.add_element(ele)
                if method.detected_change():
                    self.drifts[method_index].append(pos)

    def _vote_drift(self, window_size, threshold):
        self.vote_drifts = []
        for i in range(0, len(self.data_stream), window_size):
            pos_sum = 0
            weight_sum = 0
            for method_index, method in enumerate(self.methods):
                while len(self.drifts[method_index]) != 0:
                    pos = self.drifts[method_index][0]
                    if pos >= (i + 1) * window_size:
                        break
                    else:
                        pos_sum += self.weights[method_index] * pos
                        weight_sum += self.weights[method_index]
                        self.drifts[method_index].popleft()
            if weight_sum != 0 and self.verbose:
                print(weight_sum)
            if weight_sum > threshold:
                mean_pos = int(pos_sum / weight_sum)
                self.vote_drifts.append(mean_pos)

    def get_voted_drift(self, window_size, threshold):
        for method_idx, item in enumerate(self.drifts):
            self.drifts[method_idx] = deque()
        self._get_drift_point()
        self._vote_drift(window_size, threshold)
        return self.vote_drifts


def plot(stream_window, change, path):
    y = stream_window
    x = [i for i in range(len(y))]

    plt.figure(figsize=(30, 6))

    # Plot the data
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("Stream Data Plot")

    plt.tight_layout()

    # Add vertical lines
    index = [i for i in change]
    for i in index:
        plt.axvline(x=i, color="red", linestyle="--")

    # Show the plot
    plt.savefig(f"{path}")


def add_dist_label(data, dist: Sequence[int]):
    distributions = np.zeros((len(data[:, 0]), 1))

    print(dist)
    print(f"Dist from 0 to {dist[0]}: 0")
    distributions[: dist[0]] = 0
    for i in range(len(dist) - 2):
        print(f"Dist from {dist[i]} to {dist[i+1]}: {i+1}")
        distributions[dist[i] : dist[i + 1]] = i + 1
    print(f"Dist from {dist[i]} to {dist[-1]}: {len(dist) - 1}")
    distributions[dist[-1] :] = len(dist) - 1

    print(np.unique(distributions, return_counts=True))

    data = np.append(data, distributions, axis=1)
    return data


def main(args):
    input_path = Path(args.input)
    orig_data = np.genfromtxt(input_path, delimiter=",")

    output_path = (
        Path(args.output).joinpath("local" if args.local else "global")
        / input_path.stem
        / f"{input_path.stem}_{args.window_size}-{args.threshold}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    data = orig_data

    if args.local:
        label_index = 9
        if args.y == "cpu":
            label_index = 7
        elif args.y == "mem":
            label_index = 8
        orig_data = orig_data[1:]
        data = data[1:]
    else:
        label_index = 8
        if args.y == "cpu":
            label_index = 2
        elif args.y == "mem":
            label_index = 3

    data = data[:, label_index]

    dd = DriftDetection(data, verbose=False)
    dd.add_method(ADWIN(), 1)
    dd.add_method(DDM(), 1)
    dd.add_method(EDDM(), 1)
    dd.add_method(HDDM_A(), 1)
    dd.add_method(HDDM_W(), 1)
    dd.add_method(PageHinkley(), 1)
    dd.add_method(KSWIN(), 1)
    window_size = args.window_size
    threshold = args.threshold
    change_list = dd.get_voted_drift(window_size=window_size, threshold=threshold)
    change_list = [i + 1000 for i in change_list]
    change_list = sorted(change_list)
    # rec_time = int(time.time())

    n_dist = len(change_list)
    if n_dist == 0:
        print("No change detected")
        return

    print(f"Number of distributions: {n_dist}")
    plot(
        data,
        change_list,
        output_path / f"plot_{input_path.stem}_{window_size}_{threshold}_{args.y}.png",
    )

    data = add_dist_label(orig_data, change_list)
    np.savetxt(
        output_path / f"{input_path.stem}_{args.y}.csv",
        data,
        fmt="%.4e",
        delimiter=",",
    )

    changes = np.zeros((len(change_list), 2))
    changes[:, 0] = change_list

    for i in range(len(change_list)):
        changes[i, 1] = orig_data[change_list[i] - 1000, 1]  # get timestamp

    np.savetxt(
        output_path / f"{input_path.stem}_{args.y}_change.csv",
        changes,
        fmt="%d",
        delimiter=",",
    )

    # print(data[0:10, :])
    # print(data[-10:, :])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[Preprocess] Drift Detection")
    parser.add_argument(
        "input",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"preprocessed_data",
    )
    parser.add_argument(
        "-y",
        type=str,
        choices=["cpu", "mem", "disk"],
        default="cpu",
        help="choose the y axis",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=300,
        help="threshold for voting",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=75,
        help="window size for voting",
    )
    parser.add_argument(
        "--local",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
