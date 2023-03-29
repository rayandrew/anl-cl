from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

import torch

import numpy as np

from dataset import AlibabaMachineDataset
from plot_utils import (
    EvalResult,
    TrainResult,
    plot_diff,
    plot_prediction,
    plot_auc_roc,
    plot_avg_acc,
    plot_avg_forgetting,
)
from parse_v2 import compute_perf


def predict_seq(model_path: Path, data_path: Path, device: torch.device, args: Any):
    raw_data = AlibabaMachineDataset(
        filename=data_path,
        n_labels=args.n_labels,
        mode="predict",
        y=args.y,
        seq=True,
        seq_len=args.seq_len,
    )
    data = torch.utils.data.DataLoader(
        raw_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
    )

    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()

    res = {
        "diffs_dict": {},
        "diffs": [],
        "y_origs": [],
        "y_preds": [],
        "predict_proba": [],
        "train_results": None,
    }

    for i in range(-args.n_labels, args.n_labels + 1, 1):
        res["diffs_dict"][i] = 0

    print("Predicting using non-sequential model...")
    for i, (x, _dist, y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        pred_label = torch.argmax(y_pred, dim=1)
        diffs = y - pred_label
        for diff in diffs:
            res["diffs_dict"][diff.item()] += 1
        res["diffs"] += diffs.tolist()
        res["y_origs"] += y.tolist()
        res["y_preds"] += pred_label.tolist()
        res["predict_proba"] += y_pred.tolist()

    res["predict_proba"] = np.array(res["predict_proba"])

    res = EvalResult(**res, name="Non-Sequential")

    train_res_path = model_path.parent / f"train_results.json"

    if train_res_path.exists():
        with open(train_res_path, "r") as f:
            train_res = json.load(f)
            train_results = compute_perf(train_res)
            res.train_results = TrainResult(**train_results)

    return res


def main(args):
    data_path = Path(args.data)
    changepoints_path = data_path.parent / f"{data_path.stem}_change.csv"
    seq_path = Path(args.seq)
    non_seq_path = Path(args.non_seq)

    changepoints = np.array([])
    if changepoints_path.exists():
        # changepoints file contains of two columns: # rows, timestamp
        changepoints = np.loadtxt(changepoints_path, delimiter=",")
        changepoints = changepoints.astype(int)[:, 0]  # pick # rows

    output_folder = Path(args.output_folder) / data_path.stem / args.strategy
    output_folder.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    raw_seq_data = AlibabaMachineDataset(
        filename=data_path,
        n_labels=args.n_labels,
        mode="predict",
        y=args.y,
        seq=True,
        seq_len=args.seq_len,
    )
    seq_data = torch.utils.data.DataLoader(
        raw_seq_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
    )
    raw_non_seq_data = AlibabaMachineDataset(
        filename=data_path,
        n_labels=args.n_labels,
        mode="predict",
        y=args.y,
        seq=False,
        seq_len=args.seq_len,
    )
    non_seq_data = torch.utils.data.DataLoader(
        raw_non_seq_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
    )

    seq_model = torch.jit.load(seq_path)
    seq_model.to(device)
    seq_model.eval()

    non_seq_model = torch.jit.load(non_seq_path)
    non_seq_model.to(device)
    non_seq_model.eval()

    non_seq_res = {
        "diffs_dict": {},
        "diffs": [],
        "y_origs": [],
        "y_preds": [],
        "predict_proba": [],
        "train_results": None,
    }

    seq_res = {
        "diffs_dict": {},
        "diffs": [],
        "y_origs": [],
        "y_preds": [],
        "predict_proba": [],
        "train_results": None,
    }

    for i in range(-args.n_labels, args.n_labels + 1, 1):
        non_seq_res["diffs_dict"][i] = 0
        seq_res["diffs_dict"][i] = 0

    print("Predicting using non-sequential model...")
    for i, (x, _dist, y) in enumerate(non_seq_data):
        x = x.to(device)
        y = y.to(device)
        y_pred = non_seq_model(x)
        pred_label = torch.argmax(y_pred, dim=1)
        diffs = y - pred_label
        for diff in diffs:
            non_seq_res["diffs_dict"][diff.item()] += 1
        non_seq_res["diffs"] += diffs.tolist()
        non_seq_res["y_origs"] += y.tolist()
        non_seq_res["y_preds"] += pred_label.tolist()
        non_seq_res["predict_proba"] += y_pred.tolist()

    print("Predicting using sequential model...")
    for i, (x, _dist, y) in enumerate(seq_data):
        x = x.to(device)
        y = y.to(device)
        y_pred = seq_model(x)
        pred_label = torch.argmax(y_pred, dim=1)
        diffs = y - pred_label
        for diff in diffs:
            seq_res["diffs_dict"][diff.item()] += 1
        seq_res["diffs"] += diffs.tolist()
        seq_res["y_origs"] += y.tolist()
        seq_res["y_preds"] += pred_label.tolist()
        seq_res["predict_proba"] += y_pred.tolist()
        # break

    non_seq_res["predict_proba"] = np.array(non_seq_res["predict_proba"])
    seq_res["predict_proba"] = np.array(seq_res["predict_proba"])

    non_seq_eval_res = EvalResult(**non_seq_res, name="Non-Sequential")
    seq_eval_res = EvalResult(**seq_res, name="Sequential")

    non_seq_train_res_path = non_seq_path.parent / f"train_results.json"

    if non_seq_train_res_path.exists():
        with open(non_seq_train_res_path, "r") as f:
            non_seq_train_res = json.load(f)
            non_seq_train_results = compute_perf(non_seq_train_res)
            non_seq_eval_res.train_results = TrainResult(**non_seq_train_results)

    seq_train_res_path = seq_path.parent / f"train_results.json"
    if seq_train_res_path.exists():
        with open(seq_train_res_path, "r") as f:
            seq_train_res = json.load(f)
            seq_train_results = compute_perf(seq_train_res)
            seq_eval_res.train_results = TrainResult(**seq_train_results)

    plot_title = f"Sequential vs Non-Sequential (y={args.y}, strategy={args.strategy})"
    results = [non_seq_eval_res, seq_eval_res]
    plot_prediction(
        results,
        output_folder,
        changepoints=changepoints,
        args=args,
        title=plot_title,
    )
    plot_auc_roc(results, output_folder, args=args, title=plot_title)
    plot_diff(results, output_folder, args=args, title=plot_title)
    plot_avg_acc(
        results,
        output_folder,
        n_exp=raw_non_seq_data.n_experiences(),
        args=args,
        title=plot_title,
    )
    plot_avg_forgetting(
        results,
        output_folder,
        n_exp=raw_non_seq_data.n_experiences(),
        args=args,
        title=plot_title,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="Data file")
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("-o", "--output_folder", type=str, default="out")
    parser.add_argument("-nl", "--n_labels", type=int, default=10)
    parser.add_argument(
        "-y",
        type=str,
        choices=["cpu", "mem", "disk"],
        default="cpu",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        help="Strategy to use",
        type=str,
        choices=["gss", "agem", "naive", "lwf", "ewc", "gdumb"],
        default="naive",
    )
    parser.add_argument("--seq", help="Path to sequential model", type=str)
    parser.add_argument("--non_seq", help="Path to non-sequential model", type=str)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--type",
        type=str,
        choices=[
            "seq-non-seq",
            "local-global",
        ],
        default="test",
    )

    args = parser.parse_args()
    main(args)
