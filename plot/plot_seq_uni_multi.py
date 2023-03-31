import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch

import numpy as np

from dataset import AlibabaMachineDataset
from utils.parse_v2 import compute_perf
from utils.plot import (
    EvalResult,
    TrainResult,
    plot_auc_roc,
    plot_avg_acc,
    plot_avg_forgetting,
    plot_diff,
    plot_end_acc,
    plot_end_forgetting,
    plot_prediction,
)


def predict(
    model_path: Path,
    data_path: Path,
    device: torch.device,
    args: Any,
    univariate: bool = False,
):

    raw_data = AlibabaMachineDataset(
        filename=data_path,
        n_labels=args.n_labels,
        mode="predict",
        y=args.y,
        seq=True,
        seq_len=args.seq_len,
        univariate=univariate,
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

    print(
        "Predicting using univariate model..."
        if univariate
        else "Predicting using multivariate model..."
    )
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

    res = EvalResult(
        **res, name="Univariate" if univariate else "Multivariate"
    )

    train_res_path = model_path.parent / f"train_results.json"

    if train_res_path.exists():
        with open(train_res_path, "r") as f:
            train_res = json.load(f)
            train_results = compute_perf(train_res)
            res.train_results = TrainResult(**train_results)

    return res, raw_data.n_experiences()


def main(args):
    data_path = Path(args.data)
    changepoints_path = (
        data_path.parent / f"{data_path.stem}_change.csv"
    )
    univariate_path = Path(args.univariate)
    multivariate_path = Path(args.multivariate)

    changepoints = np.array([])
    if changepoints_path.exists():
        # changepoints file contains of two columns: # rows, timestamp
        changepoints = np.loadtxt(changepoints_path, delimiter=",")
        changepoints = changepoints.astype(int)[:, 0]  # pick # rows

    output_folder = (
        Path(args.output_folder) / data_path.stem / args.strategy
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("Using univariate sequential data...")
    uni_eval_res, n_exp = predict(
        univariate_path, data_path, device, args, univariate=True
    )

    print("Using multivariate sequential data...")
    multi_eval_res, _ = predict(
        multivariate_path, data_path, device, args, univariate=False
    )

    plot_title_prefix = "[Local]" if args.local else "[Global]"

    plot_title = f"{plot_title_prefix} Univariate vs Multivariate (y={args.y}, strategy={args.strategy})"
    results = [uni_eval_res, multi_eval_res]
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
        n_exp=n_exp,
        args=args,
        title=plot_title,
    )
    plot_avg_forgetting(
        results,
        output_folder,
        n_exp=n_exp,
        args=args,
        title=plot_title,
    )
    plot_end_acc(
        results,
        output_folder,
        n_exp=n_exp,
        args=args,
        title=plot_title,
    )
    # plot_end_forgetting(
    #     results,
    #     output_folder,
    #     n_exp=n_exp,
    #     args=args,
    #     title=plot_title,
    # )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="Data file")
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument(
        "-o", "--output_folder", type=str, default="out"
    )
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
    parser.add_argument(
        "--univariate", help="Path to univariate model", type=str
    )
    parser.add_argument(
        "--multivariate", help="Path to multivariate model", type=str
    )
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--local", action="store_true")

    args = parser.parse_args()
    main(args)
