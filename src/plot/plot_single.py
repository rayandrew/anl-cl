import json
import os
from pathlib import Path

import torch

import hydra
import numpy as np
from omegaconf import DictConfig

from src.dataset import (
    AlibabaMachineDataset,
    AlibabaMachineSequenceDataset,
    alibaba_machine_sequence_collate,
)
from src.utils.general import get_model_fname
from src.utils.plot import (
    EvalResult,
    plot_auc_roc,
    plot_avg_acc,
    plot_avg_forgetting,
    plot_diff,
    plot_end_acc,
    plot_prediction,
)
from src.utils.summary import generate_summary


def predict(
    cfg: DictConfig,
):
    data_path = Path(cfg.filename)

    device = torch.device(
        f"cuda:{cfg.cuda}"
        if torch.cuda.is_available() and cfg.cuda > 0
        else "cpu"
    )

    dataset, raw_dataset  = (
        AlibabaMachineSequenceDataset(
            filename=data_path,
            n_labels=cfg.n_labels,
            subset="all",
            y=cfg.y,
            seq_len=cfg.seq_len,
        )
        if cfg.sequential
        else AlibabaMachineDataset(
            filename=data_path,
            n_labels=cfg.n_labels,
            subset="all",
            y=cfg.y,
        )
    )
    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        collate_fn=alibaba_machine_sequence_collate
        if cfg.sequential
        else None,
    )

    output_folder = Path(cfg.out_path)

    model_path = output_folder / get_model_fname(cfg)
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

    for i in range(-cfg.n_labels, cfg.n_labels + 1, 1):
        res["diffs_dict"][i] = 0

    print(
        "Predicting using sequential model..."
        if cfg.sequential
        else "Predicting using non-sequential model..."
    )
    for i, (x, y, _dist) in enumerate(data):
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
        **res,
        name="Sequential" if cfg.sequential else "Non-Sequential",
    )

    train_res_path = model_path.parent / f"train_results.json"

    if train_res_path.exists():
        train_results = generate_summary(train_res_path)
        res.train_results = train_results
        # train_results = compute_perf(train_res)
        # res.train_results = TrainResult(**train_results)

    return res, raw_dataset.n_experiences()


@hydra.main(
    config_path="../../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    data_path = Path(cfg.filename)
    changepoints_path = (
        data_path.parent / f"{data_path.stem}_change.csv"
    )
    changepoints = np.array([])
    if changepoints_path.exists():
        # changepoints file contains of two columns: # rows, timestamp
        changepoints = np.loadtxt(changepoints_path, delimiter=",")
        changepoints = changepoints.astype(int)[:, 0]  # pick # rows

    output_folder = Path(cfg.out_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    print("Predicting data...")
    res, n_exp = predict(cfg)

    plot_title_prefix = (
        "[Sequential]" if cfg.sequential else "[Non-Sequential]"
    )

    if cfg.univariate and cfg.sequential:
        plot_title_prefix += "[Univariate]"

    plot_title = f"{plot_title_prefix} Sequential vs Non-Sequential (y={cfg.y}, strategy={cfg.strategy})"
    results = [res]
    plot_prediction(
        results,
        output_folder,
        changepoints=changepoints,
        config=cfg,
        title=plot_title,
    )
    plot_auc_roc(results, output_folder, config=cfg, title=plot_title)
    plot_diff(results, output_folder, config=cfg, title=plot_title)
    plot_avg_acc(
        results,
        output_folder,
        n_exp=n_exp,
        config=cfg,
        title=plot_title,
    )
    plot_avg_forgetting(
        results,
        output_folder,
        n_exp=n_exp,
        config=cfg,
        title=plot_title,
    )
    # TODO: fix this! needs to get `raw_acc` somehow
    # plot_end_acc(
    #     results,
    #     output_folder,
    #     n_exp=n_exp,
    #     config=cfg,
    #     title=plot_title,
    # )
    # plot_end_forgetting(
    #     results,
    #     output_folder,
    #     n_exp=n_exp,
    #     config=cfg,
    #     title=plot_title,
    # )


if __name__ == "__main__":
    main()
