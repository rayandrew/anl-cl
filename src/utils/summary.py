import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

from texttable import Texttable

from src.utils.ds import DataClassJSONEncoder


@dataclass
class TaskSummary:
    task_id: int
    acc: Sequence[float] = field(default_factory=list)
    forgetting: Sequence[float] = field(default_factory=list)
    bwt: Sequence[float] = field(default_factory=list)
    # f1: Sequence[float] = field(default_factory=list)
    # precision: Sequence[float] = field(default_factory=list)
    # recall: Sequence[float] = field(default_factory=list)
    # auroc: Sequence[float] = field(default_factory=list)


@dataclass
class TrainingSummary:
    n_tasks: int
    ovr_avg_acc: float = 0.0
    ovr_avg_forgetting: float = 0.0
    ovr_avg_bwt: float = 0.0
    avg_acc: Sequence[float] = field(default_factory=list)
    avg_forgetting: Sequence[float] = field(default_factory=list)
    avg_bwt: Sequence[float] = field(default_factory=list)
    task_data: Dict[int, TaskSummary] = field(default_factory=dict)
    avg_f1: Sequence[float] = field(default_factory=list)
    avg_precision: Sequence[float] = field(default_factory=list)
    avg_recall: Sequence[float] = field(default_factory=list)
    avg_auroc: Sequence[float] = field(default_factory=list)


def generate_summary_dict(
    path: str | Path,
) -> Dict[int, Dict[str, Sequence[float]]]:
    """Generate a summary dict from a path to a directory containing the results of a single experiment.

    Args:
        path (str | Path): Path to the file containing the results of a single experiment.

    Returns:
        Dict[int, Dict[str, Sequence[float]]]: A dictionary containing the results of the experiment.
    """

    path = Path(path)
    res = {
        "n_tasks": 0,
        "ovr_avg_acc": 0.0,
        "ovr_avg_forgetting": 0.0,
        "ovr_avg_bwt": 0.0,
        "avg_acc": [],
        "avg_forgetting": [],
        "avg_bwt": [],
        "task_data": {},
        "avg_f1": [],
        "avg_precision": [],
        "avg_recall": [],
        "avg_auroc": [],
    }

    if not path.exists():
        raise ValueError(f"Path {path} does not exist.")

    file = open(path, "r")
    data = json.load(file)
    file.close()

    res["n_tasks"] = len(data)

    for exp_str, value in data.items():
        # exp = int(exp_str)
        for key, val in value.items():
            if (
                "Top1_Acc_Stream_Tol/eval_phase/test_stream" in key
                or "Top1_Acc_Stream/eval_phase/test_stream" in key
            ):
                res["avg_acc"].append(val)
            elif (
                "StreamForgetting_Tol/eval_phase/test_stream" in key
                or "StreamForgetting/eval_phase/test_stream" in key
            ):
                res["avg_forgetting"].append(val)
            elif (
                "StreamBWT_Tol/eval_phase/test_stream" in key
                or "StreamBWT/eval_phase/test_stream" in key
            ):
                res["avg_bwt"].append(val)
            elif "F1/eval_phase/test_stream" in key:
                res["avg_f1"].append(val)
            elif "Precision_/eval_phase/test_stream" in key:
                res["avg_precision"].append(val)
            elif "Recall/eval_phase/test_stream" in key:
                res["avg_recall"].append(val)
            elif "AUROC/eval_phase/test_stream" in key:
                res["avg_auroc"].append(val)
            elif (
                "Top1_Acc_Exp_Tol/eval_phase/test_stream/Task000"
                in key
            ) or (
                "Top1_Acc_Exp/eval_phase/test_stream/Task000" in key
            ):
                task_id = key.replace(
                    "Top1_Acc_Exp_Tol/eval_phase/test_stream/Task000/Exp",
                    "",
                )
                task_id = task_id.replace(
                    "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp",
                    "",
                )
                task_id = int(task_id)
                if task_id not in res["task_data"]:
                    res["task_data"][task_id] = {}
                    res["task_data"][task_id]["task_id"] = task_id
                if "acc" not in res["task_data"][task_id]:
                    res["task_data"][task_id]["acc"] = []
                res["task_data"][task_id]["acc"].append(val)
            elif (
                "ExperienceForgetting_Tol/eval_phase/test_stream/Task000/"
                in key
            ) or (
                "ExperienceForgetting/eval_phase/test_stream/Task000/"
                in key
            ):
                task_id = key.replace(
                    "ExperienceForgetting_Tol/eval_phase/test_stream/Task000/Exp",
                    "",
                )
                task_id = task_id.replace(
                    "ExperienceForgetting/eval_phase/test_stream/Task000/Exp",
                    "",
                )
                task_id = int(task_id)
                if task_id not in res["task_data"]:
                    res["task_data"][task_id] = {}
                    res["task_data"][task_id]["task_id"] = task_id
                if "forgetting" not in res["task_data"][task_id]:
                    res["task_data"][task_id]["forgetting"] = []
                res["task_data"][task_id]["forgetting"].append(val)
            elif (
                "ExperienceBWT_Tol/eval_phase/test_stream/Task000/"
                in key
            ) or (
                "ExperienceBWT/eval_phase/test_stream/Task000/" in key
            ):
                task_id = key.replace(
                    "ExperienceBWT_Tol/eval_phase/test_stream/Task000/Exp",
                    "",
                )
                task_id = task_id.replace(
                    "ExperienceBWT/eval_phase/test_stream/Task000/Exp",
                    "",
                )
                task_id = int(task_id)
                if task_id not in res["task_data"]:
                    res["task_data"][task_id] = {}
                    res["task_data"][task_id]["task_id"] = task_id
                if "bwt" not in res["task_data"][task_id]:
                    res["task_data"][task_id]["bwt"] = []
                res["task_data"][task_id]["bwt"].append(val)

    res["ovr_avg_acc"] = sum(res["avg_acc"]) / len(res["avg_acc"])
    res["ovr_avg_forgetting"] = sum(res["avg_forgetting"]) / len(
        res["avg_forgetting"]
    )
    res["ovr_avg_bwt"] = sum(res["avg_bwt"]) / len(res["avg_bwt"])

    return res


def generate_summary(path: str | Path) -> TrainingSummary:
    res = generate_summary_dict(path)
    task_data = {
        task_id: TaskSummary(**(res["task_data"][task_id]))
        for task_id in res["task_data"]
    }
    res["task_data"] = task_data
    return TrainingSummary(**res)


def generate_summary_table(res: TrainingSummary):
    if res.n_tasks == 0:
        raise ValueError("No tasks found in the results.")

    table = Texttable()
    table.set_cols_align(["c", "c", "c", "c", "c", "c", "c", "c"])
    table.set_cols_valign(["m", "m", "m", "m", "m", "m", "m", "m"])

    for task in range(res.n_tasks):
        print(task)
        table.add_rows(
            [
                [
                    "Task #",
                    "Avg Test Acc",
                    "Avg Forgetting",
                    "Avg BWT",
                    "Avg F1",
                    "Avg Precision",
                    "Avg Recall",
                    "Avg AUROC",
                ],
                [
                    task,
                    res.avg_acc[task],
                    res.avg_forgetting[task],
                    res.avg_bwt[task],
                    res.avg_f1[task],
                    res.avg_precision[task],
                    res.avg_recall[task],
                    res.avg_auroc[task],
                ],
            ]
        )

    return table


def generate_task_table(res: TaskSummary):
    table = Texttable()
    table.set_cols_align(["c", "c", "c"])
    table.set_cols_valign(["m", "m", "m"])

    task_id = res.task_id

    for task in range(len(res.acc)):
        # this is the information of task_id=${res.task_id} when it was trained on task # {task}
        if task > task_id:
            table.add_rows(
                [
                    [
                        "Acc",
                        "Forgetting",
                        "BWT",
                        # "F1",
                        # "Precision",
                        # "Recall",
                        # "AUROC",
                    ],
                    [
                        res.acc[task],
                        res.forgetting[task - task_id - 1],
                        res.bwt[task - task_id - 1],
                        # res.f1[task - task_id - 1],
                        # res.precision[task - task_id - 1],
                        # res.recall[task - task_id - 1],
                        # res.auroc[task - task_id - 1],
                    ],
                ]
            )
            continue

        table.add_rows(
            [
                [
                    "Acc",
                    "Forgetting",
                    "BWT",
                    # "F1",
                    # "Precision",
                    # "Recall",
                    # "AUROC",
                ],
                [
                    res.acc[task],
                    "N/A",
                    "N/A",
                    # "N/A",
                    # "N/A",
                    # "N/A",
                    # "N/A",
                ],
            ]
        )

    return table


def summary_to_json(res: TrainingSummary, **kwargs):
    return json.dumps(res, cls=DataClassJSONEncoder, **kwargs)


__all__ = [
    "generate_summary_dict",
    "generate_summary_table",
    "generate_task_table",
    "generate_summary",
    "summary_to_json",
    "TrainingSummary",
    "TaskSummary",
]

if __name__ == "__main__":
    summary = generate_summary(
        "./out/training/offline-classification-retrain-chunks-from-scratch/alibaba/m_881/train_results.json"
    )
    # print(summary)
    print(generate_summary_table(summary).draw())
    for task_id in summary.task_data:
        print(f"Task #{task_id}")
        print(generate_task_table(summary.task_data[task_id]).draw())
        print()
        break
