import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy.stats as stats
from scipy.stats import sem


# https://github.com/RaptorMai/online-continual-learning/blob/6175ca034e56435acd82b8f17ff59f920f0bc45e/experiment/metrics.py#L5
def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.
    :param end_task_acc_arr:        (list) List of lists
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf(
        (1 + 0.95) / 2, n_run - 1
    )  # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]  # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(
        end_acc, axis=1
    )  # mean of end task accuracies per run
    avg_end_acc = (
        np.mean(avg_acc_per_run),
        t_coef * sem(avg_acc_per_run),
    )

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean(
        (
            np.sum(np.tril(end_task_acc_arr), axis=2)
            / (np.arange(n_tasks) + 1)
        ),
        axis=1,
    )
    avg_acc = (np.mean(acc_per_run), t_coef * sem(acc_per_run))

    # compute BWT+
    bwt_per_run = (
        np.sum(np.tril(end_task_acc_arr, -1), axis=(1, 2))
        - np.sum(
            np.diagonal(end_task_acc_arr, axis1=1, axis2=2)
            * (np.arange(n_tasks, 0, -1) - 1),
            axis=1,
        )
    ) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run))

    # compute FWT
    fwt_per_run = np.sum(
        np.triu(end_task_acc_arr, 1), axis=(1, 2)
    ) / (n_tasks * (n_tasks - 1) / 2)
    avg_fwt = (np.mean(fwt_per_run), t_coef * sem(fwt_per_run))
    return avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt


# def main(args):
#     train_results_path = Path(args.path)
#     train_results_file = open(train_results_path, "r")
#     results = json.load(train_results_file)
#     train_results_file.close()

#     avg_acc = []

#     n_tasks = len(results)
#     for t, data in results.items():
#         acc_array = np.zeros(n_tasks)

#         for task in range(n_tasks):
#             acc_array[task] = data[
#                 f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{task:03d}"
#             ]

#         print(acc_array)
#         avg_acc.append(acc_array)

#     avg_acc = np.array([avg_acc, avg_acc])

#     avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(avg_acc)
#     print(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt)


def compute_perf(results: dict):
    avg_acc = []
    raw_acc = []

    for t, data in results.items():
        accuracy_sum = 0
        raw_acc_temp = []
        for i in range(0, int(t) + 1):
            accuracy_sum += data[
                f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}"
            ]
            raw_acc_temp.append(
                data[
                    f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}"
                ]
            )
        avg_acc.append(accuracy_sum / (int(t) + 1))
        raw_acc.append(raw_acc_temp)

    # initialize a dictionary to store the forgetting values for each task
    forgetting = {}
    for task in results.keys():
        if task == "0":
            forgetting[task] = 0
            continue
        # iterate over all previous tasks
        for j in range(0, int(task) + 1):
            # print(task, j)
            # find the best test accuracy achieved on task j before learning task k
            best_acc_j = max(
                [
                    results[str(l)][
                        f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{j:03d}"
                    ]
                    for l in range(0, int(task))
                ]
            )
            # find the test accuracy achieved on task j after learning task k
            acc_k_j = results[task][
                f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{j:03d}"
            ]
            # print(best_acc_j, acc_k_j)
            # calculate the forgetting value for task i and task j
            # forgetting_ij = max(0, best_acc_j - acc_k_j)
            forgetting_ij = best_acc_j - acc_k_j
            # add the forgetting value to the running total for task i
            if task not in forgetting:
                forgetting[task] = 0
            forgetting[task] += forgetting_ij
        # calculate the average forgetting for task i
        if int(task) >= 1:
            forgetting[task] = (1 / (int(task))) * forgetting[task]

    overall_avg_acc = sum(avg_acc) / len(avg_acc)
    overall_avg_forgetting = sum(forgetting.values()) / len(
        forgetting
    )

    # Save the results
    parsed_results = {
        "avg_acc": avg_acc,
        "avg_forgetting": list(forgetting.values()),
        "ovr_avg_acc": overall_avg_acc,
        "ovr_avg_forgetting": overall_avg_forgetting,
        "raw_acc": raw_acc,
    }

    return parsed_results


def main(args):
    train_results_path = Path(args.path)
    train_results_file = open(train_results_path, "r")
    results = json.load(train_results_file)
    train_results_file.close()

    parsed_results = compute_perf(results)
    print(parsed_results)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    main(args)


__all__ = ["compute_performance", "compute_perf"]
