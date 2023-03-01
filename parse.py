from argparse import ArgumentParser
import re
import json
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from texttable import Texttable

METRIC_REGEX = re.compile(r"(\S+)/(\S+)/(\S+)/(\S+)")


def preprocess_summary(filename: str | Path):
    res = {}
    results_dict = {}
    with open(filename, "r") as f:
        results_dict = json.load(f)
        for exp in results_dict:
            current_exp = int(exp) + 1
            if current_exp not in res:
                res[current_exp] = {}

            for metric in results_dict[exp]:
                res_metrix_re = METRIC_REGEX.search(metric)

                if res_metrix_re and len(res_metrix_re.groups()) == 4:
                    g = res_metrix_re.groups()
                    val = float(results_dict[exp][metric])
                    if g[0] == "Top1_Acc_Epoch" and g[1] == "train_phase":
                        res[current_exp]["train_acc"] = val

                    if g[0] == "Top1_Acc_Exp/eval_phase" and g[1] == "test_stream":
                        if "test_acc" not in res[current_exp]:
                            res[current_exp]["test_acc"] = {}
                        res[current_exp]["test_acc"][int(g[3][-3:]) + 1] = val

                    if g[0] == "Top1_Acc_Stream" and g[1] == "eval_phase":
                        res[current_exp]["avg_test_acc_avl"] = val

    return results_dict, res


def process_single_file(filename: str | Path):
    res = {}
    raw, summary = preprocess_summary(filename)

    avg_accs = []
    avg_forgettings = []

    for exp in summary:
        avg_acc = 0.0
        n = 0
        for test_acc_exp, test_acc_val in summary[exp]["test_acc"].items():
            if test_acc_exp <= exp:
                avg_acc += test_acc_val
                n += 1

        avg_acc = avg_acc / n
        summary[exp]["avg_acc"] = avg_acc

        #### CALCULATING AVG_FORGETTING
        forgetting = 0.0

        if exp > 1:
            # log_summary[j]["test_acc"][i] current test acc on task i at exp j
            # max_test = {}
            i = exp
            # print(f"F_{i}")
            f_i_j_s = []

            for j in range(1, i):  # iterate j to exp-1 to get F_i
                max_val = -99
                # print(f"F_{i}", j)
                for l in range(1, i):  # iterate l from 0 to i-1
                    # print(l, j, summary[l]["test_acc"][j])
                    max_val = max(max_val, summary[l]["test_acc"][j])

                    # print(f"F_{i}", f"j={j}", f"l={l}", max_val, summary[i]["test_acc"][j])

                # print(i, j, max_val, summary[i]["test_acc"][j])
                F_i_j = max_val - summary[i]["test_acc"][j]
                # print(f"F_{i}_{j}")
                f_i_j_s.append(F_i_j)

            summary[exp]["avg_forgetting"] = sum(f_i_j_s) / len(f_i_j_s)
        else:
            summary[exp]["avg_forgetting"] = 0.0

        avg_accs.append(summary[exp]["avg_acc"])
        avg_forgettings.append(summary[exp]["avg_forgetting"])

    res = {
        "raw": raw,
        "log": summary,
        "avg_accs": avg_accs,
        "avg_forgettings": avg_forgettings,
    }

    return res


def plot(machine_id: str, summary: dict, n_exp: int, output_path: str | Path):
    plt.style.use("seaborn-talk")

    marker = itertools.cycle(("D", "P", "s", "v", "o", "*", "X"))
    x = list(range(1, n_exp + 1))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for strategy in summary:
        ax.plot(
            x,
            summary[strategy]["avg_accs"],
            label=strategy,
            linestyle="-",
            marker=next(marker),
        )

    # ax.set_title("Average Accuracy", pad=20)
    ax.set_xticks(x)
    ax.set_xlim([0.7, float(n_exp) + 0.3])
    ax.set_xlabel("Distribution Shift Window (Task)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Machine {machine_id} Average Accuracy")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, symbol="", decimals=0))
    # ax.legend(frameon=True)
    handles, labels = ax.get_legend_handles_labels()
    # order = [2, 0, 3, 5, 4, 6, 1]
    ax.legend(
        # [handles[idx] for idx in order], [labels[idx] for idx in order], frameon=True
        handles,
        labels,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path / "avg_acc.png", dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for strategy in summary:
        ax.plot(
            x,
            summary[strategy]["avg_forgettings"],
            label=strategy,
            linestyle="-",
            marker=next(marker),
        )

    # ax.set_title("Average Forgetting", pad=20)
    ax.set_xticks(x)
    ax.set_xlabel("Distribution Shift Window (Task)")
    ax.set_ylabel("Forgetting (%)")
    ax.set_title(f"Machine {machine_id} Average Forgetting")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, symbol="", decimals=0))
    ax.set_xlim([0.7, float(n_exp) + 0.3])
    handles, labels = ax.get_legend_handles_labels()
    # order = [2, 0, 5, 1, 4, 6, 3]
    ax.legend(
        # [handles[idx] for idx in order], [labels[idx] for idx in order], frameon=True
        handles,
        labels,
        frameon=True,
    )
    # ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path / "avg_forgetting.png", dpi=300)


def generate_table(machine_id: str, summary: dict, output_path: str | Path):
    table_1 = Texttable()
    table_1.set_cols_align(["l", "c", "c", "c", "c"])
    table_1.set_cols_valign(["t", "m", "m", "b", "b"])
    for strategy in summary:
        for exp in summary[strategy]["log"]:
            table_1.add_rows(
                [
                    ["Method", "Task #", "Train Acc", "Avg Test Acc", "Avg Forgetting"],
                    [
                        strategy,
                        exp,
                        summary[strategy]["log"][exp]["train_acc"],
                        summary[strategy]["avg_accs"][exp - 1],
                        summary[strategy]["avg_forgettings"][exp - 1],
                    ],
                ]
            )

    print(table_1.draw())
    with open(output_path / f"table_res_{machine_id}.txt", "w") as f:
        f.write(table_1.draw())


def main(args):
    path = Path(args.path)
    if path.is_dir():
        res = {}
        for file in path.glob(f"**/train_results.json"):
            summary = process_single_file(file)
            with open(file.parent / "summary.json", "w") as f:
                json.dump(summary, f)

            strategy = file.parent.stem
            res[strategy] = summary

        with open(path / "summary.json", "w") as f:
            json.dump(res, f)

        machine_id = file.parent.parent.stem
        plot(machine_id, res, args.n_experiences, path)
        generate_table(machine_id, res, path)

        return res

    res = {}
    summary = process_single_file(args)
    machine_id = file.parent.stem
    res[machine_id] = summary
    with open(file.parent / "summary.json", "w") as f:
        json.dump(summary, f)
    plot(res, args.n_experiences, file.parent)
    generate_table(machine_id, res, path)
    return res


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("-x", "--n_experiences", type=int, required=True)
    args = parser.parse_args()
    main(args)
