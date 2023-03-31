from argparse import ArgumentParser
import json
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import process_file, generate_table


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
    fig.savefig(output_path / "avg_acc.png", dpi=100)

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
    fig.savefig(output_path / "avg_forgetting.png", dpi=100)


def print_table(machine_id: str, summary: dict, output_path: str | Path):
    table = generate_table(data=summary)

    print(table.draw())
    with open(output_path / f"table_res_{machine_id}.txt", "w") as f:
        f.write(table.draw())


def main(args):
    path = Path(args.path)
    if path.is_dir():
        res = {}
        for file in path.glob(f"**/train_results.json"):
            summary = process_file(file)
            with open(file.parent / "summary.json", "w") as f:
                json.dump(summary, f)

            strategy = file.parent.stem
            res[strategy] = summary

        with open(path / "summary.json", "w") as f:
            json.dump(res, f)

        machine_id = file.parent.parent.stem
        # plot(machine_id, res, args.n_experiences, path)
        print_table(machine_id, res, path)

        return res

    res = {}
    file = Path(args.path)
    summary = process_file(file)
    machine_id = file.parent.stem
    res[machine_id] = summary
    with open(file.parent / "summary.json", "w") as f:
        json.dump(summary, f)
    # plot(machine_id, res, args.n_experiences, file.parent)
    print_table(machine_id, res, file.parent)
    return res


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("-x", "--n_experiences", type=int, required=True)
    args = parser.parse_args()
    main(args)
