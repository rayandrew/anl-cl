from argparse import ArgumentParser
import re
import json
from pathlib import Path

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


def main(args):
    path = Path(args.path)
    if path.is_dir():
        res = {}
        for file in path.glob(f"**/train_results.json"):
            summary = process_single_file(file)
            with open(file.parent / "summary.json", "w") as f:
                json.dump(summary, f)

            # machine_id = file.parent.parent.stem
            machine_id = file.parent.stem
            res[machine_id] = summary

        with open(path / "summary.json", "w") as f:
            json.dump(res, f)

        return res

    res = process_single_file(args)
    with open(file.parent / "summary.json", "w") as f:
        json.dump(summary, f)
    return res


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    main(args)
