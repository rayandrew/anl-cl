from pathlib import Path
import itertools

from helper import DATASETS, EXTENSIONS, SCENARIOS, STRATEGIES, TRAININGS, TASKS, get_all_dataset_files, get_all_dataset_files_as_dict

wildcard_constraints:
    dataset = "|".join(DATASETS),
    # path = "|".join(get_all_dataset_files("raw_data", DATASETS, return_stem=True)),
    ext = "|".join(EXTENSIONS),
    scenario = "|".join(SCENARIOS),
    strategy = "|".join(STRATEGIES),
    training = "|".join(TRAININGS),
    task = "|".join(TASKS),

# for (ext, scenario, strategy, training, task) in itertools.product(EXTENSIONS, SCENARIOS, STRATEGIES, TRAININGS, TASKS):
rule:
    # name: f"train_{task}_{training}_{scenario}_{strategy}"
    name: f"training"
    input:
        "raw_data/{dataset}/{path}.parquet",
    output:
        directory("out/training/{dataset}/{path}/{task}/{training}/{scenario}/{strategy}"),
    params:
        dataset="{dataset}",
        filepath="{path}",
        scenario="{scenario}",
        task="{task}",
        training="{training}",
    log:
        "logs/training/{dataset}/{path}/{task}/{training}/{scenario}/{strategy}.log",
    script: "../scripts/pipeline/{wildcards.scenario}.py"

def get_pipeline_output():
    final_output = []
    # dicts = get_all_dataset_files_as_dict("raw_data", return_stem=True) 
    # for dataset in dicts:
    #     for filepath in dicts[dataset]:
    #         final_output += expand("out/training/{dataset}/{path}/{task}/{training}/{scenario}/{strategy}", dataset=dataset, path=filepath, task=TASKS, scenario=SCENARIOS, strategy=STRATEGIES, training=TRAININGS)

    # print(final_output)
    return final_output
