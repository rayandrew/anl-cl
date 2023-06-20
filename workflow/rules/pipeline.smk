from pathlib import Path
import itertools

from helper import DATASETS, EXTENSIONS, SCENARIOS, STRATEGIES, TRAININGS, TASKS, MODELS, get_all_dataset_files, get_all_dataset_files_as_dict

wildcard_constraints:
    dataset = "|".join(DATASETS),
    # path = "|".join(get_all_dataset_files("raw_data", DATASETS, return_stem=True)),
    ext = "|".join(EXTENSIONS),
    scenario = "|".join(SCENARIOS),
    strategy = "|".join(STRATEGIES),
    training = "|".join(TRAININGS),
    task = "|".join(TASKS),
    model = "|".join(MODELS)

rule:
    name: f"analysis_dataset"
    input:
        "raw_data/{dataset}/{path}.parquet",
    output:
        directory("out/analysis/{dataset}/{path}"),
    params:
        dataset="{dataset}",
        filepath="{path}",
    log:
        "logs/analysis/{dataset}/{path}.log",
    script: "../scripts/analysis/dataset.py"

# for (ext, scenario, strategy, training, task) in itertools.product(EXTENSIONS, SCENARIOS, STRATEGIES, TRAININGS, TASKS):
rule:
    # name: f"train_{task}_{training}_{scenario}_{strategy}"
    name: f"training"
    input:
        "raw_data/{dataset}/{path}.parquet",
    output:
        directory("out/training/{dataset}/{path}/{task}/{training}/{scenario}/{strategy}/{model}"),
    params:
        dataset="{dataset}",
        filepath="{path}",
        scenario="{scenario}",
        task="{task}",
        model="{model}",
        training="{training}",
    log:
        "logs/training/{dataset}/{path}/{task}/{training}/{scenario}/{strategy}/{model}.log",
    script: "../scripts/pipeline/{wildcards.scenario}.py"

rule:
    name: f"eval_scenario"
    input:
        "out/training/{dataset}/{filename}/{task}/{training}/{scenario}",
    output:
        directory("out/evaluation/scenario/{dataset}/{filename}/{task}/{training}/{scenario}"),
    log:
        "logs/evaluation/scenario/{dataset}/{filename}/{task}/{training}/{scenario}.log",
    script: "../scripts/evaluation/plot-bar.py"

rule:
    name: f"eval_model"
    input:
        "out/training/{dataset}/{filename}/{task}/{training}/{scenario}/{model}",
    output:
        directory("out/evaluation/model/{dataset}/{filename}/{task}/{training}/{scenario}/{model}"),
    log:
        "logs/evaluation/model/{dataset}/{filename}/{task}/{training}/{scenario}/{model}.log",
    script: "../scripts/evaluation/plot-bar.py"

def get_pipeline_output():
    final_output = []
    # dicts = get_all_dataset_files_as_dict("raw_data", return_stem=True) 
    # for dataset in dicts:
    #     for filepath in dicts[dataset]:
    #         final_output += expand("out/training/{dataset}/{path}/{task}/{training}/{scenario}/{strategy}", dataset=dataset, path=filepath, task=TASKS, scenario=SCENARIOS, strategy=STRATEGIES, training=TRAININGS)

    # print(final_output)
    return final_output
