from pathlib import Path
import itertools

from helper import DATASETS, EXTENSIONS, SCENARIOS, STRATEGIES, TRAININGS, TASKS, MODELS, get_all_dataset_files, get_all_dataset_files_as_dict

wildcard_constraints:
    dataset = "|".join(DATASETS),
    ext = "|".join(EXTENSIONS),
    scenario = "|".join(SCENARIOS),
    strategy = "|".join(STRATEGIES),
    training = "|".join(TRAININGS),
    task = "|".join(TASKS),
    model = "|".join(MODELS)

rule:
    name: f"analysis_dataset"
    input:
        "raw_data/{dataset}/{filename}.parquet",
    output:
        directory("out/analysis/{dataset}/{filename}"),
    params:
        dataset="{dataset}",
        filename="{filename}",
    log:
        "logs/analysis/{dataset}/{filename}.log",
    script: "../scripts/analysis/dataset.py"

rule:
    name: f"training"
    input:
        "raw_data/{dataset}/{filename}.parquet",
    output:
        directory("out/training/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{strategy}/{feats}"),
    params:
        dataset="{dataset}",
        filename="{filename}",
        scenario="{scenario}",
        task="{task}",
        model="{model}",
        training="{training}",
        feats="{feats}",
    log:
        "logs/training/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{strategy}/{feats}.log",
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

rule:
    name: f"eval_strategy"
    input:
        "out/training/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{strategy}",
    output:
        directory("out/evaluation/strategy/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{strategy}"),
    log:
        "logs/evaluation/strategy/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{strategy}.log",
    script: "../scripts/evaluation/plot-bar.py"

def get_pipeline_output():
    final_output = []
    return final_output
