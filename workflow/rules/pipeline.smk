from pathlib import Path

from helper import DATASETS, EXTENSIONS, SCENARIOS, STRATEGIES, TRAININGS, TASKS, MODELS
# from helper import get_all_dataset_files, get_all_dataset_files_as_dict

# wildcard_constraints:
#     dataset = "|".join(DATASETS),
#     ext = "|".join(EXTENSIONS),
#     scenario = "|".join(SCENARIOS),
#     strategy = "|".join(STRATEGIES),
#     training = "|".join(TRAININGS),
#     task = "|".join(TASKS),
#     model = "|".join(MODELS)

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
        directory("out/training/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{feats}/{strategy}"),
    params:
        dataset="{dataset}",
        filename="{filename}",
        scenario="{scenario}",
        task="{task}",
        model="{model}",
        training="{training}",
        feats="{feats}",
        strategy="{strategy}",
    log:
        "logs/training/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{feats}/{strategy}.log",
    script: "../scripts/pipeline/{wildcards.scenario}.py"

rule:
    name: f"eval_scenario"
    input:
        # "out/training/{dataset}/{filename}/{task}/{training}/{scenario}",
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
    name: f"eval_feats"
    input:
        "out/training/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{feats}",
    output:
        directory("out/evaluation/feats/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{feats}"),
    log:
        "logs/evaluation/feats/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{feats}.log",
    script: "../scripts/evaluation/plot-bar.py"

rule:
    name: f"eval_strategy"
    input:
        "out/training/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{feats}/{strategy}",
    output:
        directory("out/evaluation/strategy/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{feats}/{strategy}"),
    log:
        "logs/evaluation/strategy/{dataset}/{filename}/{task}/{training}/{scenario}/{model}/{feats}/{strategy}.log",
    script: "../scripts/evaluation/plot-bar.py"

# def get_pipeline_output():
#     final_output = []
#     return final_output

# vim: set ft=snakemake:python:
