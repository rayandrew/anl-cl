from pathlib import Path
from helper import get_dataset_files

DATASETS = get_datasets(config)

wildcard_constraints:
    dataset = "|".join(DATASETS)

rule offline_classification_no_retrain:
    input:
        "raw_data/{dataset}/{path}.csv",
    output:
        directory("out/training/offline-classification-no-retrain/{dataset}/{path}"),
    params:
        dataset="{dataset}",
        data_path="{path}",
        dataset_config=get_database_config(config),
        training_config=config["training"]["offline"],
        model_config=config["model"],
        scenario="offline_no_retrain",
        scenario_config=config["scenario"]["offline_no_retrain"],
        task="classification",
    log:
        "logs/training/offline-classification-no-retrain/{dataset}/{path}.log",
    script: "../scripts/scenarios/offline_classification.py"

use rule offline_classification_no_retrain as offline_classification_retrain_chunks_from_scratch with:
    output:
        directory("out/training/offline-classification-retrain-chunks-from-scratch/{dataset}/{path}"),
    params:
        dataset="{dataset}",
        data_path="{path}",
        dataset_config=get_database_config(config),
        training_config=config["training"]["offline"],
        model_config=config["model"],
        scenario="offline_classification_retrain_chunks_from_scratch",
        scenario_config=config["scenario"]["offline_retrain_chunks_from_scratch"],
        task="classification",
    log:
        "logs/training/offline-classification-retrain-chunks-from-scratch/{dataset}/{path}.log",

use rule offline_classification_retrain_chunks_from_scratch as offline_classification_retrain_chunks_naive with:
    output:
        directory("out/training/offline-classification-retrain-chunks-naive/{dataset}/{path}"),
    params:
        dataset="{dataset}",
        data_path="{path}",
        dataset_config=get_database_config(config),
        training_config=config["training"]["offline"],
        model_config=config["model"],
        scenario="offline_classification_retrain_chunks_naive",
        scenario_config=config["scenario"]["offline_retrain_chunks_naive"],
        task="classification",
    log:
        "logs/training/offline-classification-retrain-chunks-naive/{dataset}/{path}.log",

def get_model_baseline_output():
    final_output = []
    for dataset in DATASETS:
        for file in get_dataset_files(config, dataset):
            filepath = file.relative_to(config["dataset"][dataset]["path"]).with_suffix("")
            final_output += expand("out/training/offline-classification-no-retrain/{dataset}/{path}", dataset=dataset, path=filepath)
            final_output += expand("out/training/offline-classification-retrain-chunks-from-scratch/{dataset}/{path}", dataset=dataset, path=filepath)
            final_output += expand("out/training/offline-classification-retrain-chunks-naive/{dataset}/{path}", dataset=dataset, path=filepath)

    return final_output
