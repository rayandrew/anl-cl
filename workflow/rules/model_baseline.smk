from pathlib import Path
from helper import get_dataset_files

DATASETS = get_datasets(config)

wildcard_constraints:
    dataset = "|".join(DATASETS)

SCENARIOS = [
    "offline_classification_no_retrain",
    "offline_classification_retrain_chunks_from_scratch", 
    "offline_classification_retrain_chunks_naive",
    "offline_classification_retrain_chunks_gss_greedy"
]

for scenario in SCENARIOS:
    rule:
        name: scenario
        input:
            "raw_data/{dataset}/{path}.parquet",
        output:
            directory("out/training/{dataset}/{path}/" + f"{scenario}"),
        params:
            dataset="{dataset}",
            data_path="{path}",
            dataset_config=get_database_config(config),
            training_config=config["training"]["offline"],
            model_config=config["model"],
            scenario=f"{scenario}",
            scenario_config=config["scenario"][scenario],
            task="classification",
        log:
            "logs/training/{dataset}/{path}/" + f"{scenario}.log",
        script: "../scripts/scenarios/offline_classification.py"

# rule offline_classification_no_retrain:
#     input:
#         "raw_data/{dataset}/{path}.parquet",
#     output:
#         directory("out/training/{dataset}/{path}/offline-classification-no-retrain"),
#     params:
#         dataset="{dataset}",
#         data_path="{path}",
#         dataset_config=get_database_config(config),
#         training_config=config["training"]["offline"],
#         model_config=config["model"],
#         scenario="offline_no_retrain",
#         scenario_config=config["scenario"]["offline_no_retrain"],
#         task="classification",
#     log:
#         "logs/training/{dataset}/{path}/offline-classification-no-retrain.log",
#     script: "../scripts/scenarios/offline_classification.py"

# use rule offline_classification_no_retrain as offline_classification_retrain_chunks_from_scratch with:
#     output:
#         directory("out/training/{dataset}/{path}/offline-classification-retrain-chunks-from-scratch"),
#     params:
#         dataset="{dataset}",
#         data_path="{path}",
#         dataset_config=get_database_config(config),
#         training_config=config["training"]["offline"],
#         model_config=config["model"],
#         scenario="offline_classification_retrain_chunks_from_scratch",
#         scenario_config=config["scenario"]["offline_retrain_chunks_from_scratch"],
#         task="classification",
#     log:
#         "logs/training/{dataset}/{path}/offline-classification-retrain-chunks-from-scratch.log",

# use rule offline_classification_retrain_chunks_from_scratch as offline_classification_retrain_chunks_naive with:
#     output:
#         directory("out/training/{dataset}/{path}/offline-classification-retrain-chunks-naive"),
#     params:
#         dataset="{dataset}",
#         data_path="{path}",
#         dataset_config=get_database_config(config),
#         training_config=config["training"]["offline"],
#         model_config=config["model"],
#         scenario="offline_classification_retrain_chunks_naive",
#         scenario_config=config["scenario"]["offline_retrain_chunks_naive"],
#         task="classification",
#     log:
#         "logs/training/{dataset}/{path}/offline-classification-retrain-chunks-naive.log",

def get_model_baseline_output():
    # final_output = expand("out/training/{dataset}/{path}/{scenario}", dataset=DATASETS, path=get_all_dataset_files(DATASETS), scenario=SCENARIOS)
    final_output = []
    for dataset in DATASETS:
        for file in get_dataset_files(config, dataset):
            filepath = file.relative_to(config["dataset"][dataset]["path"]).with_suffix("")
            final_output += expand("out/training/{dataset}/{path}/{scenario}", dataset=dataset, path=filepath, scenario=SCENARIOS)
            # final_output += expand("out/training/{dataset}/{path}/offline-classification-no-retrain", dataset=dataset, path=filepath)
            # final_output += expand("out/training/{dataset}/{path}/offline-classification-retrain-chunks-from-scratch", dataset=dataset, path=filepath)
            # final_output += expand("out/training/{dataset}/{path}/offline-classification-retrain-chunks-naive", dataset=dataset, path=filepath)

    return final_output
