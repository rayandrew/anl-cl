from pathlib import Path
from helper import get_datasets, get_dataset_files, get_database_config

DATASETS = get_datasets(config)

wildcard_constraints:
    dataset = "|".join(DATASETS).replace("cori", "") # remove cori, we do not need DM for now

if "drift_detection" in config and "voting" in config["drift_detection"] and config["drift_detection"]["voting"]["enable"]:
    rule voting:
        input:
            "raw_data/{dataset}/{path}.csv",
            # expand("{path}", path=[config["dataset"][dataset]["path"] for dataset in DATASETS]),
        output:
            directory("out/dd/{dataset}/voting/{path}"),
        params:
            method="voting",
            window_size={config["drift_detection"]["voting"]["window_size"]},
            threshold=config["drift_detection"]["voting"]["threshold"],
            dataset_config=get_database_config(config),
            dataset="{dataset}",
            data_path="{path}",
        log:
            "logs/dd/{dataset}/voting/{path}.log",
        # resources:
        #     slurm_extra="--gres=gpu:1"
        script: "../scripts/drift_detection.py"

if "drift_detection" in config and "ruptures" in config["drift_detection"] and config["drift_detection"]["ruptures"]["enable"]:
    rule ruptures:
        input:
            "raw_data/{dataset}/{path}.csv",
            # expand("{path}", path=[config["dataset"][dataset]["path"] for dataset in DATASETS]),
        output:
            directory("out/dd/{dataset}/ruptures/{path}"),
        params:
            method="ruptures",
            kernel=config["drift_detection"]["ruptures"]["kernel"],
            min_size=config["drift_detection"]["ruptures"]["min_size"],
            jump=config["drift_detection"]["ruptures"]["jump"],
            penalty=config["drift_detection"]["ruptures"]["penalty"],
            dataset_config=get_database_config(config),
            dataset="{dataset}",
            data_path="{path}",
        log:
            "logs/dd/{dataset}/ruptures/{path}.log",
        # resources:
        #     slurm_extra="--gres=gpu:1"
        script: "../scripts/drift_detection.py"



def get_drift_detection_output():
    final_output = []
    for dataset in DATASETS:
        for file in get_dataset_files(config, dataset):
            if "drift_detection" in config and "voting" in config["drift_detection"] and config["drift_detection"]["voting"]["enable"]:
                final_output += expand("out/dd/{dataset}/voting/{path}", dataset=dataset, path=file.relative_to(config["dataset"][dataset]["path"]).with_suffix(""))
            if "drift_detection" in config and "ruptures" in config["drift_detection"] and config["drift_detection"]["ruptures"]["enable"]:
                final_output += expand("out/dd/{dataset}/ruptures/{path}", dataset=dataset, path=file.relative_to(config["dataset"][dataset]["path"]).with_suffix(""))

            
    return final_output
