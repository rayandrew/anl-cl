from pathlib import Path

from helper import get_all_dataset_files

DATASETS = ["alibaba"]

wildcard_constraints:
    dataset = "|".join(DATASETS)

# def get_plot_inputs():
#     final_inputs = []
#     for dataset in DATASETS:
#         for file in get_dataset_files(config, dataset):
#             final_inputs.append(file)
#     return final_inputs

dataset_files = get_all_dataset_files(config, DATASETS)
SCENARIO = [
    "offline_classification_no_retrain",
    "offline_classification_retrain_chunks_from_scratch",
    "offline_classification_retrain_chunks_naive",
    "offline_classification_retrain_chunks_gss_greedy",
]

# def generate_inputs(dataset, machine, scenario):
#     def __generate(_):
#         inputs = []
#         for scenario in SCENARIO:
#             inputs.append(f"out/training/{dataset}/{machine}/{scenario}")
#         return inputs
#     return __generate

for dataset in dataset_files:
    for filename in dataset_files[dataset]:
        rule:
            name: f"plot_classification_{dataset}_{filename}"
            input:
                f"out/training/classification/offline/{dataset}/{filename}",
            output:
                directory(f"out/results/{dataset}/{filename}/classification"),
            log:
                f"logs/results/{dataset}/{filename}/plot_classification.log",
            script: "../scripts/plots/plot_bar.py"

            

# rule plot_auroc:
#     input:
#         expand("out/training/{dataset}/{path}/{scenario}", dataset="alibaba", path="m_881"),
#         # "out/training/offline-classification-retrain-chunks-from-scratch/{{dataset}}/{{path}}",
#     output:
#         directory(expand("out/results/{{dataset}}/{{path}}/auroc", dataset="alibaba", path="m_881")),
#         # "out/results/{{dataset}}/{{path}}/auroc.png",
#     # params:
#     #     dataset="{{dataset}}",
#     #     path="{{path}}",
#     log:
#         expand("logs/plot_auroc/{{dataset}}/{{path}}/auroc.log", dataset="alibaba", path="m_881"),
#         # "logs/{{dataset}}/{{path}}/auroc.log",
#     script: "../scripts/plots/plot_bar.py"
