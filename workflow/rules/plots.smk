from pathlib import Path

from helper import get_dataset_machines

DATASETS = ["alibaba"]

wildcard_constraints:
    dataset = "|".join(DATASETS)

# def get_plot_inputs():
#     final_inputs = []
#     for dataset in DATASETS:
#         for file in get_dataset_files(config, dataset):
#             final_inputs.append(file)
#     return final_inputs

dataset_machines = get_dataset_machines(config, DATASETS)
SCENARIO = [
    "offline-classification-no-retrain",
    "offline-classification-retrain-chunks-from-scratch",
    "offline-classification-retrain-chunks-naive",
]

# def generate_inputs(dataset, machine, scenario):
#     def __generate(_):
#         inputs = []
#         for scenario in SCENARIO:
#             inputs.append(f"out/training/{dataset}/{machine}/{scenario}")
#         return inputs
#     return __generate

for dataset in dataset_machines:
    for machine in dataset_machines[dataset]:
        rule:
            name: f"plot_auroc_{dataset}_{machine}"
            input:
                # generate_inputs(dataset, machine, SCENARIO),
                # f"out/training/{dataset}/{machine}/{{scenario}}"
                expand(f"out/training/{dataset}/{machine}/{{scenario}}", scenario=SCENARIO),
                # dataset=dataset,
                # machine=machine),
            output:
                directory(f"out/results/{dataset}/{machine}/auroc"),
            log:
                f"logs/results/{dataset}/{machine}/plot_auroc.log",
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
