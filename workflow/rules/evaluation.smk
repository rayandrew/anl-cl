from pathlib import Path

from helper import DATASETS, EXTENSIONS, TRAININGS, TASKS, SCENARIOS, get_dataset_files

wildcard_constraints:
    dataset = "|".join(DATASETS),
    # ext = "|".join(EXTENSIONS),


for (training, task, scenario) in itertools.product(TRAININGS, TASKS, SCENARIOS):
    rule:
        name: f"plot_{training}_{task}_{scenario}"
        input:
            "out/training/{dataset}/{filename}/" + f"{task}/{scenario}",
        output:
            directory("out/results/{dataset}/{filename}/" + f"{task}/{scenario}"),
        log:
            "logs/results/{dataset}/{filename}/" + f"{task}_{scenario}.log",
        script: "../scripts/plots/plot_bar.py"