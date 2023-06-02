# include: "model_baseline.smk"

DATASETS = ["alibaba"]

wildcard_constraints:
    dataset = "|".join(DATASETS)

def get_plot_inputs():
    final_inputs = []
    for dataset in DATASETS:
        for file in get_dataset_files(config, dataset):
            final_inputs.append(file)

    return final_inputs

rule plot_auroc:
    input:
        # get_model_baseline_output(),
        expand("out/training/offline-classification-retrain-chunks-from-scratch/{dataset}/{path}", dataset="alibaba", path="m_881"),
        expand("out/training/offline-classification-retrain-chunks-naive/{dataset}/{path}", dataset="alibaba", path="m_881"),
        expand("out/training/offline-classification-no-retrain/{dataset}/{path}", dataset="alibaba", path="m_881"),
        # "out/training/offline-classification-retrain-chunks-from-scratch/{{dataset}}/{{path}}",
    output:
        directory(expand("out/results/{{dataset}}/{{path}}/auroc", dataset="alibaba", path="m_881")),
        # "out/results/{{dataset}}/{{path}}/auroc.png",
    # params:
    #     dataset="{{dataset}}",
    #     path="{{path}}",
    log:
        expand("logs/plot_auroc/{{dataset}}/{{path}}/auroc.log", dataset="alibaba", path="m_881"),
        # "logs/{{dataset}}/{{path}}/auroc.log",
    script: "../scripts/plots/plot_bar.py"
