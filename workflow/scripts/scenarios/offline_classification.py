import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, RandomSampler, random_split

from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.evaluation.metrics import confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.checkpoint import (
    maybe_load_checkpoint,
    save_checkpoint,
)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import FromScratchTraining, Naive

from src.utils.logging import logging, setup_logging

if TYPE_CHECKING:
    snakemake: Any = None


setup_logging(snakemake.log[0])

from src.metrics import get_classification_default_metrics
from src.models import MLP
from src.utils.general import set_seed
from src.utils.summary import generate_summary, generate_summary_table

log = logging.getLogger(__name__)

def print_and_write(msg, out_file):
    print(msg)
    out_file.write(f"{msg}\n")



def get_device(cfg: dict):
    device = "cpu"
    cuda_config = cfg.get("cuda", {})
    if (
        "enable" in cuda_config
        and "device" in cuda_config
        and cuda_config["enable"]
        and cuda_config["device"] >= 0
        and torch.cuda.is_available()
    ):
        device = f"cuda:{cuda_config['device']}"

    return torch.device(device)

def get_optimizer(cfg: dict, model: torch.nn.Module):
    if "name" not in cfg:
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=0.001)
    else:
        if cfg["name"].lower() == "adam":
            from torch.optim import Adam
            optimizer = Adam(model.parameters(), lr=cfg["lr"])
        elif cfg["name"].lower() == "sgd":
            from torch.optim import SGD
            optimizer = SGD(model.parameters(), lr=cfg["lr"])
        else:
            raise ValueError("Unknown optimizer")
     
    return optimizer

def get_dataset(dataset: str, scenario: str, input_path: Path, y: str, num_classes: int = 10, num_split: int = 4):
    match dataset:
        case "alibaba":
            from src.dataset.alibaba import (
                get_classification_alibaba_machine_dataset_splitted as Dataset,
            )
        case "google":
            from src.dataset.google import (
                get_classification_google_machine_dataset_splitted as Dataset,
            )
        case _:
            raise ValueError("Unknown dataset")

    match scenario:
        case "offline_no_retrain" | "offline_classification_retrain_chunks_from_scratch" | "offline_classification_retrain_chunks_naive":
            dataset = Dataset(
                filename=input_path,
                n_labels=num_classes,
                y=y,
                num_split=num_split,
            )
            input_size = dataset[0].original_test_dataset.input_size
            if len(dataset) == 0:
                raise ValueError("No data in dataset")
        case _:
            raise ValueError("Unknown scenario")
    return dataset, input_size


def get_trainer(scenario: str, cl_strategy, benchmark, num_workers: int = 4):
    match scenario:
        case "offline_no_retrain":
            from src.trainers import OfflineNoRetrainingTrainer
            trainer = OfflineNoRetrainingTrainer(cl_strategy, benchmark=benchmark, num_workers=num_workers)
        case "offline_classification_retrain_chunks_from_scratch" | "offline_classification_retrain_chunks_naive":
            from src.trainers import OfflineRetrainingTrainer 
            trainer = OfflineRetrainingTrainer(cl_strategy, benchmark=benchmark, num_workers=num_workers)
        case _:
            raise ValueError("Unknown scenario")

    return trainer

def get_benchmark(scenario, dataset):
    match scenario:
        case "offline_no_retrain" | "offline_classification_retrain_chunks_from_scratch" | "offline_classification_retrain_chunks_naive":
            # creating benchmark
            train_subsets = [subset.train_dataset for subset in dataset]
            test_subsets = [subset.test_dataset for subset in dataset]
            benchmark = dataset_benchmark(train_subsets, test_subsets)
        case _:
            raise ValueError("Unknown scenario")
    return benchmark

def save_train_results(results: dict, output_folder: Path, model: torch.nn.Module):
    # Cleaning up ====
    with open(output_folder / "train_results.json", "w") as results_file:
        json.dump(results, results_file, default=lambda _: "<not serializable>")

    # Save model ====
    log.info("Saving model")
    model_name = output_folder / "model.pt"
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_name)

    # Print results ====
    out_file = open(output_folder / "train_results.txt", "w")
    for key in results:
        print_and_write(f"Experience #{key}", out_file)
        sorted_results = sorted(
            results[key].keys(), key=lambda x: x.lower().strip()
        )
        for k in sorted_results:
            print_and_write(f"{k.strip()}: {results[key][k]}", out_file)
        print_and_write("", out_file)

    # Generating summary ====
    log.info("Printing summary")
    summary = generate_summary(output_folder / "train_results.json")
    table = generate_summary_table(summary)
    print_and_write(table.draw(), out_file)



def evaluation(test_stream):
    pass


def main():
    config = snakemake.config
    dataset_config = snakemake.params.dataset_config
    training_config = snakemake.params.training_config
    scenario = snakemake.params.scenario
    dataset = snakemake.params.dataset

    set_seed(config.get("seed", 0))
    input_path = Path(str(snakemake.input))
    log.info(f"Input path: {input_path}")

    current_time = int(datetime.now().timestamp())
    run_name = f"{current_time}_{dataset}_{scenario}_{input_path.stem}"

    output_folder = Path(str(snakemake.output))
    output_folder.mkdir(parents=True, exist_ok=True)
    log.info(f"Output path: {output_folder}")

    scenario_config = {}

    if hasattr(snakemake.params, "scenario_config"):
        scenario_config = snakemake.params.scenario_config

    log.info("Run name: %s", run_name)
    log.info("Scenario: %s", scenario)
    log.info("Dataset: %s", dataset)
    log.info("Dataset config: %s", dataset_config)
    log.info("Training config: %s", training_config)
    log.info("Scenario config: %s", scenario_config)

    # Device ====
    device = get_device(config)
    log.info("Device: %s", device)

    num_classes = training_config["classification"]["num_classes"] 
    log.info("Number of classes: %d", num_classes)
    
    dataset, input_size = get_dataset(
        dataset=dataset,
        scenario=scenario,
        input_path=input_path, 
        y=dataset_config["y"], 
        num_classes=num_classes, 
        num_split=scenario_config.get("num_split", 4))
    log.info(f"Input size: {input_size}")


    # Model ====
    model_config = snakemake.params.model_config
    model = MLP(
        input_size=input_size,
        num_classes=num_classes,
        **model_config,
    )

    # Optimizer + Loss ====
    optimizer_config = training_config["classification"]["optimizer"]
    optimizer = get_optimizer(optimizer_config, model)
    criterion = CrossEntropyLoss()

    benchmark = get_benchmark(scenario, dataset)

    # Loggers ====
    wandb_config = config.get("wandb", {})

    loggers = [
        TextLogger(sys.stderr),
    ]

    wandb_enable = False
    if "enable" in wandb_config and wandb_config["enable"]:
        from avalanche.logging.wandb_logger import WandBLogger
        wandb_enable = True
        loggers.append(WandBLogger(project_name=wandb_config["project"], run_name=run_name))


    # Evaluation ====
    eval_plugin = EvaluationPlugin(
        *get_classification_default_metrics(num_classes= num_classes, tolerance=training_config["classification"]["evaluation"]["tolerance"]),
        confusion_matrix_metrics(
            stream=True, wandb=wandb_enable, class_names=[str(i) for i in range(num_classes)], save_image=True
        ),
        loggers=loggers
    )

    # Strategy ====
    match scenario:
        case "offline_no_retrain" | "offline_classification_retrain_chunks_from_scratch":
            cl_strategy = FromScratchTraining(
                model,
                optimizer,
                criterion,
                train_mb_size=training_config["batch_size"],
                train_epochs=training_config["epochs"],
                eval_mb_size=training_config["batch_size"],
                evaluator=eval_plugin,
                device=device,
                # eval_every=-1,  # at the end of each experience
            )         
        case "offline_classification_retrain_chunks_naive": 
            cl_strategy = Naive(
                model,
                optimizer,
                criterion,
                train_mb_size=training_config["batch_size"],
                train_epochs=training_config["epochs"],
                eval_mb_size=training_config["batch_size"],
                evaluator=eval_plugin,
                device=device,
                # eval_every=-1,  # at the end of each experience
            )
        case _:
            raise ValueError("Unknown scenario")
        
    # Training ====
    log.info("Starting training...")
    trainer = get_trainer(scenario, cl_strategy=cl_strategy, benchmark=benchmark, num_workers=config.get("num_workers", 4))
    train_results = trainer.train()
    log.info("Training finished")

    save_train_results(results=train_results, output_folder=output_folder, model=model)

    # # Additional evaluation ====
    # log.info("Additional evaluation...")

if __name__ == "__main__":
    main()
