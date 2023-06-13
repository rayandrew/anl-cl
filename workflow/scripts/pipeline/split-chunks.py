import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from torch.nn import CrossEntropyLoss

from avalanche.evaluation.metrics import confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin

from src.helpers.config import Config, assert_config_params
from src.helpers.dataset import get_splitted_dataset
from src.helpers.definitions import Strategy as StrategyEnum
from src.helpers.device import get_device
from src.helpers.model import get_model
from src.helpers.optimizer import get_optimizer
from src.helpers.strategy import get_strategy
from src.helpers.trainer import (
    get_benchmark,
    get_trainer,
    save_train_results,
)
from src.metrics import get_classification_default_metrics
from src.utils.general import set_seed
from src.utils.io import Transcriber
from src.utils.logging import logging, setup_logging
from src.utils.summary import generate_summary, generate_summary_table
from src.utils.time import get_current_time

if TYPE_CHECKING:
    snakemake: Any

setup_logging(snakemake.log[0])
log = logging.getLogger(__name__)


def main():
    params = snakemake.params
    config = snakemake.config
    config = Config(**config)
    assert_config_params(config, params)

    log.info("Config: %s", config)

    set_seed(config.seed)
    input_path = Path(str(snakemake.input))
    log.info(f"Input path: {input_path}")

    current_time = get_current_time()

    run_name = f"{config.dataset.name}_{config.scenario.name}_{input_path.stem}_{current_time}"
    log.info(f"Run name: {run_name}")

    output_folder = Path(str(snakemake.output))
    output_folder.mkdir(parents=True, exist_ok=True)
    log.info(f"Output path: {output_folder}")

    log.info(f"Current time: %s", current_time)

    device = get_device()
    log.info("Device: %s", device)

    Dataset = get_splitted_dataset(config.dataset)
    dataset = Dataset(
        filename=input_path,
        y=config.dataset.y,
        num_split=config.scenario.num_split,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    input_size = dataset[0].original_test_dataset.input_size
    log.info(f"Input size: {input_size}")

    # Evaluation ====
    loggers = [TextLogger(sys.stderr)]

    wandb_enable = False
    if config.wandb is not None:
        from avalanche.logging.wandb_logger import WandBLogger

        loggers.append(
            WandBLogger(project_name=config.wandb, run_name=run_name)
        )
        wandb_enable = True

    eval_plugin = EvaluationPlugin(
        *get_classification_default_metrics(
            num_classes=config.num_classes,
            tolerance=config.eval_tol,
        ),
        confusion_matrix_metrics(
            stream=True,
            wandb=wandb_enable,
            class_names=[str(i) for i in range(config.num_classes)],
            save_image=True,
        ),
        loggers=loggers,
    )

    Model = get_model(config)
    model = Model(
        input_size=input_size,
        num_classes=config.num_classes,
        **config.model.dict(exclude={"name"}),
    )

    criterion = CrossEntropyLoss()

    Optimizer = get_optimizer(config)
    optimizer = Optimizer(
        model.parameters(), lr=config.tune.learning_rate[0]
    )
    # optimizer = Optimizer(

    Strategy = get_strategy(config)
    additional_strategy_params = {}
    if config.strategy.name == StrategyEnum.GSS:
        additional_strategy_params["input_size"] = [input_size]

    strategy = Strategy(
        model,
        optimizer,
        criterion,
        train_mb_size=config.batch,
        train_epochs=config.epochs,
        eval_mb_size=config.test_batch,
        evaluator=eval_plugin,
        device=device,
        **additional_strategy_params,
    )

    benchmark = get_benchmark(config, dataset)
    Trainer = get_trainer(config)
    trainer = Trainer(
        strategy, benchmark, num_workers=config.num_workers
    )
    log.info("Starting training")
    results = trainer.train()
    log.info("Training finished")

    log.info("Generating summary and saving training results")
    save_train_results(results, output_folder, model)


if __name__ == "__main__":
    main()
