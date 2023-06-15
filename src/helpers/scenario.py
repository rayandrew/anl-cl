from pathlib import Path
from typing import Any, Callable, Tuple

from torch.nn import CrossEntropyLoss

from avalanche.benchmarks.scenarios import (
    GenericCLScenario,
    OnlineCLScenario,
)
from avalanche.evaluation.metrics import confusion_matrix_metrics
from avalanche.logging import TextLogger
from avalanche.training.plugins import EvaluationPlugin

from src.metrics import get_classification_default_metrics
from src.utils.general import set_seed
from src.utils.logging import logging, setup_logging
from src.utils.time import get_current_time

from .config import Config, assert_config_params
from .definitions import Strategy as StrategyEnum
from .definitions import Training
from .device import get_device
from .model import get_model
from .optimizer import get_optimizer
from .strategy import get_strategy
from .trainer import get_trainer, save_train_results

GetDatasetFn = Callable[[Config, Path], Tuple[Any, int]]
GetBenchmarkFn = Callable[[Any], GenericCLScenario | OnlineCLScenario]


def train_classification_scenario(
    snakemake: Any,
    get_dataset: GetDatasetFn,
    get_benchmark: GetBenchmarkFn,
):
    setup_logging(snakemake.log[0])
    log = logging.getLogger(__name__)

    params = snakemake.params
    config = snakemake.config
    config = Config(**config)
    assert_config_params(config, params)

    log.info("Config: %s", config)

    set_seed(config.seed)
    input_path = Path(str(snakemake.input))
    log.info(f"Input path: {input_path}")

    current_time = get_current_time()

    training_type = (
        Training.ONLINE if config.online else Training.BATCH
    )
    run_name = f"{config.dataset.name}_{training_type}_{config.scenario.name}_{config.strategy.name}_{input_path.stem}_{current_time}"
    log.info(f"Run name: {run_name}")

    output_folder = Path(str(snakemake.output))
    output_folder.mkdir(parents=True, exist_ok=True)
    log.info(f"Output path: {output_folder}")

    log.info(f"Current time: %s", current_time)

    device = get_device()
    log.info("Device: %s", device)

    dataset, input_size = get_dataset(config, input_path)
    log.info(f"Input size: {input_size}")

    # Evaluation ====
    loggers = [TextLogger(sys.stderr)]

    wandb_logger = None
    if config.wandb is not None:
        from avalanche.logging.wandb_logger import WandBLogger

        wandb_logger = WandBLogger(
            project_name=config.wandb, run_name=run_name
        )
        wandb_logger.wandb.config.config = config.dict()
        loggers.append(wandb_logger)

    eval_plugin = EvaluationPlugin(
        *get_classification_default_metrics(
            num_classes=config.num_classes,
            tolerance=config.eval_tol,
        ),
        confusion_matrix_metrics(
            stream=True,
            wandb=wandb_logger is not None,
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

    Strategy = get_strategy(config)
    additional_strategy_params = config.strategy.dict(
        exclude={"name"}
    )
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

    benchmark = get_benchmark(dataset)
    Trainer = get_trainer(config)
    trainer = Trainer(
        strategy, benchmark, num_workers=config.num_workers
    )
    log.info("Starting training")
    results = trainer.train()
    log.info("Training finished")

    log.info("Generating summary and saving training results")
    save_train_results(results, output_folder, model)

    log.info("Finished")


__all__ = ["train_classification_scenario"]
