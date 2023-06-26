from pathlib import Path

import torch

from src.utils.io import Transcriber
from src.utils.logging import logging
from src.utils.summary import generate_summary, generate_summary_table

from .config import Config
from .definitions import Strategy

log = logging.getLogger(__name__)


def get_batch_trainer(strategy: Strategy):
    if strategy == Strategy.NO_RETRAIN:
        from src.trainers import BatchNoRetrainTrainer

        return BatchNoRetrainTrainer

    from src.trainers import BatchSimpleRetrainTrainer

    return BatchSimpleRetrainTrainer


def get_trainer(config: Config):
    if config.online:
        raise ValueError("Online trainer not implemented yet")

    return get_batch_trainer(config.strategy.name)


# def _get_benchmark(scenario: Scenario, dataset: Any):
#     match scenario:
#         case Scenario.SPLIT_CHUNKS:
#             from avalanche.benchmarks.generators import (
#                 dataset_benchmark,
#             )

#             train_subsets = [subset.train_dataset for subset in dataset]
#             test_subsets = [subset.test_dataset for subset in dataset]
#             benchmark = dataset_benchmark(train_subsets, test_subsets)
#             return benchmark
#         case _:
#             raise ValueError("Unknown scenario")

# def get_benchmark(cfg: Config, dataset: Any):
#     return _get_benchmark(cfg.scenario.name, dataset)


def save_train_results(
    results: dict, output_folder: Path, model: torch.nn.Module
):
    import simplejson

    # Cleaning up ====
    with open(
        output_folder / "train_results.json", "w"
    ) as results_file:
        simplejson.dump(
            results,
            results_file,
            ignore_nan=True,
            default=lambda _: "<not serializable>",
        )

    # Save model ====
    log.info("Saving model")
    model_name = output_folder / "model.pt"
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_name)

    # Print results ====
    log.info("Saving training results")
    out_file = Transcriber(output_folder / "train_results.txt")
    for key in results:
        out_file.write_line(f"Experience #{key}")
        sorted_results = sorted(
            results[key].keys(), key=lambda x: x.lower().strip()
        )
        for k in sorted_results:
            out_file.write_line(f"{k.strip()}: {results[key][k]}")
        out_file.write_line("")

    # Generating summary ====
    log.info("Printing summary")
    summary = generate_summary(output_folder / "train_results.json")
    table = generate_summary_table(summary)
    out_file.write_line(table.draw())
    out_file.close()


__all__ = ["get_trainer", "get_batch_trainer", "save_train_results"]
