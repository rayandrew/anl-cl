from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from src.utils.logging import logging, setup_logging

if TYPE_CHECKING:
    snakemake: Any = None

setup_logging(snakemake.log[0])

from src.helpers.config import Config
from src.helpers.dataset import Dataset
from src.helpers.device import get_device_from_config
from src.helpers.optimizer import get_optimizer_from_config
from src.helpers.strategy import Strategy
from src.metrics import get_classification_default_metrics
from src.models import MLP
from src.utils.general import set_seed
from src.utils.summary import generate_summary, generate_summary_table
from src.utils.time import get_current_time

log = logging.getLogger(__name__)


def main():
    config = snakemake.config
    config = Config(**config)

    log.info("Config: %s", config)

    # set_seed(config.get("seed", 0))
    # input_path = Path(str(snakemake.input))
    # log.info(f"Input path: {input_path}")

    # current_time = get_current_time()

    # # run_name = (
    # #     f"{dataset}_{scenario}_{input_path.stem}_{current_time}"
    # # )

    # output_folder = Path(str(snakemake.output))
    # output_folder.mkdir(parents=True, exist_ok=True)
    # log.info(f"Output path: {output_folder}")

    # scenario_config = {}

    # if hasattr(snakemake.params, "scenario_config"):
    #     scenario_config = snakemake.params.scenario_config

    # log.info(f"Current time: %s", current_time)
    # # log.info("Run name: %s", run_name)
    # # log.info("Scenario: %s", scenario)
    # # log.info("Dataset: %s", dataset)
    # # log.info("Dataset config: %s", dataset_config)
    # # log.info("Training config: %s", training_config)
    # log.info("Scenario config: %s", scenario_config)

    # device = get_device_from_config(config)
    # log.info("Device: %s", device)


print("HEREE")

if __name__ == "__main__":
    main()
