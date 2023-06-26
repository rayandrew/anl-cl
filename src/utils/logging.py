import logging
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

# credits to https://stackoverflow.com/a/65732832


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error(
        "".join(
            [
                "Uncaught exception: ",
                *traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                ),
            ]
        )
    )


def setup_logging(filename: str | Path = "snakemake.log"):
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG)
        format="%(asctime)s %(levelname)s: %(message)s",  # Set the log message format
        filename=filename,  # Specify the log file name
        filemode="w",  # Set the file mode ('w' for write, 'a' for append)
    )
    sys.excepthook = handle_exception


Logger = logging.Logger

__all__ = ["setup_logging", "logging", "Logger"]
