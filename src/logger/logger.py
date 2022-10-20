import logging
import logging.config
from pathlib import Path
from typing import Optional, Union
from utils.project_utils import read_json
import os


def setup_logging(
    save_dir: Union[str, Path],
    log_config: Union[str, Path] = "src/logger/logger_config.json",
    default_level: Optional[int] = logging.INFO,
) -> None:
    """Setups logging configuration.

    Args:
        save_dir (str): directory where logs will be saved
        log_config (str, optional): path to logging config file. Defaults to "src/logger/logger_config.json".
        default_level (int, optional): level of logging. Defaults to logging.INFO.
    """
    log_config = Path(log_config)

    if log_config.is_file():
        config = read_json(log_config)
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = os.path.join(save_dir, handler["filename"])
        logging.config.dictConfig(config)
    else:
        print("Warning - logging configuration file is not found in" f" {log_config}.")
        logging.basicConfig(level=default_level)
