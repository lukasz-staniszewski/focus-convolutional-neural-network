import argparse
import os

import data_utils.preprocessors as module_preprocessor
from utils import ConfigParser
from utils.project_utils import read_json


def main(config: ConfigParser) -> None:
    # logger
    config.ensure_reproducibility()

    preprocessor = config.init_obj(
        name="preprocessor", module=module_preprocessor
    )
    preprocessor.preprocess()


def parse_config(args):
    config_path = args.parse_args().config
    assert os.path.exists(
        config_path
    ), f"File not found at path: {config_path}"
    cfg_json = read_json(config_path)
    assert all(
        [k in cfg_json for k in ["name", "preprocessor", "save_cfg_dir"]]
    ), "Invalid config file keys!"
    return ConfigParser.from_args(args=args)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Data preprocessor.")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-s",
        "--seed",
        default=42,
        type=int,
        help="random seed for reproducibility",
    )
    config = ConfigParser.from_args(args)
    main(config)
