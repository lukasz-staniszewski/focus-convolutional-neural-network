import argparse
import data_utils.preprocessors as module_preprocessor
from utils import ConfigParser
import os

from utils.project_utils import read_json


def main(config):
    preprocessor = config.init_obj(
        name="preprocessor", module=module_preprocessor
    )
    preprocessor.preprocess()


def parse_config(args):
    assert os.path.exists(
        args.config
    ), f"File not found at path: {args.config}"
    cfg_json = read_json(args.config)
    assert all(
        [
            k in cfg_json
            for k in ["name", "preprocessor", "save_cfg_dir"]
        ]
    ), "Invalid config file keys!"
    config = ConfigParser.from_args(args=args)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Preprocessor")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    config = ConfigParser.from_args(args=args)
    main(config)
