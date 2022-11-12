import argparse
import collections
import json
from pathlib import Path
import torch
import data_utils.data_loaders as module_data
import pipeline.metrics as module_metric
import models as module_arch
from utils import ConfigParser
from pipeline.testers import Tester
from utils.project_utils import prepare_device


def config_model(config):
    cfg_path = Path(config["model_path"]).resolve().parents[1] / "config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    config._config["arch"] = cfg["arch"]


def main(config: ConfigParser) -> None:
    # logger
    logger = config.get_logger("test")
    config.ensure_reproducibility()

    # data_loader setup
    data_loader = config.init_obj("data_loader", module_data)

    # build model
    config_model(config)
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU testing
    device, device_ids = prepare_device(config["device"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    tester = Tester(
        model=model,
        metric_ftns=metrics,
        config=config,
        device=device,
        data_loader=data_loader,
        only_predict=config["tester"]["only_predict"],
    )

    tester.test()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Model tester.")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target="data_loader;args;batch_size",
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
