import argparse
import collections
import torch
import data_utils.data_loaders as module_data
import pipeline.trainers as module_trainers
import pipeline.metrics as module_metric
import models as module_arch
from utils import ConfigParser
from utils.project_utils import prepare_device


def main(config: ConfigParser) -> None:
    # logger
    logger = config.get_logger("train")
    config.ensure_reproducibility()

    # data_loader setup
    data_loader = config.init_obj("data_loader", module_data)

    # build model
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["device"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # function handles metrics
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # optimizer, lr scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        "lr_scheduler", torch.optim.lr_scheduler, optimizer
    )

    # trainer initialization
    trainer_class = getattr(module_trainers, config["trainer"]["type"])

    trainer = trainer_class(
        model,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Model trainer.")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
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
            ["--lr", "--learning_rate"],
            type=float,
            target="optimizer;args;lr",
        ),
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target="data_loader;args;batch_size",
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
