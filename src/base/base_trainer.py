import os
from abc import abstractmethod

import numpy as np
import torch
from numpy import inf

from base import BaseModel
from utils import ConfigParser, MetricTrackerV2, inf_loop, secure_load_path


class BaseTrainer:
    """Base class for all trainers"""

    def __init__(
        self,
        model: BaseModel,
        metric_ftns: list,
        optimizer: torch.optim.Optimizer,
        config: ConfigParser,
        device: torch.device,
        data_loader: torch.utils.data.DataLoader,
        lr_scheduler: torch.optim.lr_scheduler,
        len_epoch: int = None,
        do_validation: bool = True,
    ) -> None:
        # members
        self.config = config
        self.model = model
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.device = device
        self.logger = config.get_logger(
            "trainer", config["trainer"]["verbosity"]
        )
        self.cfg_trainer = config["trainer"]
        self.epochs = self.cfg_trainer["epochs"]
        self.save_period = self.cfg_trainer["save_period"]
        self.monitor = self.cfg_trainer.get("monitor", "off")
        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
        self.data_loader = data_loader
        self.do_validation = do_validation
        self.lr_scheduler = lr_scheduler

        # data configuration
        self._configure_dataloaders()

        # configuration to monitor model performance and save best
        self._configure_metrics()

        # check if resuming training
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        # configuration of training
        self._configure_epoch_training(len_epoch=len_epoch)

    def _configure_metrics(self):
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf
        self.train_metrics = MetricTrackerV2(
            metrics_handlers=self.metric_ftns,
        )
        if self.do_validation:
            self.valid_metrics = MetricTrackerV2(
                metrics_handlers=self.metric_ftns,
            )

    def _configure_dataloaders(self):
        self.train_data_loader = self.data_loader.get_train_loader()
        if self.do_validation:
            self.valid_data_loader = self.data_loader.get_valid_loader()

    def _configure_epoch_training(self, len_epoch: int = None):
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(self.train_data_loader)
            self.len_epoch = len_epoch
        self.log_step = int(np.sqrt(self.data_loader.batch_size))

    def train(self):
        """Performs training."""
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {"epoch": epoch}
            log.update(result)
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            is_best = False
            if self.mnt_mode != "off":
                try:
                    improved = (
                        self.mnt_mode == "min"
                        and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == "max"
                        and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    is_best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {}"
                        " epochs. Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch=epoch, save_as_best=False)
            if is_best:
                self._save_checkpoint(epoch=epoch, save_as_best=True)

    def _save_checkpoint(self, epoch: int, save_as_best: bool = False) -> None:
        """By default saves checkpoint of path. If save_as_best is True, saves checkpoint as best.
        Args:
            epoch (int): current epoch number
            save_as_best (bool): if True, saves checkpoint as best; defaults to False
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        if save_as_best:
            path = os.path.join(self.checkpoint_dir / "model_best.pth")
            log_info = "Saving current best: model_best.pth ..."
        else:
            path = os.path.join(
                self.checkpoint_dir, f"checkpoint-epoch{epoch}.pth"
            )
            log_info = "Saving checkpoint: {} ...".format(path)

        torch.save(state, path)
        self.logger.info(log_info)

    def _resume_checkpoint(self, resume_path: str) -> None:
        """Resumes from saved checkpoint.

        Args:
            resume_path (str): checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        secure_load_path()
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture config given in file differs"
                " from the one of checkpoint. This may cause errors"
                " while state_dict is being loaded if differences"
                " are not caused by loss parameters."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        if (
            checkpoint["config"]["optimizer"]["type"]
            != self.config["optimizer"]["type"]
        ):
            self.logger.warning(
                "Warning: Optimizer type given in config file is"
                " different from that of checkpoint. Optimizer"
                " parameters can't be resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resuming training from epoch {}".format(
                self.start_epoch
            )
        )

    def _progress(self, batch_idx: int) -> str:
        """Progress bar logic.

        Args:
            batch_idx (int): current batch index

        Returns:
            str: progress bar
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch

        return base.format(current, total, 100.0 * current / total)

    def cpu_tensors(self, tensors):
        if isinstance(tensors, dict):
            tensors = {k: v.cpu() for (k, v) in tensors.items()}
        elif isinstance(tensors, tuple):
            tensors = tuple([t.cpu() for t in tensors])
        else:
            tensors = tensors.cpu()
        return tensors

    @abstractmethod
    def _train_epoch(self, epoch: int) -> None:
        """Training logic for an epoch

        Args:
            epoch (int): current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def calc_loss(self, output, target):
        raise NotImplementedError
