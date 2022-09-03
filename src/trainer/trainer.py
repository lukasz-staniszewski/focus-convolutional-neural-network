import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """Trainer class."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        metric_ftns: list,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: torch.device,
        data_loader: torch.utils.data.DataLoader,
        valid_data_loader: torch.utils.data.DataLoader = None,
        lr_scheduler: torch.optim.lr_scheduler = None,
        len_epoch: int = None,
    ) -> None:
        """Trainer constructor.

        Args:
            model (torch.nn.Module): model to train
            criterion (torch.nn.Module): loss function
            metric_ftns (list): metrics to compute
            optimizer (torch.optim.Optimizer): optimizer to use
            config (dict): config dictionary
            device (torch.device): device to use
            data_loader (torch.utils.data.DataLoader): data loader for training
            valid_data_loader (torch.utils.data.DataLoader, optional): data loader for validation. Defaults to None.
            lr_scheduler (torch.optim.lr_scheduler, optional): learning rate scheduler. Defaults to None.
            len_epoch (int, optional): if provided, then iteration-based training is performed with time. Defaults to None.
        """
        super().__init__(
            model, criterion, metric_ftns, optimizer, config
        )
        self.config = config
        self.device = device
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_metrics = MetricTracker(
            "loss",
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss",
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer
        )

    def _train_epoch(self, epoch: int) -> dict:
        """Training logic for an epoch.

        Args:
            epoch (int): current training epoch

        Returns:
            dict: log that contains average loss and metric in this epoch
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target, self.device)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step(
                (epoch - 1) * self.len_epoch + batch_idx
            )
            self.train_metrics.update("loss", loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(output, target)
                )

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )
                self.writer.add_image(
                    "input",
                    make_grid(data.cpu(), nrow=8, normalize=True),
                )
            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch: int) -> dict:
        """Validation logic for an epoch.

        Args:
            epoch (int): current training epoch

        Returns:
            dict: log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(
                self.valid_data_loader
            ):
                data, target = data.to(self.device), target.to(
                    self.device
                )
                output = self.model(data)
                loss = self.criterion(output, target, self.device)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader)
                    + batch_idx,
                    "valid",
                )
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output, target)
                    )
                self.writer.add_image(
                    "input",
                    make_grid(data.cpu(), nrow=8, normalize=True),
                )

        # adding histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()

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
