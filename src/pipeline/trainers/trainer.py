import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from base import BaseTrainer
from pipeline import pipeline_utils


class Trainer(BaseTrainer):
    """Trainer class."""

    def __init__(
        self,
        model: torch.nn.Module,
        metric_ftns: list,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: torch.device,
        data_loader: torch.utils.data.DataLoader,
        lr_scheduler: torch.optim.lr_scheduler,
        do_validation: bool = True,
        len_epoch: int = None,
    ) -> None:
        """Trainer constructor.

        Args:
            model (torch.nn.Module): model to train
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
            model=model,
            metric_ftns=metric_ftns,
            optimizer=optimizer,
            config=config,
            device=device,
            data_loader=data_loader,
            lr_scheduler=lr_scheduler,
            len_epoch=len_epoch,
            do_validation=do_validation,
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

        progress_bar = tqdm(
            enumerate(self.train_data_loader),
            desc=f"Training epoch {epoch}",
            colour="blue",
            total=len(self.train_data_loader),
        )
        pbar_loss = "None"

        for batch_idx, (data, target) in progress_bar:
            progress_bar.set_postfix({"loss": pbar_loss})

            data = pipeline_utils.move_tensors_to_device(data, self.device)
            target = pipeline_utils.move_tensors_to_device(target, self.device)
            self.optimizer.zero_grad()

            output = self.model(data).squeeze()
            loss = self.calc_loss(output, target)
            loss.backward()
            pbar_loss = loss.item()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            output = self.model.get_prediction(output)

            target = pipeline_utils.cpu_tensors(target)
            output = pipeline_utils.cpu_tensors(output)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )
                self.writer.add_image(
                    "input",
                    make_grid(
                        pipeline_utils.cpu_tensors(data),
                        nrow=8,
                        normalize=True,
                    ),
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

        progress_bar = tqdm(
            enumerate(self.valid_data_loader),
            desc=f"Validating epoch {epoch}",
            colour="green",
            total=len(self.valid_data_loader),
        )
        pbar_loss = "None"

        preds, targets = [], []
        with torch.no_grad():
            for batch_idx, (data, target) in progress_bar:
                progress_bar.set_postfix({"loss": pbar_loss})

                data = pipeline_utils.move_tensors_to_device(data, self.device)
                target = pipeline_utils.move_tensors_to_device(
                    target, self.device
                )

                output = self.model(data).squeeze()
                loss = self.calc_loss(output, target)
                pbar_loss = loss.item()

                output = self.model.get_prediction(output)

                output = pipeline_utils.cpu_tensors(output)
                target = pipeline_utils.cpu_tensors(target)
                preds.append(output)
                targets.append(target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx,
                    "valid",
                )
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output, target)
                    )
                self.writer.add_image(
                    "input",
                    make_grid(
                        pipeline_utils.cpu_tensors(data),
                        nrow=8,
                        normalize=True,
                    ),
                )

        if self.data_loader.is_multilabel:
            preds = torch.cat(preds)
            targets = torch.cat(targets)
            pipeline_utils.print_per_class_metrics(
                targets=targets,
                predictions=preds,
                cls_map=self.data_loader.labels,
                logger=self.logger,
            )

        # adding histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()

    def calc_loss(self, output, target):
        return self.model.calculate_loss(output=output, target=target)
