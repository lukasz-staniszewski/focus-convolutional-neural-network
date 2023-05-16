import torch
from base import BaseTrainer
from tqdm import tqdm
from pipeline import pipeline_utils


class FocusCNNTrainer(BaseTrainer):
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

        for batch_idx, data in progress_bar:
            progress_bar.set_postfix({"loss": pbar_loss})

            data_in = pipeline_utils.move_tensors_to_device(
                data[0], device=self.device
            )
            img_ids = pipeline_utils.move_tensors_to_device(
                data[1], device=self.device
            )
            target = pipeline_utils.move_tensors_to_device(
                data[2],
                device=self.device,
                dict_columns=["label", "transform"],
            )

            output = self.model(data_in, target)
            loss = output["loss"]
            self.optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            loss.backward()
            self.optimizer.step()

            pbar_loss = loss.item()

            self.train_metrics.update("loss", loss.item())

            preds = self.model.get_prediction(output, img_ids)

            # TODO: create input for map metric

            preds = pipeline_utils.cpu_tensors(preds)
            target = pipeline_utils.cpu_tensors(target)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(preds, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
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

        with torch.no_grad():
            for batch_idx, data in progress_bar:
                progress_bar.set_postfix({"loss": pbar_loss})
                data_in = pipeline_utils.move_tensors_to_device(
                    data[0], device=self.device
                )
                img_ids = pipeline_utils.move_tensors_to_device(
                    data[1], device=self.device
                )
                target = pipeline_utils.move_tensors_to_device(
                    data[2],
                    device=self.device,
                    dict_columns=["label", "transform"],
                )
                output = self.model(data_in, target)
                loss_val = output["loss"]
                pbar_loss = loss_val.item()

                self.valid_metrics.update("val_loss", loss_val.item())
                preds = self.model.get_prediction(output, img_ids)

                # here also for map metric
                preds = pipeline_utils.cpu_tensors(preds)
                target = pipeline_utils.cpu_tensors(target)

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(preds, target))

        return self.valid_metrics.result()
