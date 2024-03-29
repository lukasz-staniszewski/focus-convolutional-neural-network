import torch
from tqdm import tqdm

from base import BaseTrainer
from pipeline import pipeline_utils


class FocusTrainer(BaseTrainer):
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
        self.train_metrics.pred_columns = ["label", "bbox"]
        self.valid_metrics.pred_columns = ["label", "bbox"]

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

            data_in = pipeline_utils.to_device(
                data["image"], device=self.device
            )
            target = pipeline_utils.to_device(
                data, device=self.device, remove_keys=["image"]
            )

            output = self.model(data_in, target)
            loss_dict = output["loss"]
            loss = loss_dict["loss"]
            self.optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            loss.backward()
            self.optimizer.step()

            losses_dict = {k: v.item() for k, v in loss_dict.items()}

            pbar_loss = losses_dict["loss"]

            output = pipeline_utils.to_device(output, device="cpu")
            target = pipeline_utils.to_device(target, device="cpu")
            preds = self.model.get_prediction(output, target)

            self.train_metrics.update_batch(
                batch_model_outputs=preds,
                batch_expected_outputs=target,
                batch_loss=losses_dict,
            )

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), losses_dict["loss"]
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

                data_in = pipeline_utils.to_device(
                    data["image"], device=self.device
                )
                target = pipeline_utils.to_device(
                    data,
                    device=self.device,
                    remove_keys=["image"],
                )

                output = self.model(data_in, target)
                loss_dict = output["loss"]
                losses_dict = {k: v.item() for k, v in loss_dict.items()}

                pbar_loss = losses_dict["loss"]

                output = pipeline_utils.to_device(output, device="cpu")
                target = pipeline_utils.to_device(target, device="cpu")

                preds = self.model.get_prediction(output, target)

                self.valid_metrics.update_batch(
                    batch_model_outputs=preds,
                    batch_expected_outputs=target,
                    batch_loss=losses_dict,
                )

        return self.valid_metrics.result()
