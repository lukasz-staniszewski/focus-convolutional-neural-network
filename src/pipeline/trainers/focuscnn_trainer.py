import os

import torch
from tqdm import tqdm

from base import BaseTrainer
from pipeline import pipeline_utils
from utils import secure_load_path


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
        self.valid_metrics.reset()

        progress_bar = tqdm(
            enumerate(self.train_data_loader),
            desc=f"Training epoch {epoch}",
            colour="blue",
            total=len(self.train_data_loader),
        )
        pbar_loss = "None"

        for batch_idx, data in progress_bar:
            progress_bar.set_postfix({"loss": pbar_loss})
            data_in = pipeline_utils.to_device(data[0], device=self.device)
            target = pipeline_utils.to_device(
                data[2],
                device=self.device,
            )

            output = self.model(data_in, target)
            loss = output["loss"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar_loss = loss.item()

            output = pipeline_utils.to_device(output, device="cpu")
            preds = self.model.get_predictions(output=output, img_ids=data[1])

            predictions = {}
            targets = {}
            predictions["cls_focuscnn"] = preds["cls_predictions"]
            predictions["map_focuscnn"] = preds["map_predictions"]
            targets["cls_focuscnn"] = output["target_cls"]
            targets["map_focuscnn"] = self.model.prepare_target_for_map(
                target_bboxes=data[3], target_image_ids=data[1]
            )

            self.train_metrics.update_batch(
                batch_model_outputs=predictions,
                batch_expected_outputs=targets,
                batch_loss=pbar_loss,
            )

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
        self.train_metrics.reset()
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

                data_in = pipeline_utils.to_device(data[0], device=self.device)
                target = pipeline_utils.to_device(
                    data[2],
                    device=self.device,
                )

                output = self.model(data_in, target)
                loss = output["loss"]
                pbar_loss = loss.item()

                output = pipeline_utils.to_device(output, device="cpu")
                preds = self.model.get_predictions(
                    output=output, img_ids=data[1]
                )

                predictions = {}
                targets = {}
                predictions["cls_focuscnn"] = preds["cls_predictions"]
                predictions["map_focuscnn"] = preds["map_predictions"]
                targets["cls_focuscnn"] = output["target_cls"]
                targets["map_focuscnn"] = self.model.prepare_target_for_map(
                    target_bboxes=data[3], target_image_ids=data[1]
                )

                self.valid_metrics.update_batch(
                    batch_model_outputs=predictions,
                    batch_expected_outputs=targets,
                    batch_loss=pbar_loss,
                )

        return self.valid_metrics.result()

    def _save_checkpoint(self, epoch: int, save_as_best: bool = False) -> None:
        """By default saves checkpoint of path. If save_as_best is True, saves checkpoint as best.
        Args:
            epoch (int): current epoch number
            save_as_best (bool): if True, saves checkpoint as best; defaults to False
        """
        # classifier
        classifier_arch = type(self.model.classifier_model).__name__
        classifier_state_dict = self.model.classifier_model.state_dict()

        focus_state = {
            fm_id: {
                "arch": type(fm_model).__name__,
                "state_dict": fm_model.state_dict(),
            }
            for fm_id, fm_model in self.model.focus_models.items()
        }

        state = {
            "classifier": {
                "arch": classifier_arch,
                "state_dict": classifier_state_dict,
            },
            "focus_models": focus_state,
            "epoch": epoch,
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

        if (
            checkpoint["classifier"]["arch"]
            != self.config["classifier"]["arch"]
        ):
            self.logger.warning(
                "Warning: Architecture config given in file differs"
                " from the one of checkpoint. This may cause errors"
                " while state_dict is being loaded if differences"
                " are not caused by loss parameters."
            )
        # load architecture params from checkpoint.
        self.model.classifier_model.load_state_dict(
            checkpoint["classifier"]["state_dict"]
        )

        for focus_model_id in checkpoint["focus_models"].keys():
            if (
                checkpoint["focus_models"][focus_model_id]["arch"]
                != self.config["focus_models"][focus_model_id]["arch"]
            ):
                self.logger.warning(
                    "Warning: Architecture config given in file differs"
                    " from the one of checkpoint. This may cause errors"
                    " while state_dict is being loaded if differences"
                    " are not caused by loss parameters."
                )
            self.model.focus_models[focus_model_id].load_state_dict(
                checkpoint["focus_models"][focus_model_id]["state_dict"]
            )

        if (
            checkpoint["optimizer"]["type"]
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
