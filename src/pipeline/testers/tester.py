from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from base import BaseDataLoader, BaseTester
from pipeline import pipeline_utils


class Tester(BaseTester):
    """Tester class."""

    def __init__(
        self,
        model: torch.nn.Module,
        metric_ftns: list,
        config: dict,
        device: torch.device,
        data_loader: BaseDataLoader,
        only_predict: bool = False,
    ) -> None:
        """Tester constructor.

        Args:
            model (torch.nn.Module): model to train
            metric_ftns (list): metrics to compute
            config (dict): config dictionary
            device (torch.device): device to use
            test_data_loader (BaseDataLoader): data loader for training
            only_predict (bool, optional): if True, then only predictions without metrics calculation will be performed. Defaults to False.
        """
        super().__init__(
            model=model,
            metric_ftns=metric_ftns,
            config=config,
            only_predict=only_predict,
        )
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.test_data_loader = self.data_loader.get_test_loader()

        self.predictions_path = (
            Path(self.config.save_cfg_dir) / "predictions.csv"
        )

    def _predict(self) -> None:
        """Prediction logic."""
        self.model.eval()

        progress_bar = tqdm(
            self.test_data_loader,
            desc="Predicting",
            colour="green",
            total=len(self.test_data_loader),
        )

        predictions = []

        with torch.no_grad():
            for data in progress_bar:
                data = pipeline_utils.to_device(data, device=self.device)
                output = self.model(data).squeeze()
                output = self.model.get_prediction(output)
                predictions.append(
                    pipeline_utils.to_device(output, device="cpu")
                )

        predictions = torch.cat(predictions).numpy()
        self.data_loader.to_csv(
            csv_path=self.predictions_path, predictions=predictions
        )
        self.logger.info(f"Predictions saved to {self.predictions_path}")

    def _calculate_metrics(self) -> None:
        """Metrics calculation logic."""
        targets = self.data_loader.get_targets()
        targets = torch.Tensor(targets)
        targets = pipeline_utils.to_device(targets, device="cpu")

        df_preds = pd.read_csv(self.predictions_path)
        assert (
            "label" in df_preds.columns
        ), "Predictions file does not contain label column!"
        preds = torch.Tensor(df_preds["label"])
        preds = pipeline_utils.to_device(preds, device="cpu")

        self.test_metrics.update_batch(
            batch_model_outputs=preds,
            batch_expected_outputs=targets,
        )

        self.logger.info(
            f"Common metrics summary:\n{self.test_metrics.result()}"
        )

        if self.data_loader.is_multilabel:
            pipeline_utils.print_per_class_metrics(
                targets=targets,
                predictions=preds,
                cls_map=self.data_loader.labels,
                logger=self.logger,
            )
