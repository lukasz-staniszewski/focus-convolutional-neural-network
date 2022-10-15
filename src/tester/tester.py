from pathlib import Path
import pandas as pd
import torch
from base import BaseTester, BaseDataLoader
from tqdm import tqdm

from utils import MetricTracker


class Tester(BaseTester):
    """Tester class."""

    def __init__(
        self,
        model: torch.nn.Module,
        metric_ftns: list,
        config: dict,
        device: torch.device,
        test_data_loader: BaseDataLoader,
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
        self.test_data_loader = test_data_loader

        self.test_metrics = MetricTracker(
            *[m.__name__ for m in self.metric_ftns],
            writer=None,
        )

        if not only_predict:
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
                data = data.to(self.device)
                output = self.model(data).squeeze()
                output = (output >= self.model.threshold).float()
                predictions.append(output.cpu())

        predictions = torch.cat(predictions).numpy()
        self.test_data_loader.to_csv(
            csv_path=self.predictions_path, predictions=predictions
        )
        self.logger.info(
            f"Predictions saved to {self.predictions_path}"
        )

    def _calculate_metrics(self) -> None:
        """Metrics calculation logic."""
        targets = self.test_data_loader.get_targets()
        df_preds = pd.read_csv(self.predictions_path)
        assert (
            "label" in df_preds.columns
        ), "Predictions file does not contain label column!"

        for met in self.metric_ftns:
            self.test_metrics.update(
                met.__name__,
                met(
                    torch.Tensor(df_preds["label"]).to(self.device),
                    torch.Tensor(targets).to(self.device),
                ),
            )

        self.logger.info(
            f"Test metrics summary:\n{self.test_metrics.result()}"
        )