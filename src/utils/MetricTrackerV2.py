from typing import Any, Callable, Dict, List

import torch
from tqdm import tqdm


class MetricTrackerV2:
    """Class that keeps track of metrics during training and validation along with mean loss."""

    def __init__(self, metrics_handlers: List[Callable[[Any, Any], float]]):
        """Metric tracker constructor.

        Args:
            metrics_handlers (List[Callable[[Any, Any], float]]): list of metrics callables.
        """
        self.model_outputs = []
        self.expected_outputs = []
        self.model_losses = []
        self.metrics_handlers = metrics_handlers

    def _concatenate_outputs(self) -> None:
        self.model_outputs = torch.cat(self.model_outputs)
        self.expected_outputs = torch.cat(self.expected_outputs)

    def get_model_outputs(self) -> List[Any]:
        return self.model_outputs

    def get_expected_outputs(self) -> List[Any]:
        return self.expected_outputs

    def reset(self) -> None:
        """Resets all outputs."""
        self.model_outputs = []
        self.expected_outputs = []
        self.model_losses = []

    def update_batch(
        self,
        batch_model_outputs: List[Any],
        batch_expected_outputs: List[Any],
        batch_loss: float = None,
    ) -> None:
        """Updates both model outputs and expected outputs for epoch per batch."""
        self.model_outputs.append(batch_model_outputs)
        self.expected_outputs.append(batch_expected_outputs)
        if batch_loss is not None:
            self.model_losses.append(batch_loss)

    def result(self) -> Dict:
        """Returns the average value of all metrics.

        Returns:
            Dict: dictionary with metric names and values
        """
        if isinstance(self.model_outputs, List):
            self._concatenate_outputs()
        self._data = {}
        if len(self.model_losses) > 0:
            self._data["loss"] = sum(self.model_losses) / len(self.model_losses)
        for metric in tqdm(self.metrics_handlers):
            self._data[metric.__name__] = metric(
                output=self.model_outputs, target=self.expected_outputs
            )
        return self._data
