from typing import Any, Callable, Dict, List, Union

import torch
from tqdm import tqdm


class MetricTrackerV2:
    """Class that keeps track of metrics during training and validation along with mean of losses."""

    def __init__(self, metrics_handlers: List[Callable[[Any, Any], float]]):
        """Metric tracker constructor.

        Args:
            metrics_handlers (List[Callable[[Any, Any], float]]): list of metrics callables.
        """
        self.model_outputs = []
        self.expected_outputs = []
        self.metrics_handlers = metrics_handlers
        self.model_losses = None

    def _concatenate_outputs(self) -> None:
        assert (
            len(self.model_outputs) == len(self.expected_outputs)
            and len(self.model_outputs) > 0
        )
        if torch.is_tensor(self.model_outputs[0]):
            self.model_outputs = torch.cat(self.model_outputs)
            self.expected_outputs = torch.cat(self.expected_outputs)
        elif isinstance(self.model_outputs[0], dict):
            self.new_model_outputs = {}
            self.new_expected_outputs = {}
            for output_name in self.model_outputs[0].keys():
                if torch.is_tensor(self.model_outputs[0][output_name]):
                    self.new_model_outputs[output_name] = torch.cat(
                        [output[output_name] for output in self.model_outputs]
                    )
                    self.new_expected_outputs[output_name] = torch.cat(
                        [output[output_name] for output in self.expected_outputs]
                    )
                else:
                    self.new_model_outputs[output_name] = [
                        item for output in self.model_outputs for item in output[output_name]
                    ]
                    self.new_expected_outputs[output_name] = [
                        item for output in self.expected_outputs for item in output[output_name]
                    ]
            self.model_outputs = self.new_model_outputs
            self.expected_outputs = self.new_expected_outputs

    def get_model_outputs(self) -> List[Any]:
        return self.model_outputs

    def get_expected_outputs(self) -> List[Any]:
        return self.expected_outputs

    def reset(self) -> None:
        """Resets all outputs."""
        self.model_outputs = []
        self.expected_outputs = []
        self.model_losses = None

    def update_batch(
        self,
        batch_model_outputs: List[Any],
        batch_expected_outputs: List[Any],
        batch_loss: Union[float, Dict[str, float]] = None,
    ) -> None:
        """Updates both model outputs and expected outputs for epoch per batch."""
        self.model_outputs.append(batch_model_outputs)
        self.expected_outputs.append(batch_expected_outputs)
        if batch_loss is not None:
            if isinstance(batch_loss, Dict):
                if self.model_losses is None:
                    self.model_losses = {k: [] for k in batch_loss.keys()}
                for k, v in batch_loss.items():
                    self.model_losses[k].append(v)
            else:
                if self.model_losses is None:
                    self.model_losses = []
                self.model_losses.append(batch_loss)

    def result(self) -> Dict:
        """Returns the average value of all metrics.

        Returns:
            Dict: dictionary with metric names and values
        """
        if isinstance(self.model_outputs, List):
            self._concatenate_outputs()
        self._data = {}
        if self.model_losses is not None:
            if isinstance(self.model_losses, Dict):
                for k, v in self.model_losses.items():
                    self._data[k] = sum(v) / len(v)
            else:
                self._data["loss"] = sum(self.model_losses) / len(
                    self.model_losses
                )
        for metric in tqdm(self.metrics_handlers):
            self._data[metric.__name__] = metric(
                output=self.model_outputs, target=self.expected_outputs
            )
        return self._data
