from typing import Dict
import pandas as pd
from utils.logger import TensorboardWriter


class MetricTracker:
    """Class that keeps track of metrics during training and validation."""

    def __init__(self, *keys, writer: TensorboardWriter = None):
        """Metric tracker constructor.

        Args:
            writer (Callable, optional): tensorboard writer function. Defaults to None.
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self) -> None:
        """Resets all metrics."""
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key: str, value: int, n: int = 1):
        """Updates the metric.

        Args:
            key (int): metric name
            value (int): metric value
            n (int, optional): number of samples. Defaults to 1.
        """
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key: str) -> float:
        """Returns the average value of the metric.

        Args:
            key (str): metric name

        Returns:
            float: metric value
        """
        return self._data.average[key]

    def result(self) -> Dict:
        """Returns the average value of all metrics.

        Returns:
            Dict: dictionary with metric names and values
        """
        return dict(self._data.average)
