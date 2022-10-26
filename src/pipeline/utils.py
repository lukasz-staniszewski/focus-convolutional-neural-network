from pipeline.metrics import _get_class_cm
import torch
from typing import Union, Optional, Dict, Any
import pandas as pd
from rich.console import Console
from rich.table import Table


def print_per_class_metrics(
    targets: Union[pd.Series, torch.Tensor],
    predictions: Union[pd.Series, torch.Tensor],
    cls_map: Optional[Dict[int, str]] = None,
    logger: Optional[Any] = None,
) -> None:
    """Method prints per-class metrics in tabular view."""
    if isinstance(targets, pd.Series):
        targets = torch.Tensor(targets)
    if isinstance(predictions, pd.Series):
        predictions = torch.Tensor(predictions)
    targets, predictions = targets.cpu(), predictions.cpu()

    TP, FP, FN, TN = _get_class_cm(output=predictions, target=targets)
    idx2cls = (
        lambda x: cls_map[str(x)] if cls_map is not None else str(x)
    )

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Class", style="dim", width=10)
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-score", justify="right")

    for cls_idx, (tp, fp, fn, tn) in enumerate(zip(TP, FP, FN, TN)):
        acc = (tp + tn) / (tp + fp + fn + tn)
        rec = tp / (tp + fn)
        prec = tp / (tp + fp)
        f1 = 2 * (prec * rec) / (prec + rec)
        table.add_row(
            idx2cls(cls_idx),
            f"{acc:.3f}",
            f"{prec:.3f}",
            f"{rec:.3f}",
            f"{f1:.3f}",
        )

        if logger is not None:
            logger.debug(
                f"Class {idx2cls(cls_idx)} metrics summary:\n"
                f"Accuracy: {acc:.4f}\n"
                f"Recall: {rec:.4f}\n"
                f"Precision: {prec:.4f}\n"
                f"F1: {f1:.4f}"
            )
    console.print(table)
