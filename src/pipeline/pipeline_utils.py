from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import torch
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
    targets, predictions = move_tensors_to_device(
        targets, device="cpu"
    ), move_tensors_to_device(predictions, device="cpu")

    TP, FP, FN, TN = get_class_cm(output=predictions, target=targets)

    idx2cls = (
        (lambda x: cls_map[str(x)])
        if cls_map is not None
        else (lambda x: str(x))
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
            logger.info(
                "{"
                f"class: {idx2cls(cls_idx)}, "
                f"accuracy: {acc:.4f}, "
                f"recall: {rec:.4f}, "
                f"precision: {prec:.4f}, "
                f"f1-score: {f1:.4f}"
                "}"
            )
    console.print(table)


def convert_tf_params_to_bbox(
    translations: torch.Tensor, scales: torch.Tensor, img_size: Tuple[int, int]
) -> torch.Tensor:
    """Converts translation and scale parameters to bounding boxes."""
    bbox_width = bbox_height = torch.exp(scales) * img_size[0]
    bbox_center_x = 0.5 * img_size[0] * translations[:, 0:1] + img_size[0] / 2
    bbox_center_y = 0.5 * img_size[1] * translations[:, 1:2] + img_size[1] / 2
    bbox_xmin = bbox_center_x - bbox_width / 2
    bbox_ymin = bbox_center_y - bbox_height / 2
    bbox_xmax = bbox_center_x + bbox_width / 2
    bbox_ymax = bbox_center_y + bbox_height / 2
    return torch.cat([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax], dim=1)


def get_class_cm(output, target):
    N_CLASSES = int(torch.max(target).item() + 1)
    TP = [0 for _ in range(N_CLASSES)]
    FP = [0 for _ in range(N_CLASSES)]
    TN = [0 for _ in range(N_CLASSES)]
    FN = [0 for _ in range(N_CLASSES)]
    with torch.no_grad():
        pred = output
        assert pred.shape[0] == len(target)
        for c in range(N_CLASSES):
            TP[c] = torch.sum((target == pred) * (pred == c))
            FP[c] = torch.sum((target != pred) * (pred == c))
            FN[c] = torch.sum((target != pred) * (target == c))
            TN[c] = torch.sum((target != c) * (pred != c))
    return TP, FP, FN, TN


def move_tensors_to_device(tensors, device, dict_columns=None):
    # is just tensor
    if torch.is_tensor(tensors):
        return tensors.to(device)
    # is tuple
    elif isinstance(tensors, tuple):
        return tuple(
            move_tensors_to_device(
                tensors=tens, device=device, dict_columns=dict_columns
            )
            for tens in tensors
        )
    # is dict
    elif isinstance(tensors, dict):
        # is dict contains tensors as values
        if torch.is_tensor(list(tensors.values())[0]) and dict_columns:
            return {
                k: move_tensors_to_device(
                    tensors=v, device=device, dict_columns=dict_columns
                )
                if k in dict_columns
                else v
                for (k, v) in tensors.items()
            }
        # is dict contains not tensors as values
        else:
            return {
                k: move_tensors_to_device(
                    tensors=v, device=device, dict_columns=dict_columns
                )
                for (k, v) in tensors.items()
            }
    else:
        return tensors
