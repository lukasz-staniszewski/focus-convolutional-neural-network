import torch
from pipeline.utils import get_class_cm
from torchvision.ops import box_iou
from typing import List


def micro_accuracy(output, target):
    TP, FP, FN, TN = get_class_cm(output, target)
    TP = sum(TP)
    FP = sum(FP)
    FN = sum(FN)
    TN = sum(TN)
    ACC = (TP + TN) / (TP + FP + TN + FN)
    return ACC.item()


def macro_accuracy(output, target):
    TP, FP, FN, TN = get_class_cm(output, target)
    accuracies = [0 for _ in range(len(TP))]
    for c in range(len(accuracies)):
        accuracies[c] = (TP[c] + TN[c]) / (TP[c] + FP[c] + TN[c] + FN[c])
    return torch.Tensor(accuracies).mean().item()


def micro_recall(output, target):
    TP, _, FN, _ = get_class_cm(output=output, target=target)
    TP = sum(TP)
    FN = sum(FN)
    return 0.0 if (TP + FN) == 0 else (TP / (TP + FN)).item()


def macro_recall(output, target):
    TP, _, FN, _ = get_class_cm(output=output, target=target)
    recalls = [0 for _ in range(len(TP))]
    for c in range(len(recalls)):
        recalls[c] = (
            torch.Tensor([0.0])
            if (TP[c] + FN[c]).item() == 0
            else (TP[c] / (TP[c] + FN[c]))
        )
    return torch.Tensor(recalls).mean().item()


def micro_precision(output, target):
    TP, FP, _, _ = get_class_cm(output=output, target=target)
    TP = sum(TP)
    FP = sum(FP)
    return 0.0 if (TP + FP) == 0 else (TP / (TP + FP)).item()


def macro_precision(output, target):
    TP, FP, _, _ = get_class_cm(output=output, target=target)
    precisions = [0 for _ in range(len(TP))]
    for c in range(len(TP)):
        precisions[c] = (
            torch.Tensor([0.0])
            if (TP[c] + FP[c]).item() == 0
            else (TP[c] / (TP[c] + FP[c]))
        )
    return torch.Tensor(precisions).mean().item()


def micro_f1(output, target):
    precision = micro_precision(output=output, target=target)
    recall = micro_recall(output=output, target=target)
    return (
        0.0
        if (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )


def macro_f1(output, target):
    TP, FP, FN, TN = get_class_cm(output=output, target=target)
    f1s = [0 for _ in range(len(TP))]
    for c in range(len(f1s)):
        recall = (
            torch.Tensor([0.0])
            if (TP[c] + FN[c]).item() == 0
            else (TP[c] / (TP[c] + FN[c]))
        )
        precision = (
            torch.Tensor([0.0])
            if (TP[c] + FP[c]).item() == 0
            else (TP[c] / (TP[c] + FP[c]))
        )
        f1s[c] = (
            torch.Tensor([0.0])
            if (precision + recall).item() == 0
            else 2 * precision * recall / (precision + recall)
        )
    return torch.Tensor(f1s).mean().item()


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def true_positive(output, target):
    with torch.no_grad():
        tp = torch.sum(output * target).item()
    return tp


def true_negative(output, target):
    with torch.no_grad():
        tn = torch.sum((1 - output) * (1 - target)).item()
    return tn


def false_positive(output, target):
    with torch.no_grad():
        fp = torch.sum(output * (1 - target)).item()
    return fp


def false_negative(output, target):
    with torch.no_grad():
        fn = torch.sum((1 - output) * target).item()
    return fn


def accuracy(output, target):
    tp = true_positive(output, target)
    tn = true_negative(output, target)
    fp = false_positive(output, target)
    fn = false_negative(output, target)
    return (tp + tn) / (tp + tn + fp + fn)


def recall(output, target):
    tp = true_positive(output, target)
    fn = false_negative(output, target)
    if tp + fn == 0:
        return 0.0
    return (tp) / (tp + fn)


def precision(output, target):
    tp = true_positive(output, target)
    fp = false_positive(output, target)
    if tp + fp == 0:
        return 0.0
    return (tp) / (tp + fp)


def f1(output, target):
    prec = precision(output, target)
    rec = recall(output, target)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# focus classification metrics
def focus_accuracy(output, target):
    target = target["label"]
    output = output["label"]
    return accuracy(output, target)


def focus_recall(output, target):
    target = target["label"]
    output = output["label"]
    return recall(output, target)


def focus_precision(output, target):
    target = target["label"]
    output = output["label"]
    return precision(output, target)


def focus_f1(output, target):
    target = target["label"]
    output = output["label"]
    return f1(output, target)


def _get_binary_ious(output, target) -> List[torch.Tensor]:
    target_cls = target["label"]
    out_cls = output["label"]
    positives = torch.logical_and(target_cls == 1, out_cls == 1)
    out_bbox = output["bbox"][positives]
    target_bbox = target["bbox"][positives]

    ious = []
    for i in range(positives.sum().item()):
        ious.append(
            box_iou(
                out_bbox[i].unsqueeze(0),
                target_bbox[i].unsqueeze(0),
            ).squeeze(0)
        )
    return ious


def mean_iou(output, target):
    ious = _get_binary_ious(output, target)

    if len(ious) == 0:
        return 0.0
    else:
        return torch.cat(ious).mean().item()


def iou50_accuracy(output, target):
    ious = _get_binary_ious(output, target)

    if len(ious) == 0:
        return 0.0
    else:
        return (torch.cat(ious) > 0.5).float().mean().item()
