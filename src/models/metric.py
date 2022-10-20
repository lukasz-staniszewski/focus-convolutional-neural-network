import torch
import json


def _get_class_cm(output, target):
    N_CLASSES = torch.max(target).item() + 1
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


def micro_accuracy(output, target):
    TP, FP, FN, TN = _get_class_cm(output, target)
    TP = sum(TP)
    FP = sum(FP)
    FN = sum(FN)
    TN = sum(TN)
    ACC = (TP + TN) / (TP + FP + TN + FN)
    return ACC.item()


def macro_accuracy(output, target):
    TP, FP, FN, TN = _get_class_cm(output, target)
    accuracies = [0 for _ in range(len(TP))]
    for c in range(len(accuracies)):
        accuracies[c] = (TP[c] + TN[c]) / (
            TP[c] + FP[c] + TN[c] + FN[c]
        )
    return torch.Tensor(accuracies).mean().item()


def micro_recall(output, target):
    TP, _, FN, _ = _get_class_cm(output, target)
    TP = sum(TP)
    FN = sum(FN)
    return 0.0 if (TP + FN) == 0 else (TP / (TP + FN)).item()


def macro_recall(output, target):
    TP, _, FN, _ = _get_class_cm(output, target)
    recalls = [0 for _ in range(len(TP))]
    for c in range(len(recalls)):
        recalls[c] = (
            torch.Tensor([0.0])
            if (TP[c] + FN[c]).item() == 0
            else (TP[c] / (TP[c] + FN[c]))
        )
    return torch.Tensor(recalls).mean().item()


def micro_precision(output, target):
    TP, FP, _, _ = _get_class_cm(output, target)
    TP = sum(TP)
    FP = sum(FP)
    return 0.0 if (TP + FP) == 0 else (TP / (TP + FP)).item()


def macro_precision(output, target):
    TP, FP, _, _ = _get_class_cm(output, target)
    precisions = [0 for _ in range(len(TP))]
    for c in range(len(TP)):
        precisions[c] = (
            torch.Tensor([0.0])
            if (TP[c] + FP[c]).item() == 0
            else (TP[c] / (TP[c] + FP[c]))
        )
    return torch.Tensor(precisions).mean().item()


def micro_f1(output, target):
    precision = micro_precision(output, target)
    recall = micro_recall(output, target)
    return (
        0.0
        if (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )


def macro_f1(output, target):
    TP, FP, FN, TN = _get_class_cm(output, target)
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
