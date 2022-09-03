import torch
import json


def _get_class_cm(output, target):
    with open("config.json", "r") as f:
        N_CLASSES = json.load(f)["arch"]["args"]["num_classes"]
    TPR = [0 for _ in range(N_CLASSES)]
    FPR = [0 for _ in range(N_CLASSES)]
    TNR = [0 for _ in range(N_CLASSES)]
    FNR = [0 for _ in range(N_CLASSES)]
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        for c in range(N_CLASSES):
            TPR[c] = torch.sum((target == pred) * (pred == c))
            FPR[c] = torch.sum((target != pred) * (pred == c))
            FNR[c] = torch.sum((target != pred) * (target == c))
            TNR[c] = torch.sum((target != c) * (pred != c))
    return TPR, FPR, FNR, TNR


def micro_accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def macro_accuracy(output, target):
    TPR, FPR, FNR, TNR = _get_class_cm(output, target)
    ACC = [0 for _ in range(len(TPR))]
    for c in range(len(TPR)):
        ACC[c] = (TPR[c] + TNR[c]) / (TPR[c] + FPR[c] + TNR[c] + FNR[c])
    return torch.Tensor(ACC).mean().item()


def micro_recall(output, target):
    TPR, FPR, FNR, TNR = _get_class_cm(output, target)
    TPR = sum(TPR)
    FPR = sum(FPR)
    FNR = sum(FNR)
    TNR = sum(TNR)
    REC = TPR / max(TPR + FNR, 1)
    return REC.item()


def macro_recall(output, target):
    TPR, FPR, FNR, TNR = _get_class_cm(output, target)
    REC = [0 for _ in range(len(TPR))]
    for c in range(len(TPR)):
        REC[c] = TPR[c] / max(TPR[c] + FNR[c], 1)
    return torch.Tensor(REC).mean().item()


def micro_precision(output, target):
    TPR, FPR, FNR, TNR = _get_class_cm(output, target)
    TPR = sum(TPR)
    FPR = sum(FPR)
    FNR = sum(FNR)
    TNR = sum(TNR)
    PREC = TPR / max(TPR + FPR, 1)
    return PREC.item()


def macro_precision(output, target):
    TPR, FPR, FNR, TNR = _get_class_cm(output, target)
    PREC = [0 for _ in range(len(TPR))]
    for c in range(len(TPR)):
        PREC[c] = TPR[c] / max(TPR[c] + FPR[c], 1)
    return torch.Tensor(PREC).mean().item()


def micro_f1(output, target):
    TPR, FPR, FNR, TNR = _get_class_cm(output, target)
    TPR = sum(TPR)
    FPR = sum(FPR)
    FNR = sum(FNR)
    TNR = sum(TNR)
    REC = TPR / max(TPR + FNR, 1)
    PREC = TPR / max(TPR + FPR, 1)
    F1 = 2 * REC * PREC / max(REC + PREC, 1e-6)
    return F1.item()


def macro_f1(output, target):
    TPR, FPR, FNR, TNR = _get_class_cm(output, target)
    PREC = [0 for _ in range(len(TPR))]
    REC = [0 for _ in range(len(TPR))]
    F1 = [0 for _ in range(len(TPR))]
    for c in range(len(TPR)):
        REC[c] = TPR[c] / max(TPR[c] + FNR[c], 1)
        PREC[c] = TPR[c] / max(TPR[c] + FPR[c], 1)
        F1[c] = 2 * REC[c] * PREC[c] / max(REC[c] + PREC[c], 1)
    return torch.Tensor(F1).mean().item()


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
