import torch.nn.functional as F
import torch.nn as nn
from kornia.losses import binary_focal_loss_with_logits


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return nn.CrossEntropyLoss()(output, target)


def cross_entropy_loss_weighted(output, target, weights):
    return nn.CrossEntropyLoss(weight=weights)(output, target)


def binary_cross_entropy_loss(output, target):
    target = target.float()
    return nn.BCELoss()(output, target)


def focus_multiloss(output, target, alpha=0.25, gamma=2.0, lambd=10.0):
    # calculate focal loss per classes and l1 smooth for positive classes
    cls_out, reg_out = output
    cls_target, reg_target = target["label"], target["transform"]

    cls_loss = binary_focal_loss_with_logits(
        input=cls_out,
        target=cls_target,
        alpha=alpha,
        gamma=gamma,
        reduction="mean",
    )

    positives = cls_target == 1
    reg_loss = nn.SmoothL1Loss()(
        input=reg_out[positives], target=reg_target[positives]
    )

    return cls_loss + lambd * reg_loss


def focus_multiloss_ce(output, target, lambd=10.0):
    # calculate focal loss per classes and l1 smooth for positive classes
    cls_out, reg_out = output
    cls_target, reg_target = target["label"], target["transform"]

    cls_loss = F.binary_cross_entropy_with_logits(
        input=cls_out.squeeze(),
        target=cls_target.float(),
        reduction="mean",
    )

    positives = cls_target == 1
    reg_loss = nn.SmoothL1Loss()(
        input=reg_out[positives], target=reg_target[positives]
    )

    return cls_loss + lambd * reg_loss
