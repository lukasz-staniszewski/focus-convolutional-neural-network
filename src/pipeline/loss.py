import torch.nn.functional as F
import torch.nn as nn
import torch
from kornia.losses import binary_focal_loss_with_logits


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return nn.CrossEntropyLoss()(output, target)


def cross_entropy_loss_weighted(output, target, weights):
    weights = torch.Tensor(weights).to(output.get_device())
    return nn.CrossEntropyLoss(weight=weights)(output, target)


def binary_cross_entropy_loss(output, target):
    target = target.float()
    return nn.BCELoss()(output, target)


def smooth_loss(output, target):
    return nn.SmoothL1Loss()(input=output, target=target)


def focal_l1smooth_loss(output, target, alpha=0.25, gamma=2.0, lambd=10.0):
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


def ce_l1smooth_loss(output, target, lambd=10.0):
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


def focus_multiloss(
    output, target, lambda_translation, lambda_scale, lambda_rotation
):
    cls_out, out_translate, out_scale, out_rotate = output
    cls_target, reg_target = target["label"], target["transform"]
    positives = cls_target == 1

    cls_loss = F.binary_cross_entropy(
        input=cls_out.squeeze(), target=cls_target.float()
    )
    translation_loss = nn.L1Loss()(
        input=out_translate[positives], target=reg_target[:, 0:2][positives]
    )
    scale_log_loss = nn.L1Loss()(
        input=out_scale[positives], target=reg_target[:, 2:3][positives]
    )
    rotation_loss = nn.L1Loss()(
        input=out_rotate[positives], target=reg_target[:, 3:4][positives]
    )
    loss = (
        cls_loss
        + lambda_translation * translation_loss
        + lambda_scale * scale_log_loss
        + lambda_rotation * rotation_loss
    )

    return {
        "loss": loss,
        "cls_loss": cls_loss,
        "translation_loss": translation_loss,
        "scale_log_loss": scale_log_loss,
        "rotation_loss": rotation_loss,
    }
