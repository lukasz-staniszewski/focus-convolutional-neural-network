from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import binary_focal_loss_with_logits
from torch import Tensor


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


def focal_l1smooth_loss(
    output_cls: Tensor,
    output_tf: Tensor,
    target: Dict[str, Tensor],
    alpha: float = 0.25,
    gamma: float = 2.0,
    lambd: float = 10.0,
):
    """Calculates focal loss for classification and l1 smooth for transform params (on positive classes).

    Args:
        output_cls (Tensor): class outputs
        output_tf (Tensor): transform outputs
        target (Tensor): target dictionary
        alpha (float, optional): alpha focal loss param. Defaults to 0.25.
        gamma (float, optional): gamma focal loss param. Defaults to 2.0.
        lambd (float, optional): lambda param - weight of regression loss. Defaults to 10.0.

    Returns:
        Tensor: full loss
    """
    target_cls, target_tf = target["label"], target["transform"]

    cls_loss = binary_focal_loss_with_logits(
        input=output_cls,
        target=target_cls,
        alpha=alpha,
        gamma=gamma,
        reduction="mean",
    )

    positives = target_cls == 1
    reg_loss = nn.SmoothL1Loss()(
        input=output_tf[positives], target=target_tf[positives]
    )

    return cls_loss + lambd * reg_loss


def entropy_l1smooth_loss(
    output_cls: Tensor,
    output_tf: Tensor,
    target: Dict[str, Tensor],
    lambd: float = 10.0,
):
    """Calculates cross entropy loss for classification and l1 smooth for transform params (on positive classes).

    Args:
        output_cls (Tensor): class outputs
        output_tf (Tensor): transform outputs
        target (Tensor): target dictionary
        lambd (float, optional): lambda param - weight of regression loss. Defaults to 10.0.

    Returns:
        Tensor: full loss
    """
    target_cls, target_tf = target["label"], target["transform"]

    cls_loss = F.binary_cross_entropy(
        input=output_cls.squeeze(),
        target=target_cls.float(),
        reduction="mean",
    )

    positives = target_cls == 1
    reg_loss = nn.SmoothL1Loss()(
        input=output_tf[positives], target=target_tf[positives]
    )

    return cls_loss + lambd * reg_loss


def _focus_l1_loss(output: Tensor, target: Tensor, target_cls: Tensor):
    """Modified version of L1 loss for focus model.

    Args:
        output (Tensor): output of the model
        target (Tensor): target
        target_cls (Tensor): target classes, loss will be count only for positive classes

    Returns:
        Tensor: modified version of L1 loss
    """
    loss = (
        F.l1_loss(input=output, target=target, reduction="none").mean(dim=1)
        * target_cls.T
    )
    if target_cls.sum() == 0:
        return loss.sum()
    else:
        return loss.sum() / target_cls.sum()


def focus_multiloss(
    output_cls: Tensor,
    output_translate_x: Tensor,
    output_translate_y: Tensor,
    output_scale: Tensor,
    output_rotate: Tensor,
    target: Dict[str, Tensor],
    lambda_translation: float,
    lambda_scale: float,
    lambda_rotation: float,
):
    """Calculates loss for focus model - weighted sum of classification loss and regression on transform param loss.

    Args:
        output_cls (Tensor): output of the model on classification task
        output_translate (Tensor): output of the model on translation
        output_scale (Tensor): output of the model on scale
        output_rotate (Tensor): output of the model on rotation
        target (Dict[str, Tensor]): targets dictionary
        lambda_translation (float): lambda param - weight of translation loss
        lambda_scale (float): lambda param - weight of scale loss
        lambda_rotation (float): lambda param - weight of rotation loss

    Returns:
        Tensor: full loss
    """
    target_cls, target_tf = target["label"], target["transform"]
    target_translate_x = target_tf[:, 0:1]
    target_translate_y = target_tf[:, 1:2]
    target_scale = target_tf[:, 2:3]
    target_rotate = target_tf[:, 3:4]
    positives = target_cls == 1

    cls_loss = F.binary_cross_entropy_with_logits(
        input=output_cls.squeeze(), target=target_cls.float()
    )
    translation_x_loss = _focus_l1_loss(
        output=output_translate_x,
        target=target_translate_x,
        target_cls=target_cls,
    )
    translation_y_loss = _focus_l1_loss(
        output=output_translate_y,
        target=target_translate_y,
        target_cls=target_cls,
    )
    scale_log_loss = F.l1_loss(
        input=output_scale[positives], target=target_scale[positives]
    )
    rotation_loss = F.l1_loss(
        input=output_rotate[positives], target=target_rotate[positives]
    )

    loss = (
        cls_loss
        + lambda_translation * translation_x_loss
        + lambda_translation * translation_y_loss
        + lambda_scale * scale_log_loss
        + lambda_rotation * rotation_loss
    )

    return {
        "loss": loss,
        "cls_loss": cls_loss,
        "translation_x_loss": translation_x_loss,
        "translation_y_loss": translation_y_loss,
        "scale_log_loss": scale_log_loss,
        "rotation_loss": rotation_loss,
    }


def focus_multiloss_2(
    output_cls: Tensor,
    output_translate_x: Tensor,
    output_translate_y: Tensor,
    output_scale: Tensor,
    output_rotate: Tensor,
    target: Dict[str, Tensor],
    lambda_translation: float,
    lambda_scale: float,
    lambda_rotation: float,
    weights: List[float] = None,
    loss_rot: bool = False,
):
    """Calculates second version of loss for focus model - weighted sum of classification loss and regression on transform param loss.

    Args:
        output_cls (Tensor): output of the model on classification task
        output_translate (Tensor): output of the model on translation
        output_scale (Tensor): output of the model on scale
        output_rotate (Tensor): output of the model on rotation
        target (Dict[str, Tensor]): targets dictionary
        lambda_translation (float): lambda param - weight of translation loss
        lambda_scale (float): lambda param - weight of scale loss
        lambda_rotation (float): lambda param - weight of rotation loss

    Returns:
        Tensor: full loss
    """
    target_cls, target_tf = target["label"], target["transform"]
    target_translate_x = target_tf[:, 0:1]
    target_translate_y = target_tf[:, 1:2]
    target_scale = target_tf[:, 2:3]
    target_rotate = target_tf[:, 3:4]

    cls_loss = F.binary_cross_entropy_with_logits(
        input=output_cls.squeeze(),
        target=target_cls.float(),
        pos_weight=torch.tensor(weights).to(output_cls.device),
    )
    translation_x_loss = _focus_l1_loss(
        output=output_translate_x,
        target=target_translate_x,
        target_cls=target_cls,
    )
    translation_y_loss = _focus_l1_loss(
        output=output_translate_y,
        target=target_translate_y,
        target_cls=target_cls,
    )
    scale_log_loss = _focus_l1_loss(
        output=output_scale, target=target_scale, target_cls=target_cls
    )
    loss = (
        cls_loss
        + lambda_translation * translation_x_loss
        + lambda_translation * translation_y_loss
        + lambda_scale * scale_log_loss
    )

    rotation_loss = torch.nan

    if loss_rot:
        rotation_loss = _focus_l1_loss(
            output=output_rotate, target=target_rotate, target_cls=target_cls
        )
        loss += lambda_rotation * rotation_loss

    return {
        "loss": loss,
        "cls_loss": cls_loss,
        "translation_x_loss": translation_x_loss,
        "translation_y_loss": translation_y_loss,
        "scale_log_loss": scale_log_loss,
        "rotation_loss": rotation_loss,
    }
