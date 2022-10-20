import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return nn.CrossEntropyLoss()(output, target)


def cross_entropy_loss_weighted(output, target, weights):
    return nn.CrossEntropyLoss(weight=weights)(output, target)


def binary_cross_entropy_loss(output, target):
    return nn.BCELoss()(output, target)
