import torch.nn as nn
from base import BaseModel
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet18_Weights
from pipeline import loss
import torch


class ResFocusNetwork(BaseModel):
    def __init__(
        self, threshold: float = 0.5, loss_lambda: float = None
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.loss_lambda = loss_lambda

        self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        res_fc_out = self.model.fc.out_features
        self.fc_out = nn.Sequential(
            nn.Linear(in_features=res_fc_out, out_features=2048),
            nn.LeakyReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.LeakyReLU(),
        )

        self.cls_fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.tf_fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc_out(x)
        cls = self.cls_fc(x)
        tf = self.tf_fc(x)
        return cls, tf

    def get_prediction(self, output):
        cls, tf_params = output
        return (cls >= self.threshold).float().squeeze(), tf_params

    def calculate_loss(self, output, target):
        losses = {}
        focal_loss = loss.focus_multiloss_ce(
            output=output, target=target, lambd=self.loss_lambda
        )
        losses["loss"] = focal_loss
        with torch.no_grad():
            cls_out, reg_out = output
            cls_target, reg_target = target["label"], target["transform"]
            positives = cls_target == 1
            losses["translation_loss"] = nn.SmoothL1Loss()(
                input=reg_out[:, 0:2][positives],
                target=reg_target[:, 0:2][positives],
            )
            losses["scale_log_loss"] = nn.SmoothL1Loss()(
                input=reg_out[:, 2:3][positives],
                target=reg_target[:, 2:3][positives],
            )
            losses["rotation_loss"] = nn.SmoothL1Loss()(
                input=reg_out[:, 3:4][positives],
                target=reg_target[:, 3:4][positives],
            )
        return losses
