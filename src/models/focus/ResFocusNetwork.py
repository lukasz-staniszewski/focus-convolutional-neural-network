import torch.nn as nn
from base import BaseModel
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch
from pipeline import loss


class ResFocusNetwork(BaseModel):
    def __init__(
        self, threshold: float = 0.5, loss_lambda: float = None
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.loss_lambda = loss_lambda

        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        res_fc_out = self.model.fc.out_features
        self.fc_out = nn.Sequential(
            nn.Linear(in_features=res_fc_out, out_features=500),
            nn.Tanh(),
            nn.Linear(in_features=500, out_features=128),
        )

        self.cls_fc = nn.Linear(128, 1)
        self.tf_fc = nn.Linear(128, 4)

    def forward(self, x):
        x = self.model(x)
        x = self.fc_out(x)
        cls = self.cls_fc(x)
        tf = self.tf_fc(x)
        return cls, tf

    def get_prediction(self, output):
        cls, tf_params = output
        return (cls >= self.threshold).float(), tf_params

    def calculate_loss(self, output, target):
        # return loss.focus_multiloss(
        #     output=output, target=target, lambd=self.loss_lambda
        # )
        return loss.focus_multiloss_ce(
            output=output, target=target, lambd=self.loss_lambda
        )
        
