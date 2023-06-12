import torch
import torch.nn as nn

from base import BaseModel
from models import models_utils
from pipeline import loss


class MultiClassifier(BaseModel):
    def __init__(
        self, n_classes=1, backbone="resnet34", class_weights=None
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.class_weights = class_weights
        if self.class_weights is not None:
            assert len(self.class_weights) == self.n_classes, (
                "If passed, number of class weights must match number of"
                " classes."
            )

        self.model = models_utils.assign_backbone(backbone)
        res_fc_out = self.model.fc.out_features

        self.fc_out = nn.Sequential(
            nn.Linear(in_features=res_fc_out, out_features=500),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(in_features=500, out_features=50),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=50, out_features=self.n_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc_out(x)
        return x

    def get_prediction(self, output):
        _, prediction = torch.max(output, 1)
        return prediction

    def calculate_loss(self, output, target):
        if self.class_weights is not None:
            return loss.cross_entropy_loss_weighted(
                output=output, target=target, weights=self.class_weights
            )
        else:
            return loss.cross_entropy_loss(output=output, target=target)
