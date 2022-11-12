import torch.nn as nn
from base import BaseModel
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch


class ResNetFCClassifier(BaseModel):
    def __init__(self, n_classes=1) -> None:
        super().__init__()
        self.n_classes = n_classes

        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
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
