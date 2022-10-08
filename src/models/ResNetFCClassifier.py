import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models


class ResNetFCClassifier(BaseModel):
    def __init__(self, n_classes=1, threshold=0.5) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.threshold = threshold

        self.model = models.resnet18(pretrained=True)
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
