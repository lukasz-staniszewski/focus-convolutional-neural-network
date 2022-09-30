import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models


class ResNetClassifier(BaseModel):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        res_fc_out = self.model.fc.out_features
        self.fc_out = nn.Linear(in_features=res_fc_out, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc_out(x)
        x = F.softmax(x, 1)
        return x
