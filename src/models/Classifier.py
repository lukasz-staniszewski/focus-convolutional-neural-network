import torch.nn as nn
from base import BaseModel
import torch.nn.functional as F
import torch


class Classifier(BaseModel):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=15,
                kernel_size=(3, 3),
                stride=(2, 2),
            ),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=15,
                out_channels=45,
                kernel_size=(3, 3),
                stride=(2, 2),
            ),
            nn.ReLU(),
            nn.BatchNorm2d(45),
            nn.MaxPool2d(kernel_size=3, stride=(1, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=45,
                out_channels=100,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=100,
                out_channels=120,
                kernel_size=(3, 3),
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(120),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=1080, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.final_act(x)

        return x