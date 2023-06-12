import torch
import torch.nn as nn

from base import BaseModel
from pipeline import loss


class Classifier(BaseModel):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=15,
                kernel_size=(3, 3),
                stride=(2, 2),
            ),
            nn.ReLU(),
            nn.Dropout2d(0.1),
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
            nn.Dropout2d(0.1),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=100,
                out_channels=120,
                kernel_size=(2, 2),
                stride=(1, 1),
            ),
            nn.ReLU(),
            nn.Dropout2d(0.15),
            nn.BatchNorm2d(120),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=1080, out_features=650),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=650, out_features=300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=300, out_features=50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=50, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)

        return x

    def get_prediction(self, output):
        return (output >= self.threshold).float()

    def calculate_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return loss.binary_cross_entropy_loss(output=output, target=target)
