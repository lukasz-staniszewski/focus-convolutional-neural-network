from torchvision import models
from torchvision.models import (
    ResNet34_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)

BACKBONES = {
    "resnet18": (models.resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, ResNet50_Weights.DEFAULT),
}


def assign_backbone(backbone_name):
    assert (
        backbone_name in BACKBONES.keys()
    ), f"Backbone {backbone_name} not supported"
    model, model_weights = BACKBONES[backbone_name]
    return model(weights=model_weights)
