import torch.nn as nn
from base import BaseModel
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet18_Weights
from pipeline import loss
from pipeline.utils import convert_tf_params_to_bbox
from typing import Tuple


class ResFocusNetwork(BaseModel):
    def __init__(
        self,
        loss_lambda_tr: float,
        loss_lambda_sc: float,
        loss_lambda_rot: float,
        backbone: str,
        threshold: float = 0.5,
        inp_img_size: Tuple[int, int] = (650, 650),
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.loss_lambda_tr = loss_lambda_tr
        self.loss_lambda_sc = loss_lambda_sc
        self.loss_lambda_rot = loss_lambda_rot
        self.inp_img_size = inp_img_size
        assert backbone in ["resnet18", "resnet34"]
        self.model = (
            models.resnet18(weights=ResNet18_Weights.DEFAULT)
            if backbone == "resnet18"
            else models.resnet34(weights=ResNet34_Weights.DEFAULT)
        )

        res_fc_out = self.model.fc.out_features
        self.fc_out = nn.Sequential(
            nn.Linear(in_features=res_fc_out, out_features=2048),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(),
            nn.LeakyReLU(),
        )

        self.cls_fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.tf_fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
        )

        self.tf_tr = nn.Sequential(
            nn.Linear(512, 128), nn.Dropout(), nn.ReLU(), nn.Linear(128, 2)
        )
        self.tf_scale = nn.Sequential(
            nn.Linear(512, 128),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )
        self.tf_rot = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc_out(x)
        out_cls = self.cls_fc(x)
        out_tf = self.tf_fc(x)
        out_translate = self.tf_tr(out_tf)
        out_scale = self.tf_scale(out_tf)
        out_rotate = self.tf_rot(out_tf)

        return out_cls, out_translate, out_scale, out_rotate

    def get_prediction(self, output):
        cls, tf_translate, tf_scale, _ = output
        bbox = convert_tf_params_to_bbox(
            translations=tf_translate,
            scales=tf_scale,
            img_size=self.inp_img_size,
        )
        return {
            "label": (cls >= self.threshold).float().squeeze(),
            "confidence": cls.squeeze(),
            "bbox": bbox,
        }

    def calculate_loss(self, output, target):
        return loss.focus_multiloss(
            output=output,
            target=target,
            lambda_translation=self.loss_lambda_tr,
            lambda_scale=self.loss_lambda_sc,
            lambda_rotation=self.loss_lambda_rot,
        )
