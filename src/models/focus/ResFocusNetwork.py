from typing import List, Tuple

import torch
import torch.nn as nn

from base import BaseModel
from models.models_utils import assign_backbone
from pipeline import loss
from pipeline.pipeline_utils import convert_tf_params_to_bbox


class ResFocusNetwork(BaseModel):
    def __init__(
        self,
        loss_lambda_tr: float,
        loss_lambda_sc: float,
        loss_lambda_rot: float,
        backbone: str,
        threshold: float = 0.5,
        inp_img_size: Tuple[int, int] = (640, 640),
        loss_weights: List[float] = [1.0],
        loss_rot: bool = True,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.loss_lambda_tr = loss_lambda_tr
        self.loss_lambda_sc = loss_lambda_sc
        self.loss_lambda_rot = loss_lambda_rot
        self.loss_weights = loss_weights
        self.inp_img_size = inp_img_size
        self.loss_rot = loss_rot

        self.model = assign_backbone(backbone)
        res_fc_out = self.model.fc.out_features
        for param in self.model.parameters():
            param.require_grad = True

        self.cls_fc = nn.Sequential(
            nn.Linear(in_features=res_fc_out, out_features=1024),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.tf_fc = nn.Sequential(
            nn.Linear(in_features=res_fc_out, out_features=1024),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
        )
        self.tf_trx = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.tf_try = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.tf_scale = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.tf_rot = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, x, target):
        x = self.model(x)
        out_cls = self.cls_fc(x)
        out_tf = self.tf_fc(x)
        out_translate_x = self.tf_trx(out_tf)
        out_translate_y = self.tf_try(out_tf)
        out_scale = self.tf_scale(out_tf)
        out_rotate = self.tf_rot(out_tf)

        loss_dict = loss.focus_multiloss_2(
            output_cls=out_cls,
            output_translate_x=out_translate_x,
            output_translate_y=out_translate_y,
            output_scale=out_scale,
            output_rotate=out_rotate,
            target=target,
            lambda_translation=self.loss_lambda_tr,
            lambda_scale=self.loss_lambda_sc,
            lambda_rotation=self.loss_lambda_rot,
            weights=self.loss_weights,
            loss_rot=self.loss_rot,
        )

        return {
            "out_cls": out_cls,
            "out_translate_x": out_translate_x,
            "out_translate_y": out_translate_y,
            "out_scale": out_scale,
            "out_rotate": out_rotate,
            "loss": loss_dict,
        }

    def get_prediction(self, output, target):
        cls_out = output["out_cls"]
        translate_x_out = output["out_translate_x"]
        translate_y_out = output["out_translate_y"]
        scale_out = output["out_scale"]
        bbox = convert_tf_params_to_bbox(
            translations=torch.cat([translate_x_out, translate_y_out], dim=1),
            scales=scale_out,
            img_size=self.inp_img_size,
        )
        return {
            "label": (cls_out >= self.threshold).float().squeeze(),
            "confidence": cls_out.squeeze(),
            "bbox": bbox,
        }
