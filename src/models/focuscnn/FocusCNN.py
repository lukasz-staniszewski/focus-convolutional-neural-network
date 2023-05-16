from utils.ConfigParser import ConfigParser
import models as module_arch
from models import TransformModule
from base import BaseModel
from pipeline import loss
from pipeline.pipeline_utils import convert_tf_params_to_bbox
from typing import List, Dict, Any, Tuple
import torch


class FocusCNN(BaseModel):
    def __init__(
        self,
        classifier_model: Dict[str, Any],
        focus_models: List[Dict[str, Any]],
        inp_img_size: Tuple[int, int] = (640, 640),
        out_img_size: Tuple[int, int] = (300, 300),
    ) -> None:
        super().__init__()
        self.inp_img_size = inp_img_size
        self.out_img_size = out_img_size
        self.classifier_dict = classifier_model
        self.focus_models_dicts = focus_models
        self.cfg_parser_cls = ConfigParser(classifier_model)
        self.cfg_parsers_focus = {
            f["id_cls"]: ConfigParser(f) for f in focus_models
        }
        self.classifier_model = self.cfg_parser_cls.init_obj(
            "arch", module=module_arch
        )
        self.focus_models = {
            id_cls: parserer.init_obj("arch", module=module_arch)
            for id_cls, parserer in self.cfg_parsers_focus.items()
        }
        self.tf_module = TransformModule(img_out_sz=out_img_size)

    def get_n_model_params(self) -> str:
        """Function to print number of model trainable parameters.

        Returns:
            str: model information as string
        """
        all_params = 0
        all_params += self.classifier.get_n_model_params()
        for focus_model in self.focus_models.values():
            all_params += focus_model.get_n_model_params()
        return "Trainable parameters: {}".format(all_params)

    def __str__(self) -> str:
        """Base function to print model information.

        Returns:
            str: model information as string
        """
        focus_details = "\n\n".join(
            [
                f"Focus({cls_idx}):\n{str(model)}"
                for cls_idx, model in self.focus_models.items()
            ]
        )
        return (
            "Trainable parameters:\nFocus network params:{}\n\nClassifier"
            " params:{}".format(focus_details, str(self.classifier_model))
        )

    def load_model(self, path: str) -> None:
        """Loads model from config file.

        Args:
            path (str): UNUSED
        """
        assert self.classifier_dict["model_checkpoint"] is not None and all(
            [
                f["model_checkpoint"] is not None
                for f in self.focus_models_dicts
            ]
        ), "Models checkpoint paths are not specified!"

        self.classifier_model.load_state_dict(
            torch.load(self.classifier_dict["model_checkpoint"])
        )
        for focus_model_info in self.focus_models_dicts():
            self.focus_models[focus_model_info["id_cls"]].load_state_dict(
                torch.load(focus_model_info["model_checkpoint"])
            )

    def forward(self, x, target):
        outs_focus = [
            focus_model(x, target[cls_idx])
            for cls_idx, focus_model in self.focus_models.items()
        ]

        outs_focus_cat = {
            idx_cls: torch.cat(
                [
                    out_focus["out_translate_x"],
                    out_focus["out_translate_y"],
                    out_focus["out_scale"],
                    out_focus["out_rotate"],
                ],
                dim=1,
            )
            for idx_cls, out_focus in outs_focus.items()
        }

        inputs_cls = torch.cat(
            [
                self.tf_module(x, out_focus_cat)
                for out_focus_cat in outs_focus_cat.values()
            ]
        )

        target_cls = (
            torch.cat(
                [
                    target[cls_idx]["label"]
                    for cls_idx in self.focus_models.keys()
                ]
            )
            .type(torch.LongTensor)
            .to(x.device)
        )

        outs_cls = self.classifier_model(inputs_cls, target_cls)

        loss_val = loss.cross_entropy_loss(output=outs_cls, target=target_cls)

        return {
            "out_cls": outs_cls,
            "outs_translate_x": {
                cls_idx: out_focus["out_translate_x"]
                for cls_idx, out_focus in outs_focus.items()
            },
            "outs_translate_y": {
                cls_idx: out_focus["out_translate_y"]
                for cls_idx, out_focus in outs_focus.items()
            },
            "outs_scale": {
                cls_idx: out_focus["out_scale"]
                for cls_idx, out_focus in outs_focus.items()
            },
            "loss": loss_val,
        }

    def get_prediction(self, output, img_ids):
        out_cls = output["out_cls"]
        outs_translate_x = output["outs_translate_x"]
        outs_translate_y = output["outs_translate_y"]
        outs_scale = output["outs_scale"]

        confidences, predictions = torch.max(out_cls, dim=1)

        final_out = []
        for img_idx in img_ids.shape[0]:
            for cls_idx in self.focus_models.keys():
                if predictions[
                    (int(cls_idx) - 1) * img_ids.shape[0] + img_idx
                ] == int(cls_idx):
                    # correct prediction
                    final_out.append(
                        {
                            "img_id": img_ids[img_idx],
                            "bbox": convert_tf_params_to_bbox(
                                translations=torch.cat(
                                    [
                                        outs_translate_x[cls_idx][img_idx],
                                        outs_translate_y[cls_idx][img_idx],
                                    ],
                                    dim=1,
                                ),
                                scales=outs_scale[cls_idx][img_idx],
                                img_size=self.inp_img_size,
                            ),
                            "label": int(cls_idx),
                            "confidence": confidences[
                                (int(cls_idx) - 1) * img_ids.shape[0] + img_idx
                            ],
                        },
                    )

        return final_out
