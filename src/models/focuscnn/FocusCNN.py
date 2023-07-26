from collections import OrderedDict
from typing import Dict, Iterator, Tuple

import torch
from torch.nn.parameter import Parameter

from base import BaseModel
from models import TransformModule
from models.classifier.MultiClassifier import MultiClassifier
from models.focus.ResFocusNetwork import ResFocusNetwork
from pipeline import loss
from pipeline.pipeline_utils import convert_tf_params_to_bbox


class FocusCNN(BaseModel):
    def __init__(
        self,
        classifier_model: MultiClassifier,
        focus_models: OrderedDict[str, ResFocusNetwork],
        inp_img_size: Tuple[int, int] = (640, 640),
        out_img_size: Tuple[int, int] = (300, 300),
        model_checkpoints: Dict[str, str] = None,
    ) -> None:
        super().__init__()
        self.inp_img_size = inp_img_size
        self.out_img_size = out_img_size
        self.classifier_model = classifier_model
        self.focus_models = focus_models
        self.tf_module = TransformModule(image_out_sz=out_img_size)
        if model_checkpoints is not None:
            self.load_model(
                classifier_model_path=model_checkpoints["classifier_model"],
                focus_models_path=model_checkpoints["focus_models"],
            )
            print("FocusCNN model loaded from checkpoint.")

    def get_n_model_params(self) -> str:
        """Function to print number of model trainable parameters.

        Returns:
            str: model information as string
        """

        def get_n_params(child):
            return int(child.get_n_model_params()[22:])

        all_params = 0
        all_params += get_n_params(self.classifier)
        for focus_model in self.focus_models.values():
            all_params += get_n_params(focus_model)
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

    def load_model(
        self,
        classifier_model_path: str,
        focus_models_path: OrderedDict[str, str],
    ) -> None:
        """Loads model from config file.

        Args:
            classifier_model_path (str): path to classifier model checkpoint
            focus_models_path (Dict[str, str]): dictionary of paths to focus models
        """
        assert classifier_model_path is not None and all(
            [
                focus_model_path is not None
                for focus_model_path in focus_models_path.values()
            ]
        ), "Models checkpoint paths are not specified!"

        assert set(focus_models_path.keys()) == set(
            self.focus_models.keys()
        ), "Focus models ids are not matching!"

        self.classifier_model.load_state_dict(
            torch.load(classifier_model_path)["state_dict"]
        )
        for fm_id, fm_model in self.focus_models.items():
            state_dict = torch.load(focus_models_path[fm_id])["state_dict"]
            try:
                fm_model.load_state_dict(state_dict)
            except RuntimeError:
                state_dict = {
                    k.replace("_orig_mod.", ""): v
                    for (k, v) in state_dict.items()
                }
                fm_model.load_state_dict(state_dict)

    def forward(self, x, target):
        outs_focus = {
            cls_idx: focus_model(x, target[cls_idx])
            for cls_idx, focus_model in self.focus_models.items()
        }

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

        focus_models_loss = [
            out_focus["loss"]["translation_x_loss"]
            + out_focus["loss"]["translation_y_loss"]
            + out_focus["loss"]["scale_log_loss"]
            for out_focus in outs_focus.values()
        ]

        inputs_cls = torch.cat(
            [
                self.tf_module(x, out_focus_cat)
                for out_focus_cat in outs_focus_cat.values()
            ]
        )
        target_cls = torch.cat(
            [target[cls_idx]["label"] for cls_idx in self.focus_models.keys()]
        )
        outs_cls = self.classifier_model(inputs_cls)

        loss_val = loss.cross_entropy_loss_weighted(
            output=outs_cls,
            target=target_cls,
            weights=[0.3221, 1.3864, 7.2780, 27.5914],
        ) + 0.25 * sum(focus_models_loss)
        # )

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
            "target_cls": target_cls,
            "loss": loss_val,
        }

    def get_predictions(self, output, img_ids):
        out_cls = output["out_cls"]
        outs_translate_x = output["outs_translate_x"]
        outs_translate_y = output["outs_translate_y"]
        outs_scale = output["outs_scale"]

        cls_soft = torch.nn.functional.softmax(out_cls, dim=1)
        confidences = torch.max(cls_soft, dim=1)[0]
        predictions = torch.argmax(out_cls, dim=1)

        model_predictions = []
        for img_idx in range(img_ids.shape[0]):
            image_predictions = {
                "boxes": [],
                "scores": [],
                "labels": [],
            }
            n_predictions = 0
            for cls_idx in self.focus_models.keys():
                if predictions[
                    (int(cls_idx) - 1) * img_ids.shape[0] + img_idx
                ] == int(cls_idx):
                    # prediction for own class
                    n_predictions += 1
                    image_predictions["scores"].append(
                        float(
                            confidences[
                                (int(cls_idx) - 1) * img_ids.shape[0] + img_idx
                            ]
                        )
                    )
                    image_predictions["labels"].append(int(cls_idx))
                    image_predictions["boxes"].append(
                        (
                            convert_tf_params_to_bbox(
                                translations=torch.cat(
                                    [
                                        outs_translate_x[cls_idx][img_idx],
                                        outs_translate_y[cls_idx][img_idx],
                                    ]
                                ).unsqueeze(0),
                                scales=outs_scale[cls_idx][img_idx],
                                img_size=self.inp_img_size,
                            )
                            .squeeze(0)
                            .tolist()
                        )
                    )
            image_predictions["boxes"] = torch.Tensor(
                image_predictions["boxes"]
            )
            image_predictions["scores"] = torch.Tensor(
                image_predictions["scores"]
            )
            image_predictions["labels"] = torch.LongTensor(
                image_predictions["labels"]
            )

            model_predictions.append(image_predictions)
        return {
            "map_predictions": model_predictions,
            "cls_predictions": predictions,
        }

    def prepare_target_for_map(self, target_bboxes, target_image_ids):
        target = []
        for image_id in target_image_ids:
            image_bboxes = target_bboxes[
                target_bboxes[:, 0] == image_id
            ]  # Filter bboxes for current image
            if len(image_bboxes) > 0:
                # Convert [x, y, w, h] to [xmin, ymin, xmax, ymax]
                boxes = torch.stack(
                    [
                        image_bboxes[:, 2],
                        image_bboxes[:, 3],
                        image_bboxes[:, 2] + image_bboxes[:, 4],
                        image_bboxes[:, 3] + image_bboxes[:, 5],
                    ],
                    dim=1,
                )
                labels = image_bboxes[:, 1].long()
            else:
                boxes = torch.tensor([])
                labels = torch.tensor([]).long()
            target.append({"boxes": boxes, "labels": labels})
        return target

    def train(self, mode: bool = True):
        super().train(mode)
        self.classifier_model.train(mode)
        for focus_model in self.focus_models.values():
            focus_model.train(mode)
        return self

    def eval(self):
        super().eval()
        self.classifier_model.eval()
        for focus_model in self.focus_models.values():
            focus_model.eval()
        return self

    def to(self, device):
        super().to(device)
        self.classifier_model.to(device)
        for focus_model in self.focus_models.values():
            focus_model.to(device)
        return self
