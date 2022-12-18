import random
from base.base_preprocessor import BasePreprocessor
from PIL import Image, ImageOps
from data_utils.constants import COCO_2017_LABEL_MAP
from base import BasePreprocessor
from math import inf
import torchvision
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from math import log


class CocoFocusPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_pos = kwargs["label_pos"]
        self.labels_neg = kwargs["labels_neg"]
        self.max_input_size = kwargs["max_input_size"]
        self.img_out_shape = kwargs["img_out_shape"]

        self.enable_rotation = kwargs["enable_rotation"]

        if kwargs["label_max_size"] is None:
            self.label_max_size = inf
        elif isinstance(kwargs["label_max_size"], int):
            self.label_max_size = kwargs["label_max_size"]
        else:
            raise ValueError("Parameter label_max_size must be int or None.")

        self.logger.info("Starting mapping CocoDetection dataset to memory...")
        self.dataset = torchvision.datasets.CocoDetection(
            root=self.img_in_dir_path, annFile=self.ann_file_path
        )
        self.logger.info(f"Mapping finished.")

        self.labels = []
        self.translate_xs = []
        self.translate_ys = []
        self.thetas = []
        self.scale_factors = []

        self.filenames = []
        self.df_out = None
        self.img_counter = 0

        self._prepare_output_img_dir()

    def _calculate_max_image_size(self):
        # assuming max possible image is square
        # then max image size is after rotation by 45 degrees
        width = self.max_input_size[0]
        height = self.max_input_size[1]

        max_width, max_height = width, height

        if self.enable_rotation:
            max_width = width * np.cos(np.deg2rad(45)) + height * np.sin(
                np.deg2rad(45)
            )
            max_height = width * np.sin(np.deg2rad(45)) + height * np.cos(
                np.deg2rad(45)
            )

        return int(max_width), int(max_height)

    def _pad_with_bboxes(self, img, bboxes):
        img_padded = self._add_padding_to_img(img)
        pad_width = (img_padded.size[0] - img.size[0]) // 2
        pad_height = (img_padded.size[1] - img.size[1]) // 2
        new_bboxes = [
            [bbox[0] + pad_width, bbox[1] + pad_height, bbox[2], bbox[3]]
            for bbox in bboxes
        ]
        return img_padded, new_bboxes

    def _add_padding_to_img(self, img):
        max_width, max_height = self._calculate_max_image_size()
        delta_width = max_width - img.size[0]
        delta_height = max_height - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return ImageOps.expand(img, padding)

    def _make_bbox_square(self, bbox):
        center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        radius = max(bbox[2], bbox[3]) / 2
        return [center[0] - radius, center[1] - radius, radius * 2, radius * 2]

    def _make_bbox_rotate(self, bbox, deg, img_center):
        center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        radius = max(bbox[2], bbox[3]) / 2

        new_center_x = (
            (center[0] - img_center[0]) * np.cos(np.deg2rad(deg))
            + (center[1] - img_center[1]) * np.sin(np.deg2rad(deg))
            + img_center[0]
        )
        new_center_y = (
            -(center[0] - img_center[0]) * np.sin(np.deg2rad(deg))
            + (center[1] - img_center[1]) * np.cos(np.deg2rad(deg))
            + img_center[1]
        )

        new_center = [new_center_x, new_center_y]
        return [
            new_center[0] - radius,
            new_center[1] - radius,
            radius * 2,
            radius * 2,
        ]

    def _get_bbox_closest_to_center(self, bboxes, image):
        centers = [
            (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2) for bbox in bboxes
        ]
        img_center = (image.size[0] / 2, image.size[1] / 2)
        distances_to_center = [
            np.sqrt(
                (center[0] - img_center[0]) ** 2
                + (center[1] - img_center[1]) ** 2
            )
            for center in centers
        ]
        idx_min = distances_to_center.index(min(distances_to_center))
        return bboxes[idx_min]

    def _collect_data(self):
        """Function collects data to create dataset."""
        self.collect_files()

        class_distribution = {
            "none": (len(self.labels) - sum(self.labels)),
            self.label_pos: sum(self.labels),
        }

        self.logger.info(
            "Images preprocessed, initial class distribution:"
            f" {class_distribution}"
        )

        self.df_out = pd.DataFrame(
            {
                "filename": self.filenames,
                "label": self.labels,
                "translate_x": self.translate_xs,
                "translate_y": self.translate_ys,
                "scale_factor": self.scale_factors,
                "theta": self.thetas,
            }
        )

    def _get_transform_params_from_bbox(self, bbox, image_padded, rot_deg):
        center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        radius = max(bbox[2], bbox[3]) / 2
        img_padded_center = (
            image_padded.size[0] / 2,
            image_padded.size[1] / 2,
        )

        scale = log(radius * 2 / image_padded.size[0])
        diff_x = 2 * (center[0] - img_padded_center[0]) / image_padded.size[0]
        diff_y = 2 * (center[1] - img_padded_center[1]) / image_padded.size[1]

        rotation = -rot_deg * np.pi / 180

        return scale, diff_x, diff_y, rotation

    def _perform_rotation(self, image):
        # probability of rotation is 1/2
        if random.random() < 0.5:
            return image, 0
        else:
            # rotate image random angle from -70 to 70 degrees
            deg = random.randint(-85, 85)
            return image.rotate(deg), deg

    def collect_files(self):
        for image, annotations in tqdm(
            self.dataset,
            total=len(self.dataset),
            desc="Collecting data",
            colour="green",
        ):
            if len(annotations) == 0:
                continue
            # bboxes annotated as label_pos
            pos_bboxes = [
                ann["bbox"]
                for ann in annotations
                if COCO_2017_LABEL_MAP[ann["category_id"]] == self.label_pos
            ]
            # pad image to same size and change bboxes coordinates
            image_padded, pos_bboxes = self._pad_with_bboxes(
                img=image, bboxes=pos_bboxes
            )
            # rotate with probability
            if self.enable_rotation:
                image_rotated, rot_degree = self._perform_rotation(
                    image_padded
                )
            else:
                image_rotated, rot_degree = image_padded, 0
            # save file
            self.img_counter += 1
            filename = f"{self.img_counter}.jpg"
            image_rotated.save(os.path.join(self.img_out_dir_path, filename))
            self.filenames.append(filename)
            # if there are no bboxes annotated as label_pos, save image as negative
            if len(pos_bboxes) == 0:
                self.labels.append(0)
                self.translate_xs.append(0)
                self.translate_ys.append(0)
                self.thetas.append(0)
                self.scale_factors.append(0)
            # positive image
            else:
                # make bbox square
                pos_bboxes = list(
                    map(lambda x: self._make_bbox_square(bbox=x), pos_bboxes)
                )
                # modify bbox center according to rotation
                pos_bboxes = list(
                    map(
                        lambda x: self._make_bbox_rotate(
                            bbox=x,
                            deg=rot_degree,
                            img_center=(
                                image_rotated.size[0] / 2,
                                image_rotated.size[1] / 2,
                            ),
                        ),
                        pos_bboxes,
                    )
                )
                # if there are more than one bbox annotated as label_pos, choose the one closest to the center
                if len(pos_bboxes) > 1:
                    bbox = self._get_bbox_closest_to_center(
                        pos_bboxes, image_padded
                    )
                # if there is only one bbox annotated as label_pos, choose it
                else:
                    bbox = pos_bboxes[0]
                # get transformation parameters
                (
                    scale,
                    diff_x,
                    diff_y,
                    rotation,
                ) = self._get_transform_params_from_bbox(
                    bbox=bbox,
                    image_padded=image_padded,
                    rot_deg=rot_degree,
                )
                self.labels.append(1)
                self.translate_xs.append(diff_x)
                self.translate_ys.append(diff_y)
                self.thetas.append(rotation)
                self.scale_factors.append(scale)

    def preprocess(self):
        self._collect_data()
        self._save_data()

    def _save_data(self):
        """Saves data to disk. In this case, saved is csv file with images filenames and labels."""
        assert (
            self.labels
            and self.filenames
            and self.translate_xs
            and self.translate_ys
            and self.thetas
            and self.scale_factors
            and self.df_out is not None
        ), "Data not collected."

        self.df_out.to_csv(
            os.path.join(self.out_dir_path, "labels.csv"),
            index=False,
            sep=",",
            header=True,
        )
