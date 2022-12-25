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
import cv2


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

    def _calculate_rotation_size(self, current_size):
        # assuming max possible image is square
        # then max image size is after rotation by 45 degrees
        width = current_size[0]
        height = current_size[1]
        assert width == height

        max_width = width * np.cos(np.deg2rad(45)) + height * np.sin(
            np.deg2rad(45)
        )
        max_height = width * np.sin(np.deg2rad(45)) + height * np.cos(
            np.deg2rad(45)
        )

        return int(max_width), int(max_height)

    def _add_gauss_reflect_padding(self, img, desired_size):
        # convert to opencv image
        cv2_img = np.array(img)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        # calculate padding sizes
        delta_width = desired_size[0] - img.size[0]
        delta_height = desired_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        # pad image with reflect border
        image = cv2.copyMakeBorder(
            cv2_img,
            top=pad_height,
            bottom=delta_height - pad_height,
            left=pad_width,
            right=delta_width - pad_width,
            borderType=cv2.BORDER_REFLECT,
        )
        # add gauss blur to this image
        image = cv2.GaussianBlur(
            image,
            ksize=(0, 0),
            sigmaX=10,
            sigmaY=10,
            borderType=cv2.BORDER_REFLECT,
        )
        # replace center of image with non-blurred image
        h_in, w_in, _ = cv2_img.shape
        h_bg, w_bg, _ = image.shape
        diff_y, diff_x = round((h_bg - h_in) / 2), round((w_bg - w_in) / 2)
        image[diff_y : diff_y + h_in, diff_x : diff_x + w_in] = cv2_img
        # return as Pillow Image
        img_out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_out)
        return pil_img

    def _move_bboxes(self, pos_bboxes, img_before, img_after):
        pad_width = (img_after.size[0] - img_before.size[0]) // 2
        pad_height = (img_after.size[1] - img_before.size[1]) // 2
        new_bboxes = [
            [bbox[0] + pad_width, bbox[1] + pad_height, bbox[2], bbox[3]]
            for bbox in pos_bboxes
        ]
        return new_bboxes

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
            deg = random.randint(-89, 89)
            return image.rotate(deg), deg

    def _process_image(self, image, pos_bboxes):
        # first pad image to have all images with same size
        img_padded = self._add_gauss_reflect_padding(
            img=image, desired_size=self.max_input_size
        )
        bboxes = self._move_bboxes(
            pos_bboxes=pos_bboxes, img_before=image, img_after=img_padded
        )
        # process rotation
        if self._perform_rotation:
            # pad image for rotation
            rotation_desired_size = self._calculate_rotation_size(
                current_size=img_padded.size
            )
            img_padded_2 = self._add_gauss_reflect_padding(
                img=img_padded, desired_size=rotation_desired_size
            )
            bboxes = self._move_bboxes(
                pos_bboxes=bboxes,
                img_before=img_padded,
                img_after=img_padded_2,
            )
            # additional padding
            rotation_desired_size_adding = self._calculate_rotation_size(
                current_size=img_padded_2.size
            )
            img_padded_3 = self._add_gauss_reflect_padding(
                img=img_padded_2, desired_size=rotation_desired_size_adding
            )
            # rotation
            image_rotated, rot_degree = self._perform_rotation(img_padded_3)
            # calculate position of crop
            img_center = (img_padded_3.size[0] / 2, img_padded_3.size[1] / 2)
            region = (
                img_center[0] - img_padded_2.size[0] / 2,
                img_center[1] - img_padded_2.size[1] / 2,
                img_center[0] + img_padded_2.size[0] / 2,
                img_center[1] + img_padded_2.size[1] / 2,
            )
            image_rotated = image_rotated.crop(region)
        else:
            image_rotated, rot_degree = img_padded, 0

        return image_rotated, rot_degree, bboxes

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
            # image modification because of rotations
            image_rotated, rot_degree, pos_bboxes = self._process_image(
                image=image, pos_bboxes=pos_bboxes
            )
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
                        pos_bboxes, image_rotated
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
                    image_padded=image_rotated,
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
