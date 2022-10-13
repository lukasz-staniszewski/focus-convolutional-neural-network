from typing import List, Union

import PIL
from base import BasePreprocessor
import torchvision
from data_utils.constants import COCO_2017_LABEL_MAP
from tqdm import tqdm
import os
import pandas as pd
from copy import deepcopy
from math import floor, inf
import numpy as np
from PIL import Image


class CocoClassifierPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        """Preprocessor constructor."""
        super().__init__(*args, **kwargs)
        self.label_pos = kwargs["label_pos"]
        self.labels_neg = kwargs["labels_neg"]
        self.img_out_shape = kwargs["img_out_shape"]
        self.reflect_padding_cut = kwargs["reflect_padding_cut"]
        self.cut_fn = getattr(self, kwargs["cut_fn"])

        if kwargs["label_max_sz"] is None:
            self.label_max_sz = inf
        elif isinstance(kwargs["label_max_sz"], int):
            self.label_max_sz = kwargs["label_max_sz"]
        else:
            raise ValueError(
                "Parameter label_max_sz must be int or None."
            )

        self.logger.info(
            f"Starting mapping CocoDetection dataset to memory..."
        )
        self.dataset = torchvision.datasets.CocoDetection(
            root=self.img_in_dir_path, annFile=self.ann_file_path
        )
        self.logger.info(f"Mapping finished.")

        self.y = None
        self.filenames = None
        self.df_out = None
        self.img_idx = 0

        self._prepare_output_img_dir()

    def _prepare_output_img_dir(self):
        """Prepares output directory for images."""
        assert os.path.exists(
            self.out_dir_path
        ), "Output dir not found."
        self.img_out_dir_path = os.path.join(
            self.out_dir_path, "images"
        )
        os.makedirs(self.img_out_dir_path, exist_ok=True)

    def _resize_img(self, img: PIL.Image) -> PIL.Image:
        """Resizes image to own shape.

        Args:
            img (PIL.Image): image to resize

        Returns:
            PIL.Image: resized image
        """
        if self.img_out_shape:
            img = img.resize(self.img_out_shape)
        return img

    def _cut_square(
        self, image: PIL.Image, bbox: List[int]
    ) -> PIL.Image:
        """Cuts minimal square that contains bounding box.

        Args:
            image (PIL.Image): image to cut from
            bbox (List[int]): bounding box coordinates

        Returns:
            PIL.Image: cut image
        """
        center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        radius = max(bbox[2], bbox[3]) / 2
        return image.crop(
            (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            )
        )

    def _cut_bbox(self, image: PIL.Image, bbox: List[int]) -> PIL.Image:
        """Cuts bounding box from image.

        Args:
            image (PIL.Image): image to cut from
            bbox (List[int]): bounding box coordinates

        Returns:
            PIL.Image: cut image
        """
        return image.crop(
            (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        )

    def collect_filenames_per_label(
        self, labels: Union[List[str], str] = None
    ) -> List[str]:
        """Function collects images and their pathnames with correct preprocessing of bounding box.

        Args:
            labels (Union[List[str], str], optional): objects with those labels will be taken into account. If none, images of all labels are taken into account. Defaults to None.

        Returns:
            List[str]: filenames for objects
        """
        if self.preprocessing_pos:
            assert labels and isinstance(labels, str)
            labels = [labels]
        else:
            assert labels is None or isinstance(labels, list)
            if labels is None:
                labels = list(COCO_2017_LABEL_MAP.values())
                labels.remove(self.label_pos)
        filenames = []
        n_imgs_label = 0

        for img, objects in self.dataset:
            for obj in objects:
                if COCO_2017_LABEL_MAP[obj["category_id"]] in labels:
                    img_in = deepcopy(img)
                    self.img_idx += 1
                    bbox = deepcopy(obj["bbox"])
                    img_in = self.cut_fn(img_in=img_in, bbox=bbox)

                    filename = f"{self.img_idx}.jpg"
                    img_in.save(
                        os.path.join(self.img_out_dir_path, filename)
                    )
                    filenames.append(filename)

                    n_imgs_label += 1
                    if (
                        self.label_max_sz != 0
                        and n_imgs_label >= self.label_max_sz
                    ):
                        return filenames

        return filenames

    def cut_min_covering_square(
        self, img_in: PIL.Image, bbox: List[int]
    ) -> PIL.Image:
        """Function cuts minimal square that contains fully bounding box and resizes it to correct shape.

        Args:
            img_in (PIL.Image): image to cut from
            bbox (List[int]): original bounding box coordinates

        Returns:
            PIL.Image: transformed image
        """
        if self.reflect_padding_cut:
            diff = floor(abs(bbox[2] - bbox[3]))
            padded = np.pad(
                img_in,
                ((diff, diff), (diff, diff), (0, 0)),
                "reflect",
            )
            img_in = Image.fromarray(padded.astype("uint8"), "RGB")
            bbox = [
                bbox[0] + diff,
                bbox[1] + diff,
                bbox[2],
                bbox[3],
            ]
        img_in = self._cut_square(img_in, bbox=bbox)
        img_in = self._resize_img(img_in)
        return img_in

    def cut_bbox_stretched(
        self, img_in: PIL.Image, bbox: List[int]
    ) -> PIL.Image:
        """Function cuts bounding box and resizes it to correct shape (with possible stretch of image).

        Args:
            img_in (PIL.Image): image to cut from
            bbox (List[int]): original bounding box coordinates

        Returns:
            PIL.Image: transformed image
        """
        img_in = self._cut_bbox(img_in=img_in, bbox=bbox)
        img_in = self._resize_img(img_in=img_in)
        return img_in

    def _collect_data(self):
        """Function collects data to create dataset."""
        self.logger.info(
            f"Starting preprocessing images for label 1..."
        )
        self.preprocessing_pos = True
        filenames_pos = self.collect_filenames_per_label(self.label_pos)
        y_pos = [1 for _ in range(len(filenames_pos))]
        self.logger.info(f"Preprocessing label 1 images finished.")

        if self.label_max_sz == 0:
            self.label_max_sz = len(filenames_pos)
        self.logger.info(
            f"Starting preprocessing images for label 0..."
        )
        self.preprocessing_pos = False
        filenames_neg = self.collect_filenames_per_label(
            self.labels_neg
        )
        y_neg = [0 for _ in range(len(filenames_neg))]
        self.logger.info(f"Preprocessing label 0 images finished.")
        self.logger.info(
            "Images preprocessed, number of label 1 images:"
            f" {len(filenames_pos)} and label 0 images:"
            f" {len(filenames_neg)}."
        )

        self.y = y_pos + y_neg
        self.filenames = filenames_pos + filenames_neg
        self.df_out = pd.DataFrame(
            {
                "filename": self.filenames,
                "label": self.y,
            }
        )

    def _save_data(self):
        """Saves data to disk. In this case, saved is csv file with images filenames and labels."""
        assert (
            self.y is not None
            and self.df_out is not None
            and self.filenames is not None
        ), "Data not collected."

        self.df_out.to_csv(
            os.path.join(self.out_dir_path, "labels.csv"),
            index=False,
            sep=",",
            header=True,
        )

    def preprocess(self):
        """General method for preprocessing images."""
        self._collect_data()
        self._save_data()
