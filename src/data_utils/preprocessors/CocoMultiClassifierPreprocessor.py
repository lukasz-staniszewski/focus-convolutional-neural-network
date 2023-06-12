import os
from copy import deepcopy
from math import floor, inf
from typing import List

import numpy as np
import pandas as pd
import PIL
import torchvision
from PIL import Image
from tqdm import tqdm

from base import BasePreprocessor
from data_utils.constants import COCO_2017_LABEL_MAP


class CocoMultiClassifierPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        """Preprocessor constructor."""
        super().__init__(*args, **kwargs)
        self.labels = {int(k): v for k, v in kwargs["labels"].items()}

        self.labels_cnt = {}
        for label in self.labels.keys():
            self.labels_cnt[label] = 0

        self.idx_2_label = self.labels
        self.label_2_idx = {v: k for k, v in self.labels.items()}

        self.img_out_shape = kwargs["img_out_shape"]
        self.reflect_padding_cut = kwargs["reflect_padding_cut"]
        self.cut_fn = getattr(self, kwargs["cut_fn"])

        if (
            "label_max_sz" not in kwargs.keys()
            or kwargs["label_max_sz"] is None
        ):
            self.label_max_sz = inf
        elif isinstance(kwargs["label_max_sz"], int):
            self.label_max_sz = kwargs["label_max_sz"]
        else:
            raise ValueError("Parameter label_max_sz must be int or None.")

        self.logger.info(
            f"Starting mapping CocoDetection dataset to memory..."
        )
        self.dataset = torchvision.datasets.CocoDetection(
            root=self.img_in_dir_path, annFile=self.ann_file_path
        )
        self.logger.info(f"Mapping finished.")

        self.y = []
        self.filenames = []
        self.df_out = None
        self.img_idx = 0

        self._prepare_output_img_dir()

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

    def _cut_square(self, image: PIL.Image, bbox: List[int]) -> PIL.Image:
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

    def collect_files(self) -> None:
        """Function collects images and their pathnames with correct preprocessing of bounding box."""
        for img, objects in tqdm(
            self.dataset,
            total=len(self.dataset),
            desc="Collecting data",
            colour="green",
        ):
            for obj in objects:
                label_COCO = COCO_2017_LABEL_MAP[obj["category_id"]]
                if label_COCO in self.labels.values():
                    label_idx = self.label_2_idx[label_COCO]
                else:
                    label_idx = 0

                if self.labels_cnt[label_idx] > self.label_max_sz:
                    continue
                else:
                    img_in = deepcopy(img)
                    bbox = deepcopy(obj["bbox"])

                    self.img_idx += 1
                    img_in = self.cut_fn(img_in=img_in, bbox=bbox)

                    filename = f"{self.img_idx}.jpg"
                    img_in.save(os.path.join(self.img_out_dir_path, filename))

                    self.filenames.append(filename)
                    self.y.append(label_idx)
                    self.labels_cnt[label_idx] += 1

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
        self.collect_files()
        class_distribution = {
            self.idx_2_label[k]: v for k, v in self.labels_cnt.items()
        }
        self.logger.info(
            "Images preprocessed, initial class distribution:"
            f" {class_distribution}"
        )

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
