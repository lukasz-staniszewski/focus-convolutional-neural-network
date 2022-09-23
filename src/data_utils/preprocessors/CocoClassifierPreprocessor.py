from base import BasePreprocessor
import torchvision
from data_utils.constants import COCO_2017_LABEL_MAP
from tqdm import tqdm
import os
import pandas as pd
from copy import deepcopy
from math import floor
import numpy as np
from PIL import Image


class CocoClassifierPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_pos = kwargs["label_pos"]
        self.labels_neg = kwargs["labels_neg"]
        self.img_out_shape = kwargs["img_out_shape"]
        self.reflect_padding_cut = kwargs["reflect_padding_cut"]
        self.cut_fn = getattr(self, kwargs["cut_fn"])

        self.logger.info(
            f"Starting mapping CocoDetection dataset to memory..."
        )
        self.dataset = torchvision.datasets.CocoDetection(
            root=self.img_in_dir_path, annFile=self.ann_file_path
        )
        self.logger.info(f"Mapping finished.")

        self.X, self.y = None, None
        self.filenames = None
        self.df_out = None
        self.img_idx = 0

    def _resize_img(self, img):
        if self.img_out_shape:
            img = img.resize(self.img_out_shape)
        return img

    def _cut_square(self, image, bbox):
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

    def _cut_bbox(self, image, bbox):
        return image.crop(
            (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        )

    def cut_min_covering_square(self, labels=None):
        if labels and type(labels) == str:
            labels = [labels]
        X, filenames = [], []

        for img, objects in tqdm(
            self.dataset, desc="Preprocessing images cutting"
        ):
            for obj in objects:
                img_in = deepcopy(img)
                if (
                    labels is None
                    or COCO_2017_LABEL_MAP[obj["category_id"]] in labels
                ):
                    self.img_idx += 1
                    bbox = deepcopy(obj["bbox"])
                    if self.reflect_padding_cut:
                        diff = floor(abs(bbox[2] - bbox[3]))
                        padded = np.pad(
                            img_in,
                            ((diff, diff), (diff, diff), (0, 0)),
                            "reflect",
                        )
                        img_in = Image.fromarray(
                            padded.astype("uint8"), "RGB"
                        )
                        bbox = [
                            bbox[0] + diff,
                            bbox[1] + diff,
                            bbox[2],
                            bbox[3],
                        ]

                    img_in = self._cut_square(img_in, bbox=bbox)
                    img_in = self._resize_img(img_in)

                    X.append(img_in)
                    filenames.append(f"{self.img_idx}.jpg")

        return X, filenames

    def cut_bbox_stretched(self, labels=None):
        if labels and type(labels) == str:
            labels = [labels]
        X, filenames = [], []

        for img, objects in tqdm(
            self.dataset, desc="Preprocessing images cutting"
        ):
            for obj in objects:
                img_in = deepcopy(img)
                if (
                    labels is None
                    or COCO_2017_LABEL_MAP[obj["category_id"]] in labels
                ):
                    self.img_idx += 1
                    img_in = self._cut_bbox(img_in, obj["bbox"])
                    img_in = self._resize_img(img_in)
                    X.append(img_in)
                    filenames.append(f"{self.img_idx}.jpg")

        return X, filenames

    def _collect_data(self):
        self.logger.info(
            f"Starting preprocessing images for label 1..."
        )
        X_pos, filenames_pos = self.cut_fn(self.label_pos)
        y_pos = [1 for _ in range(len(X_pos))]
        self.logger.info(f"Preprocessing label 1 images finished.")
        self.logger.info(
            f"Starting preprocessing images for label 0..."
        )
        X_neg, filenames_neg = self.cut_fn(self.labels_neg)
        y_neg = [0 for _ in range(len(X_neg))]
        self.logger.info(f"Preprocessing label 0 images finished.")
        self.logger.info(
            "Images preprocessed, number of label 1 images:"
            f" {len(X_pos)} and label 0 images: {len(X_neg)}."
        )

        self.X = X_pos + X_neg
        self.y = y_pos + y_neg
        self.filenames = filenames_pos + filenames_neg
        self.df_out = pd.DataFrame(
            {
                "filename": self.filenames,
                "label": self.y,
            }
        )

    def _save_data(self):
        assert (
            self.X is not None
            and self.y is not None
            and self.df_out is not None
            and self.filenames is not None
        ), "Data not collected."

        assert os.path.exists(
            self.img_out_dir_path
        ), "Output dir not found."
        os.makedirs(
            os.path.join(self.img_out_dir_path, "images"), exist_ok=True
        )

        self.df_out.to_csv(
            os.path.join(self.img_out_dir_path, "labels.csv"),
            index=False,
            sep=";",
            header=True,
        )

        self.logger.info("Saving images...")
        for img, filename in tqdm(
            zip(self.X, self.filenames), desc="Saving images"
        ):
            img.save(
                os.path.join(self.img_out_dir_path, "images", filename)
            )
        self.logger.info(
            "Images saved. Preprocessed data saved in"
            f" {self.img_out_dir_path}"
        )

    def preprocess(self):
        self._collect_data()
        self._save_data()
