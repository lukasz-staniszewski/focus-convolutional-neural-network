from base import BasePreprocessor
import torchvision
from data_utils.constants import COCO_2017_LABEL_MAP
import tqdm
import os
import pandas as pd


class CocoClassifierPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.img_in_dir_path = config["img_in_dir_path"]
        self.ann_file_path = config["ann_file_path"]
        self.img_out_dir_path = config["img_out_dir_path"]
        self.label_pos = config["label_pos"]
        self.labels_neg = config["labels_neg"]
        self.dataset = torchvision.datasets.CocoDetection(
            root=self.img_in_dir_path, annFile=self.ann_file_path
        )
        self.img_out_shape = config["img_out_shape"]
        self.X, self.y = None, None
        self.filenames = None
        self.df_out = None

    def _cut_imgs_per_labels(self, labels):
        if labels and type(labels) == str:
            labels = [labels]

        X, filenames = [], []
        for img, objects in tqdm(
            self.dataset, desc="Preprocessing images cutting:"
        ):
            for obj in objects:
                if labels is None or (
                    labels
                    and COCO_2017_LABEL_MAP[obj["category_id"]]
                    in labels
                ):
                    obj_bbox = obj["bbox"]
                    img = img.crop(
                        (
                            obj_bbox[0],
                            obj_bbox[1],
                            obj_bbox[0] + obj_bbox[2],
                            obj_bbox[1] + obj_bbox[3],
                        )
                    ).resize(self.img_out_shape)

                    X.append(img)
                    filenames.append(obj["file_name"])

        return X, filenames

    def _collect_data(self):
        X_pos, filenames_pos = self._cut_imgs_per_labels(self.label_pos)
        y_pos = [1 for _ in range(len(X_pos))]
        X_neg, filenames_neg = self._cut_imgs_per_labels(
            self.labels_neg
        )
        y_neg = [0 for _ in range(len(X_neg))]
        print(
            "Images preprocessed, number of positive images:"
            f" {len(X_pos)} and negative images: {len(X_neg)}"
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
        for img, filename in zip(self.X, self.filenames):
            img.save(
                os.path.join(self.img_out_dir_path, "images", filename)
            )
        print(f"Preprocessed data saved in {self.img_out_dir_path}")

    def preprocess(self):
        self._collect_data()
        self._save_data()
