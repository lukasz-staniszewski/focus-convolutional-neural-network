from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple, Union, List

import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from data_utils.data_loaders.focus_cnn.BaseFocusCNNLoader import (
    BaseFocusCNNLoader,
)
from data_utils.data_loaders.utils import remove_only_0
from data_utils.data_sets import FocusCNNDataset


def collate_func(batch):
    images = torch.stack([item["image"] for item in batch])
    image_id = torch.LongTensor([item["image_id"] for item in batch])
    outputs = {cls: {} for cls in batch[0]["outs"].keys()}
    for key in outputs.keys():
        outputs[key] = {
            "label": torch.LongTensor(
                [item["outs"][key]["label"] for item in batch]
            ),
            "transform": torch.stack(
                [item["outs"][key]["transform"] for item in batch]
            ),
        }
    all_boxes = sum([item["bboxes"] for item in batch], [])
    if len(all_boxes) > 0:
        bboxes = torch.cat(all_boxes)
    else:
        bboxes = None
    return images, image_id, outputs, bboxes


class FocusCNNLoader(BaseFocusCNNLoader):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        images_dir: str,
        csv_path: str,
        labels: Dict[str, str],
        tf_image_size: Tuple[int] = None,
        save_out_dir: str = None,
        transform_mean: Tuple[float] = None,
        transform_std: Tuple[float] = None,
        validation_split: float = 0.0,
        is_test: bool = False,
        balance_train: bool = False,
        balance_methods: List[str] = None,
    ):
        # members
        self.validation_split = validation_split
        self.save_out_dir = save_out_dir
        self.tf_image_size = tf_image_size
        self.balance_train = balance_train
        self.balance_methods = balance_methods

        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            images_dir=images_dir,
            csv_path=csv_path,
            is_test=is_test,
            labels=labels,
            transform_mean=transform_mean,
            transform_std=transform_std,
            collate_fn=collate_func,
        )

    def prepare_dataset_test(self):
        self.dataset_test = FocusCNNDataset(
            images_dir=self.images_dir,
            csv_path=self.csv_path_train,
            transform=self.transform,
            labels=self.labels,
        )

    def prepare_datasets_train_valid(self):
        assert 0.0 <= self.validation_split < 1.0, (
            "Validation split must be in [0.0, 1.0), where 0.0 means no"
            " validation set."
        )

        if self.validation_split == 0.0:
            self.csv_path_train = self.csv_path
        else:
            # validation set
            self._split_train_valid()
            self.dataset_validate = FocusCNNDataset(
                images_dir=self.images_dir,
                csv_path=self.csv_path_valid,
                transform=self.transform,
                labels=self.labels,
            )
        # train set
        if self.balance_train:
            self.balance_data()
        # no balancing
        self.dataset_train = FocusCNNDataset(
            images_dir=self.images_dir,
            csv_path=self.csv_path_train,
            transform=self.transform,
            labels=self.labels,
        )

    def balance_data(self) -> None:
        """Function balances training data by removing examples with only 0 as labels (no objects).

        Possibilities in balance_methods:
        * undersampling:
            -- 'remove_only_0' - remove examples with only 0 as labels (no objects on the image)
        """
        # oversampling
        if "remove_only_0" in self.balance_methods:
            self.csv_path_train = remove_only_0(
                csv_path_train_orig=self.csv_path_train,
            )

    def combine_transforms(
        self,
        transform_mean: Union[Tuple[float], None] = None,
        transform_std: Union[Tuple[float], None] = None,
    ) -> None:
        tf_list = []
        if self.tf_image_size:
            tf_list.append(T.Resize(self.tf_image_size))
        if not self.is_test:
            tf_list.append(
                T.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
                )
            )
        tf_list.append(T.ToTensor())
        if transform_mean and transform_std:
            tf_list.append(T.Normalize(transform_mean, transform_std))
        self.transform = T.Compose(transforms=tf_list)

    def _split_train_valid(self) -> None:
        df = pd.read_csv(self.csv_path)

        X = df["filename"]
        y = df.iloc[:, 1:]

        X_train, X_validate, y_train, y_validate = train_test_split(
            X, y, test_size=self.validation_split, random_state=42
        )

        self.csv_path_train = Path(
            deepcopy(self.csv_path).replace(".csv", "_train.csv")
        )
        self.csv_path_valid = Path(
            deepcopy(self.csv_path).replace(".csv", "_valid.csv")
        )
        if self.save_out_dir:
            out_dir = Path(self.save_out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path_train = out_dir / self.csv_path_train.name
            self.csv_path_valid = out_dir / self.csv_path_valid.name

        df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        df_valid = pd.concat([X_validate, y_validate], axis=1).reset_index(
            drop=True
        )

        df_train.to_csv(self.csv_path_train, index=False, header=True)
        df_valid.to_csv(self.csv_path_valid, index=False, header=True)
