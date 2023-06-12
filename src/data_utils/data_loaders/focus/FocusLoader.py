from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from data_utils.data_loaders.focus.BaseFocusLoader import BaseFocusLoader
from data_utils.data_loaders.utils import label_make_0_half, label_undersample
from data_utils.data_sets import FocusDataset


class FocusLoader(BaseFocusLoader):
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
        balance_train: bool = False,
        balance_methods: List[str] = None,
        balance_max_multiplicity: int = None,
    ):
        # members
        self.validation_split = validation_split
        self.save_out_dir = save_out_dir
        self.tf_image_size = tf_image_size

        assert (balance_train and balance_methods) or (
            not balance_train and not balance_methods
        ), "balance_train and balance_methods must be both set or both not set"
        self.balance_train = balance_train
        self.balance_methods = balance_methods
        self.balance_max_multiplicity = balance_max_multiplicity

        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            images_dir=images_dir,
            csv_path=csv_path,
            is_test=False,
            labels=labels,
            transform_mean=transform_mean,
            transform_std=transform_std,
        )

    def prepare_dataset_test(self):
        raise NotImplementedError

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
            self.dataset_validate = FocusDataset(
                images_dir=self.images_dir,
                csv_path=self.csv_path_valid,
                transform=self.transform,
            )
        if self.balance_train:
            self.balance_data()
        # train set
        self.dataset_train = FocusDataset(
            images_dir=self.images_dir,
            csv_path=self.csv_path_train,
            transform=self.transform,
        )

    def balance_data(self) -> None:
        """Function balances training data using firstly oversampling
        minority classes and then undersampling majority classes.

        Posibilities in balance_methods:
        * undersampling:
            -- 'undersample' - undersample majority classes
            -- 'make_0_half' - make class 0 (none) as max 50% of all data
        """
        # undersampling
        if "undersample" in self.balance_methods:
            self.csv_path_train = label_undersample(
                csv_path_train_orig=self.csv_path_train,
                balance_max_multiplicity=self.balance_max_multiplicity,
            )
        elif "make_0_half" in self.balance_methods:
            self.csv_path_train = label_make_0_half(
                csv_path_train_orig=self.csv_path_train,
                csv_path_train_aug=self.csv_path_train_aug,
            )

    def combine_transforms(
        self,
        transform_mean: Union[Tuple[float], None] = None,
        transform_std: Union[Tuple[float], None] = None,
    ) -> None:
        tf_list = []
        if self.tf_image_size:
            tf_list.append(T.Resize(self.tf_image_size))
        tf_list += [
            T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
            ),
            T.ToTensor(),
        ]
        if transform_mean and transform_std:
            tf_list.append(T.Normalize(transform_mean, transform_std))
        self.transform = T.Compose(transforms=tf_list)

    def _split_train_valid(self) -> None:
        df = pd.read_csv(self.csv_path)

        X = df["filename"]
        y = df.iloc[:, 1:]

        X_train, X_validate, y_train, y_validate = train_test_split(
            X,
            y,
            test_size=self.validation_split,
            random_state=42,
            stratify=y["label"],
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
