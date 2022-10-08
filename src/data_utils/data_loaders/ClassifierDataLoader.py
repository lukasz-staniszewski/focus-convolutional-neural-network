from pathlib import Path
from base import BaseDataLoader
from data_utils.data_sets import ClassifierDataset
from torch.utils.data import ConcatDataset
from typing import Tuple
import torchvision.transforms as T
import pandas as pd
from copy import deepcopy


class ClassifierDataLoader(BaseDataLoader):
    def __init__(
        self,
        images_dir: str,
        batch_size: int,
        csv_path: str,
        transform_mean: Tuple[float] = None,
        transform_std: Tuple[float] = None,
        shuffle: bool = True,
        validation_split: float = 0.0,
        num_workers: int = 1,
        is_test: bool = False,
        balance: bool = False,
    ):
        self.images_dir = images_dir
        self.csv_path = csv_path
        self.balance = balance

        self.combine_transforms(
            transform_mean=transform_mean, transform_std=transform_std
        )

        if is_test:
            self.dataset = ClassifierDataset(
                images_dir=self.images_dir,
                csv_path=self.csv_path,
                transform=self.transform,
            )
            validation_split = 0.0
            shuffle = False
        else:
            self.csv_aug_path = (
                self.balance_lower_class()
                if self.balance
                else self.csv_path
            )
            self.dataset = ConcatDataset(
                [
                    ClassifierDataset(
                        images_dir=self.images_dir,
                        csv_path=self.csv_path,
                        transform=self.transform,
                    ),
                    ClassifierDataset(
                        images_dir=self.images_dir,
                        csv_path=self.csv_aug_path,
                        transform=self.transform_aug,
                    ),
                ]
            )

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
        )

    def balance_lower_class(self) -> Path:
        df_orig = pd.read_csv(self.csv_path)
        label_cnt = df_orig["label"].value_counts()

        n_0, n_1 = label_cnt[0], label_cnt[1]
        freq_cls = int(n_0 <= n_1)
        cnt_diff = abs(n_0 - n_1)

        df_aug = df_orig[df_orig["label"] != freq_cls].sample(
            cnt_diff, replace=True
        )
        new_path = deepcopy(self.csv_path).replace(".csv", "_aug.csv")
        df_aug.to_csv(new_path, index=False, header=True)
        return new_path

    def combine_transforms(
        self,
        transform_mean: Tuple[float] = None,
        transform_std: Tuple[float] = None,
    ) -> None:
        if transform_mean and transform_std:
            self.transform = T.Compose(
                transforms=[
                    T.ToPILImage(),
                    T.ToTensor(),
                    T.Normalize(transform_mean, transform_std),
                ]
            )
            self.transform_aug = T.Compose(
                transforms=[
                    T.ToPILImage(),
                    T.ToTensor(),
                    T.Normalize(transform_mean, transform_std),
                    T.RandomHorizontalFlip(0.5),
                    T.RandomErasing(0.5),
                ]
            )
        else:
            self.transform = T.Compose(
                transforms=[T.ToPILImage(), T.ToTensor()]
            )
            self.transform_aug = T.Compose(
                transforms=[
                    T.ToPILImage(),
                    T.ToTensor(),
                    T.RandomHorizontalFlip(0.5),
                    T.RandomErasing(0.5),
                ]
            )
