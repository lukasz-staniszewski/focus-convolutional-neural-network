from pathlib import Path
from base import BaseDataLoader
from data_utils.data_sets import ClassifierDataset
from torch.utils.data import ConcatDataset
from typing import Tuple, Union, Dict
import torchvision.transforms as T
import pandas as pd
from copy import deepcopy


class ClassifierDataLoader(BaseDataLoader):
    def __init__(
        self,
        images_dir: str,
        batch_size: int,
        csv_path: str,
        is_multiclass: bool = False,
        transform_mean: Tuple[float] = None,
        transform_std: Tuple[float] = None,
        shuffle: bool = True,
        validation_split: float = 0.0,
        num_workers: int = 1,
        is_test: bool = False,
        balance: bool = False,
        labels: Dict[str, str] = None,
    ):
        self.images_dir = images_dir
        self.csv_path = csv_path
        self.balance = balance
        self.is_multiclass = is_multiclass
        self.is_test = is_test
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.labels = labels

        self.combine_transforms(
            transform_mean=transform_mean, transform_std=transform_std
        )

        self.prepare_datasets()

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
        )

    def prepare_datasets(self):
        assert not (
            self.is_test and self.balance
        ), "Test data cannot be balanced"

        if self.balance:
            self.csv_aug_path = self.balance_data()
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

        else:
            self.dataset = ClassifierDataset(
                images_dir=self.images_dir,
                csv_path=self.csv_path,
                transform=self.transform,
            )
            if self.is_test:
                self.validation_split = 0.0
                self.shuffle = False

    def balance_data(self) -> Path:
        """Function balances data by oversampling minority
        classes and optionally undersampling majority class.

        Returns:
            Path: path to new augmented csv file
        """
        return (
            self.balance_multiclass()
            if self.is_multiclass
            else self.balance_lower_class()
        )

    def balance_multiclass(self) -> Path:
        """Function is desingned for multiclass classification and balances data using firstly oversampling minority classes and then undersampling majority class.

        Returns:
            Path: path to new augmented csv file
        """
        assert (
            self.is_multiclass
        ), "Cannot balance binary labeled data using this function"

        # balances data by oversampling minority classes
        df_orig = pd.read_csv(self.csv_path)
        label_cnt = df_orig["label"].value_counts().sort_index()
        aug_labels_cnt = label_cnt.max() / label_cnt
        aug_labels_cnt = aug_labels_cnt / aug_labels_cnt.max()
        aug_labels_cnt = (
            aug_labels_cnt[aug_labels_cnt > 0.1] * label_cnt.min()
        ).astype(int)
        dfs_aug = []
        for label, cnt in aug_labels_cnt.iteritems():
            dfs_aug.append(
                df_orig[df_orig["label"] == label].sample(
                    cnt, replace=False, random_state=0
                )
            )
        df_aug = pd.concat(dfs_aug, ignore_index=True)
        new_path_aug = deepcopy(self.csv_path).replace(
            ".csv", "_aug.csv"
        )
        df_aug.to_csv(new_path_aug, index=False, header=True)

        # balances data by undersampling majority class
        new_label_cnt = (
            df_orig["label"]
            .value_counts()
            .add(df_aug["label"].value_counts(), fill_value=0)
        )
        diff_biggest_rest = new_label_cnt.max() - (
            new_label_cnt.sum() - new_label_cnt.max()
        )
        if diff_biggest_rest > 0:
            biggest_class = new_label_cnt.idxmax()
            df_undersampled = df_orig.drop(
                df_orig[
                    df_orig["label"].astype(int) == int(biggest_class)
                ]
                .sample(
                    n=int(diff_biggest_rest),
                    replace=False,
                    random_state=0,
                )
                .index
            )
            new_path_undersampled = deepcopy(self.csv_path).replace(
                ".csv", "_undersampled.csv"
            )
            df_undersampled.to_csv(
                new_path_undersampled, index=False, header=True
            )
            self.csv_path = new_path_undersampled

        return new_path_aug

    def balance_lower_class(self) -> Path:
        """Function is desingned for binary classification and
        balances data by oversampling minority class.

        Returns:
            Path: path to new augmented csv file
        """
        assert (
            not self.is_multiclass
        ), "Cannot balance multiclass labeled data using this function"

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
        return Path(new_path)

    def combine_transforms(
        self,
        transform_mean: Union[Tuple[float], None] = None,
        transform_std: Union[Tuple[float], None] = None,
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
