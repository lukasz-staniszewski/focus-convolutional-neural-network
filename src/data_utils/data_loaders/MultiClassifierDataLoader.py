from torch.utils.data import DataLoader
from data_utils.data_sets import ClassifierDataset
from torch.utils.data import ConcatDataset
from typing import Tuple, Union, Dict
import torchvision.transforms as T
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate
from typing import Callable


class MultiClassifierDataLoader(DataLoader):
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
        balance_train: bool = False,
        labels: Dict[str, str] = None,
        balance_max_multiplicity: int = 15,
        collate_fn: Callable = default_collate,
    ):
        # members
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_train = balance_train
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.labels = labels
        self.balance_max_multiplicity = balance_max_multiplicity
        self.collate_fn = collate_fn
        self.is_multiclass = True

        # csv files with labels for training and validation
        self.orig_csv_path = csv_path
        self.csv_path_train = None
        self.csv_path_train_aug = None
        self.csv_path_valid = None

        # transformations
        self.combine_transforms(
            transform_mean=transform_mean, transform_std=transform_std
        )
        self.prepare_datasets()

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )

    def get_valid_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_validate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )

    def prepare_datasets(self):
        assert (
            0.0 < self.validation_split < 1.0
        ), "Validation split must be in (0.0, 1.0)"
        self._split_train_valid()

        self.dataset_validate = ClassifierDataset(
            images_dir=self.images_dir,
            csv_path=self.csv_path_valid,
            transform=self.transform,
        )

        if self.balance_train:
            self.balance_data()
            self.dataset_train = ConcatDataset(
                [
                    ClassifierDataset(
                        images_dir=self.images_dir,
                        csv_path=self.csv_path_train,
                        transform=self.transform,
                    ),
                    ClassifierDataset(
                        images_dir=self.images_dir,
                        csv_path=self.csv_path_train_aug,
                        transform=self.transform_aug,
                    ),
                ]
            )
        else:
            self.dataset_train = ClassifierDataset(
                images_dir=self.images_dir,
                csv_path=self.csv_path_train,
                transform=self.transform,
            )

    def balance_data(self) -> None:
        """Function is desingned for multiclass classification and
        balances data using firstly oversampling minority classes
        and then undersampling majority class of training set.
        """
        # -- balances data by oversampling minority classes -- #
        df_orig = pd.read_csv(self.csv_path_train)
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
        self.csv_path_train_aug = deepcopy(self.csv_path_train).replace(
            ".csv", "_aug.csv"
        )
        df_aug.to_csv(self.csv_path_train_aug, index=False, header=True)

        # -- balances data by undersampling majority classes -- #
        # NOTE: max(x,1) is used to avoid division by zero
        new_label_cnt = (
            df_orig["label"]
            .value_counts()
            .add(df_aug["label"].value_counts(), fill_value=0)
        )

        if (
            new_label_cnt.max() / max(new_label_cnt.min(), 1)
            > self.balance_max_multiplicity
        ):
            df_undersampled = deepcopy(df_orig)
            min_cnt = new_label_cnt.min()
            for label, cnt in new_label_cnt.iteritems():
                if cnt / max(min_cnt, 1) > self.balance_max_multiplicity:
                    df_undersampled = df_undersampled.drop(
                        df_undersampled[df_undersampled["label"] == label]
                        .sample(
                            int(
                                cnt - (self.balance_max_multiplicity * min_cnt)
                            ),
                            replace=False,
                            random_state=0,
                        )
                        .index
                    )
            path_undersampled = deepcopy(self.csv_path_train).replace(
                ".csv", "_undersampled.csv"
            )
            df_undersampled.to_csv(path_undersampled, index=False, header=True)
            self.csv_path_train = path_undersampled

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

    def _split_train_valid(self) -> None:
        df = pd.read_csv(self.orig_csv_path)
        X_train, X_validate, y_train, y_validate = train_test_split(
            df["filename"],
            df["label"],
            test_size=self.validation_split,
            random_state=42,
            stratify=df["label"],
        )

        self.csv_path_train = deepcopy(self.orig_csv_path).replace(
            ".csv", "_train.csv"
        )
        self.csv_path_valid = deepcopy(self.orig_csv_path).replace(
            ".csv", "_valid.csv"
        )

        df_train = pd.DataFrame(
            {"filename": X_train, "label": y_train}
        ).reset_index(drop=True)
        df_valid = pd.DataFrame(
            {"filename": X_validate, "label": y_validate}
        ).reset_index(drop=True)

        df_train.to_csv(self.csv_path_train, index=False, header=True)
        df_valid.to_csv(self.csv_path_valid, index=False, header=True)
