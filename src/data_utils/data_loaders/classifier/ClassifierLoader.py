from pathlib import Path
from data_utils.data_loaders.classifier.BaseClassifierLoader import (
    BaseClassifierLoader,
)
from data_utils.data_sets import ClassifierDataset
from torch.utils.data import ConcatDataset
from typing import Tuple, Union, Dict, List
import torchvision.transforms as T
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from data_utils.data_loaders.utils import (
    label_oversample,
    label_undersample,
    label_make_0_half,
)


class ClassifierLoader(BaseClassifierLoader):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        images_dir: str,
        csv_path: str,
        labels: Dict[str, str],
        save_out_dir: str = None,
        transform_mean: Tuple[float] = None,
        transform_std: Tuple[float] = None,
        validation_split: float = 0.0,
        balance_train: bool = False,
        balance_methods: List[str] = None,
        balance_max_multiplicity: int = None,
    ):
        # members
        self.balance_train = balance_train
        self.balance_methods = balance_methods
        self.validation_split = validation_split
        self.balance_max_multiplicity = balance_max_multiplicity
        self.save_out_dir = save_out_dir

        assert (balance_train and balance_methods) or (
            not balance_train and not balance_methods
        ), "balance_train and balance_methods must be both set or both not set"

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
            self.dataset_validate = ClassifierDataset(
                images_dir=self.images_dir,
                csv_path=self.csv_path_valid,
                transform=self.transform,
            )

        # train set
        if self.balance_train:
            self.balance_data()
            if self.csv_path_train_aug:
                # balancing with augmentation
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
                # balancing with no augmentation
                self.dataset_train = ClassifierDataset(
                    images_dir=self.images_dir,
                    csv_path=self.csv_path_train,
                    transform=self.transform,
                )
        else:
            # no balancing
            self.dataset_train = ClassifierDataset(
                images_dir=self.images_dir,
                csv_path=self.csv_path_train,
                transform=self.transform,
            )

    def balance_data(self) -> None:
        """Function balances training data using firstly oversampling
        minority classes and then undersampling majority classes.

        Posibilities in balance_methods:
        * oversampling:
            -- 'oversample' - oversample minority classes
        * undersampling:
            -- 'undersample' - undersample majority classes
            -- 'make_0_half' - make class 0 (none) as max 50% of all data
        """
        # oversampling
        if "oversample" in self.balance_methods:
            self.csv_path_train_aug = label_oversample(
                csv_path_train_orig=self.csv_path_train,
            )
        else:
            self.csv_path_train_aug = None

        # undersampling
        if "undersample" in self.balance_methods:
            self.csv_path_train = label_undersample(
                csv_path_train_orig=self.csv_path_train,
                balance_max_multiplicity=self.balance_max_multiplicity,
                csv_path_train_aug=self.csv_path_train_aug,
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
        if transform_mean and transform_std:
            self.transform = T.Compose(
                transforms=[
                    T.ToTensor(),
                    T.Normalize(transform_mean, transform_std),
                ]
            )
            self.transform_aug = T.Compose(
                transforms=[
                    T.RandomHorizontalFlip(0.5),
                    T.ToTensor(),
                    T.RandomErasing(0.5),
                    T.Normalize(transform_mean, transform_std),
                ]
            )
        else:
            self.transform = T.Compose(transforms=[T.ToTensor()])
            self.transform_aug = T.Compose(
                transforms=[
                    T.RandomHorizontalFlip(0.5),
                    T.ToTensor(),
                    T.RandomErasing(0.5),
                ]
            )

    def _split_train_valid(self) -> None:
        df = pd.read_csv(self.csv_path)
        X_train, X_validate, y_train, y_validate = train_test_split(
            df["filename"],
            df["label"],
            test_size=self.validation_split,
            random_state=42,
            stratify=df["label"],
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

        df_train = pd.DataFrame(
            {"filename": X_train, "label": y_train}
        ).reset_index(drop=True)
        df_valid = pd.DataFrame(
            {"filename": X_validate, "label": y_validate}
        ).reset_index(drop=True)

        df_train.to_csv(self.csv_path_train, index=False, header=True)
        df_valid.to_csv(self.csv_path_valid, index=False, header=True)
