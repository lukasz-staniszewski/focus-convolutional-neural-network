from typing import Callable, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """Base class for all data loaders."""

    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        collate_fn: Callable = default_collate,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.dataset_train = None
        self.dataset_validate = None
        self.dataset_test = None
        self.batch_idx = 0

    def get_train_loader(self) -> DataLoader:
        if self.dataset_train is None:
            raise AttributeError("No training dataset provided.")
        self._set_n_samples()

        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )

    def get_valid_loader(self) -> DataLoader:
        if self.dataset_validate is None:
            raise AttributeError("No validation dataset provided.")

        return DataLoader(
            dataset=self.dataset_validate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )

    def get_test_loader(self) -> DataLoader:
        if self.dataset_test is None:
            raise AttributeError("No test dataset provided.")
        self._set_n_samples()

        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )

    def prepare_datasets(self) -> None:
        raise NotImplementedError

    def _set_n_samples(self) -> None:
        if self.dataset_train is None and self.dataset_test is None:
            raise AttributeError("No dataset provided.")
        self.n_samples = (
            len(self.dataset_train)
            if self.dataset_train
            else len(self.dataset_test)
        )
