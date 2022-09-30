import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from utils.project_utils import set_seed
from typing import Callable, Tuple, Union


class BaseDataLoader(DataLoader):
    """Base class for all data loaders."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        validation_split: Union[float, int],
        num_workers: int,
        collate_fn: Callable = default_collate,
    ) -> None:
        """Base data loader constructor.

        Args:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int): how many samples per batch to load.
            shuffle (bool): set to ``True`` to have the data reshuffled.
            validation_split (Union[float, int]): fraction of the training set used
            num_workers (int): how many subprocesses to use for data
            collate_fn (Callable, optional): funtion performing collation. Defaults to torch.utils.data.dataloader.default_collate.
        """
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)
        set_seed(0)
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split: Union[float, int]) -> Tuple[SubsetRandomSampler]:
        """Creates and returns a subset sampler for the training and validation.

        Args:
            split (Union[float, int]): fraction of the training set used for validation

        Returns:
            Tuple[SubsetRandomSampler]: train and validation samplers
        """
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)
        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, (
                "validation set size is configured to be larger than" " entire dataset."
            )
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # shuffle option is mutually exclusive with sampler, so set it to False
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self) -> DataLoader:
        """Splits validation dataloader.

        Returns:
            DataLoader: validation data loader
        """
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
