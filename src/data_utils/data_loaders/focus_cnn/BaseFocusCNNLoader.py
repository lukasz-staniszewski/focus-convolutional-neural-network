from typing import Callable, Dict, Tuple, Union

from base import BaseDataLoader


class BaseFocusCNNLoader(BaseDataLoader):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        images_dir: str,
        is_test: bool,
        labels: Dict[str, str],
        csv_path: str = None,
        transform_mean: Tuple[float] = None,
        transform_std: Tuple[float] = None,
        collate_fn: Callable = None,
    ):
        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        self.images_dir = images_dir
        self.is_test = is_test
        self.labels = labels
        self.is_multilabel = (
            len(self.labels.keys()) > 2 if self.labels else None
        )

        assert (
            csv_path is not None or is_test
        ), "csv_path must be provided for training"
        self.csv_path = csv_path if csv_path else None

        self.combine_transforms(
            transform_mean=transform_mean, transform_std=transform_std
        )

        self.prepare_datasets()

    def prepare_datasets(self):
        if self.is_test:
            self.prepare_dataset_test()
        else:
            self.prepare_datasets_train_valid()

    def prepare_datasets_train_valid(self):
        raise NotImplementedError

    def prepare_dataset_test(self):
        raise NotImplementedError

    def combine_transforms(
        self,
        transform_mean: Union[Tuple[float], None] = None,
        transform_std: Union[Tuple[float], None] = None,
    ) -> None:
        raise NotImplementedError
