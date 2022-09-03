from torchvision import transforms
from base import BaseDataLoader
from data_utils.data_sets import ClassifierDataset
from torch.utils.data import ConcatDataset
from typing import List, Any
import torchvision.transforms as T


class ClassifierDataLoader(BaseDataLoader):
    def __init__(
        self,
        X: List[Any],
        y: List[int],
        batch_size: int,
        shuffle: bool = False,
        validation_split: float = 0.0,
        num_workers: int = 1,
        is_test: bool = False,
    ):
        self.transform = transforms.Compose(transforms=[T.ToTensor()])
        self.transform_aug = transforms.Compose(
            transforms=[
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.RandomErasing(0.5),
            ]
        )

        if is_test:
            self.dataset = ClassifierDataset(
                X_inp=X, y_inp=y, transform=self.transform
            )
            validation_split = 0.0
            shuffle = False
        else:
            self.dataset = ConcatDataset(
                [
                    ClassifierDataset(
                        X_inp=X, y_inp=y, transform=self.transform
                    ),
                    ClassifierDataset(
                        X_inp=X, y_inp=y, transform=self.transform_aug
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
