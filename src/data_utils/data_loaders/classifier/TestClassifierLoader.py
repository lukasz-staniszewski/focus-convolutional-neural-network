import pandas as pd
from data_utils.data_loaders.classifier.BaseClassifierLoader import (
    BaseClassifierLoader,
)
from data_utils.data_sets import ClassifierTestDataset
from typing import Any, Dict, Tuple, Union
import torchvision.transforms as T
from pathlib import Path
from copy import deepcopy


class TestClassifierLoader(BaseClassifierLoader):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        images_dir: str,
        labels: Dict[str, str],
        csv_path: Union[str, Path] = None,
        transform_mean: Tuple[float] = None,
        transform_std: Tuple[float] = None,
    ):
        self.validation_split = 0.0
        self.balance_train = False

        super().__init__(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            images_dir=images_dir,
            csv_path=csv_path,
            is_test=True,
            labels=labels,
            transform_mean=transform_mean,
            transform_std=transform_std,
        )

    def prepare_dataset_train_valid(self):
        raise NotImplementedError

    def prepare_dataset_test(self):
        self.dataset_test = ClassifierTestDataset(
            images_dir=self.images_dir,
            csv_path=self.csv_path,
            transform=self.transform,
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
        else:
            self.transform = T.Compose(transforms=[T.ToTensor()])

    def to_csv(self, csv_path: str, predictions: Any = None) -> None:
        out = {"filename": self.dataset_test.files}
        if predictions is not None:
            out["label"] = predictions

        df = pd.DataFrame(out)
        df.to_csv(csv_path, index=False)

    def get_targets(self) -> Any:
        if self.dataset_test.targets is not None:
            return self.dataset_test.targets
        else:
            raise Exception("No targets found in dataset!")
