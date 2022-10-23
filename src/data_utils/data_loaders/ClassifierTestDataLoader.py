import pandas as pd
from base import BaseDataLoader
from data_utils.data_sets import ClassifierTestDataset
from typing import Any, Dict, Tuple, Union
import torchvision.transforms as T
from pathlib import Path


class ClassifierTestDataLoader(BaseDataLoader):
    def __init__(
        self,
        images_dir: str,
        batch_size: int,
        csv_path: Union[str, Path, None] = None,
        transform_mean: Union[Tuple[float], None] = None,
        transform_std: Union[Tuple[float], None] = None,
        num_workers: int = 1,
        is_multiclass: bool = False,
        labels: Dict[str, str] = None,
    ):
        self.images_dir = images_dir
        self.labels = labels
        self.is_multiclass = is_multiclass

        self.combine_transforms(
            transform_mean=transform_mean, transform_std=transform_std
        )

        self.dataset = ClassifierTestDataset(
            images_dir=self.images_dir,
            csv_path=csv_path,
            transform=self.transform,
        )

        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            validation_split=0.0,
            num_workers=num_workers,
        )

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
        else:
            self.transform = T.Compose(
                transforms=[T.ToPILImage(), T.ToTensor()]
            )

    def to_csv(self, csv_path: str, predictions: Any = None) -> None:
        out = {"filename": self.dataset.files}
        if predictions is not None:
            out["label"] = predictions

        df = pd.DataFrame(out)
        df.to_csv(csv_path, index=False)

    def get_targets(self) -> Any:
        if self.dataset.targets is not None:
            return self.dataset.targets
        else:
            raise Exception("No targets found in dataset!")
