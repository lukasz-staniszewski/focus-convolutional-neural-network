import pandas as pd
from base import BaseDataLoader
from data_utils.data_sets import ClassifierTestDataset
from typing import Any, Dict, Tuple, Union
import torchvision.transforms as T
from pathlib import Path
from copy import deepcopy


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
        balance: bool = False,
        labels: Dict[str, str] = None,
        balance_max_multiplicity: int = 15,
    ):
        self.images_dir = images_dir
        self.csv_path = csv_path
        self.labels = labels
        self.is_multiclass = is_multiclass
        self.balance = balance
        self.balance_max_multiplicity = balance_max_multiplicity

        self.combine_transforms(
            transform_mean=transform_mean, transform_std=transform_std
        )

        if self.balance:
            self.balance_data()

        self.dataset = ClassifierTestDataset(
            images_dir=self.images_dir,
            csv_path=self.csv_path,
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

    def balance_data(self) -> None:
        """Function balances data by undersampling majority class."""
        assert self.balance, "Argument balance is False, cannot balance data!"
        df_orig = pd.read_csv(self.csv_path)
        label_cnt = df_orig["label"].value_counts()
        if label_cnt.max() / label_cnt.min() > self.balance_max_multiplicity:
            df_undersampled = deepcopy(df_orig)
            min_cnt = label_cnt.min()
            for label, cnt in label_cnt.iteritems():
                if cnt / min_cnt > self.balance_max_multiplicity:
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
            path_undersampled = deepcopy(self.csv_path).replace(
                ".csv", "_undersampled.csv"
            )
            df_undersampled.to_csv(path_undersampled, index=False, header=True)
            self.csv_path = path_undersampled
