import torchvision.transforms as T
import torchvision
from torch.utils.data import Dataset
import torch
import os
import pandas as pd
from typing import Union, Any
from pathlib import Path


class ClassifierTestDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        csv_path: Union[str, Path, None] = None,
        transform: Any = None,
    ) -> None:
        self.images_dir = images_dir

        if csv_path is None:
            self.files = [
                f
                for f in os.listdir(self.images_dir)
                if os.path.isfile(os.path.join(self.images_dir, f))
            ]
        else:
            df_data = pd.read_csv(csv_path, header=0)
            self.files = df_data["filename"]
            self.targets = df_data["label"]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        filename = self.files[idx]
        img = torchvision.io.read_image(
            os.path.join(self.images_dir, filename)
        ).float()

        if self.transform is not None:
            img = self.transform(img)

        return img
