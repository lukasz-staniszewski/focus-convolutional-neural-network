import os
from pathlib import Path
from typing import Any, Union

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


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
        # img = torchvision.io.read_image(
        #     os.path.join(self.images_dir, filename)
        # ).float()
        img = Image.open(os.path.join(self.images_dir, filename))

        if self.transform is not None:
            img = self.transform(img)

        return img
