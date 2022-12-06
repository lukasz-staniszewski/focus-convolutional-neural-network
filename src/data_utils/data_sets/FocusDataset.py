import torchvision.transforms as T
import torchvision
from torch.utils.data import Dataset
from typing import Tuple, Any, Union
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch


class FocusDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        csv_path: Union[str, Path],
        transform: Any = None,
    ) -> None:
        self.df = pd.read_csv(csv_path, header=0)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        row = self.df.iloc[idx]
        filename = row["filename"]

        label = int(row["label"])
        transform = torch.FloatTensor(row.iloc[2:])

        img = Image.open(os.path.join(self.images_dir, filename))
        if self.transform is not None:
            img = self.transform(img)

        data = {
            "image": img,
            "label": label,
            "transform": transform,
        }

        return data
