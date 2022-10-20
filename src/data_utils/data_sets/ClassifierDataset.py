import torchvision.transforms as T
import torchvision
from torch.utils.data import Dataset
from typing import List, Tuple, Any, Union
import os
from pathlib import Path
import pandas as pd


class ClassifierDataset(Dataset):
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
        filename = self.df["filename"][idx]
        label = self.df["label"][idx]
        img = torchvision.io.read_image(
            os.path.join(self.images_dir, filename)
        ).float()
        if self.transform is not None:
            img = self.transform(img)

        return img, label
