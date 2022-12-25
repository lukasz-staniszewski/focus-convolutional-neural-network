import torchvision.transforms as T
import torchvision
from torch.utils.data import Dataset
from typing import Tuple, Any, Union
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from pipeline.utils import convert_tf_params_to_bbox


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
        transformations = torch.FloatTensor(row.iloc[2:])

        img = Image.open(os.path.join(self.images_dir, filename))
        if self.transform is not None:
            img = self.transform(img)

        bbox = convert_tf_params_to_bbox(
            translations=transformations[0:2].unsqueeze(0),
            scales=transformations[2:3].unsqueeze(0),
            img_size=img.shape[1:],
        ).squeeze()

        data = {
            "image": img,
            "label": label,
            "transform": transformations,
            "bbox": bbox,
        }

        return data
