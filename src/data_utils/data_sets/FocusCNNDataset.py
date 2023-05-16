import torchvision.transforms as T
import torchvision
from torch.utils.data import Dataset
from typing import Tuple, Any, Union, Dict
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from pipeline.pipeline_utils import convert_tf_params_to_bbox


class FocusCNNDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        csv_path: Union[str, Path],
        labels: Dict[str, str],
        transform: Any = None,
    ) -> None:
        self.df = pd.read_csv(csv_path, header=0)
        self.images_dir = images_dir
        self.transform = transform
        self.labels = labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        row = self.df.iloc[idx]
        filename = row["filename"]
        img_id = int(filename.split(".")[0])

        img = Image.open(os.path.join(self.images_dir, filename))
        if self.transform is not None:
            img = self.transform(img)

        outs = {}
        bboxes = []
        for cls in list(self.labels.keys())[1:]:
            label = int(row[f"label_{cls}"])
            idx_in_cols = 1 + (int(cls) - 1) * 4 + 1
            tf_params = torch.FloatTensor(
                row.iloc[idx_in_cols : idx_in_cols + 3]
            )

            outs[cls] = {}
            outs[cls]["label"] = int(row[f"label_{cls}"])
            outs[cls]["transform"] = tf_params

            if label != 0:
                bboxes.append(
                    torch.cat(
                        [
                            torch.Tensor([[img_id]]),
                            torch.Tensor([[label]]),
                            convert_tf_params_to_bbox(
                                translations=tf_params[0:2].unsqueeze(0),
                                scales=tf_params[2:3].unsqueeze(0),
                                img_size=img.shape[1:],
                            ),
                        ],
                        dim=1,
                    )
                )

        data = {
            "image": img,
            "image_id": img_id,
            "outs": outs,
            "bboxes": bboxes,
        }

        return data
