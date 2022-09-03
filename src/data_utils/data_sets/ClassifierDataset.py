import torchvision.transforms as T
from torch.utils.data import Dataset
from typing import List, Tuple, Any


class ClassifierDataset(Dataset):
    def __init__(
        self, X_inp: List[int], y_inp: List[int], transform: T = None
    ) -> None:
        self.X = X_inp
        self.y = y_inp
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        X, y = self.X[idx], self.y[idx]
        if self.transform:
            X = self.transform(X)
        return X, y
