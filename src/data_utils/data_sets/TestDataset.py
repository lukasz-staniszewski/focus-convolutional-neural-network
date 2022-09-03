import torch
import torchvision
import os


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.files_names = [
            f
            for f in os.listdir(self.images_folder)
            if os.path.isfile(os.path.join(self.images_folder, f))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, idx):
        filename = self.files_names[idx]
        image = torchvision.io.read_image(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, filename
