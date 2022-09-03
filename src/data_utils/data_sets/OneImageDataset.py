import torch
import torchvision
import os


class OneImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, image_name, transform=None):
        self.image_folder = image_folder
        self.image_name = image_name
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        filename = self.image_name
        image = torchvision.io.read_image(os.path.join(self.image_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image
