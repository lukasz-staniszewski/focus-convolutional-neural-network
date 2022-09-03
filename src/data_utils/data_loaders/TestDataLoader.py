from torchvision import transforms
from base import BaseDataLoader
from data_utils.data_sets import TestDataset


class TestDataLoader(BaseDataLoader):
    def __init__(
        self,
        images_folder,
        batch_size,
        shuffle=False,
        validation_split=0.0,
        num_workers=1,
    ):
        trsfm = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize(size=(360, 640)),
                transforms.Normalize(
                    (0.3683047, 0.42932022, 0.29250222),
                    (0.15938677, 0.16319054, 0.17476037),
                ),
            ]
        )
        self.index2class = {
            0: "side_view",
            1: "closeup",
            2: "non_match",
            3: "front_view",
            4: "side_gate_view",
            5: "aerial_view",
            6: "wide_view",
        }
        self.images_folder = images_folder
        self.dataset = TestDataset(
            images_folder=self.images_folder, transform=trsfm
        )

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
        )
