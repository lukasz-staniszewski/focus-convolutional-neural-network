import torch
import torchvision


class MyCocoDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_path, images_dir, transforms=None):
        self.annotation_path = annotation_path
        self.images_dir = images_dir
        self.transforms = transforms

        # load all image files
        self.dataset = torchvision.datasets.CocoDetection(
            root=images_dir, annFile=annotation_path
        )

    def __getitem__(self, idx):
        # load images and annotations
        image, annotations = self.dataset[idx]
        image = image.convert("RGB")

        # get bounding box coordinates and labels for each object
        boxes = []
        labels = []
        for i in range(len(annotations)):
            ann_bbox = annotations[i]["bbox"]
            ann_cat = annotations[i]["category_id"]
            xmin = ann_bbox[0]
            xmax = ann_bbox[0] + ann_bbox[2]
            ymin = ann_bbox[1]
            ymax = ann_bbox[1] + ann_bbox[3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann_cat)

        # convert everything into a torch.Tensor
        if len(annotations) != 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.as_tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.as_tensor([0], dtype=torch.int64)
            area = torch.zeros_like(labels, dtype=torch.float32)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros_like(
            labels, dtype=torch.int64
        )  # all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.dataset)


class MyPascalDataset:
    pass
