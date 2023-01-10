from external import transforms as T, utils
from datasets import MyCocoDataset
import torch


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def split_dls_coco(
    train_ann_path,
    train_img_dir,
    test_ann_path,
    test_img_dir,
    batch_size,
    num_workers=-1,
    validation_split=0.10,
    seed=0):
    
    # use our dataset and defined transformations
    dataset_train = MyCocoDataset(
        annotation_path=train_ann_path,
        images_dir=train_img_dir,
        transforms=get_transform(train=True),
    )
    dataset_validate = MyCocoDataset(
        annotation_path=train_ann_path,
        images_dir=train_img_dir,
        transforms=get_transform(train=False),
    )
    dataset_test = MyCocoDataset(
        annotation_path=test_ann_path,
        images_dir=test_img_dir,
        transforms=get_transform(train=False),
    )
    
    return split_train_test_valid_dls(
        dataset_train = dataset_train,
        dataset_validate = dataset_validate,
        dataset_test = dataset_test,
        batch_size = batch_size,
        num_workers = num_workers,
        validation_split = validation_split,
        seed = seed,
    )



def split_train_test_valid_dls(
    dataset_train,
    dataset_validate,
    dataset_test,
    batch_size,
    num_workers=-1,
    validation_split=0.10,
    seed=0,
):
    # split the valid dataset from the training dataset
    torch.manual_seed(seed)
    n_examples_validate = int(validation_split * len(dataset_train))
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train_final = torch.utils.data.Subset(
        dataset_train, indices[:-n_examples_validate]
    )
    dataset_validate_final = torch.utils.data.Subset(
        dataset_validate, indices[-n_examples_validate:]
    )

    # define training and validation data loaders
    dl_train = torch.utils.data.DataLoader(
        dataset_train_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )

    dl_validate = torch.utils.data.DataLoader(
        dataset_validate_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )

    dl_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )

    return dl_train, dl_validate, dl_test
