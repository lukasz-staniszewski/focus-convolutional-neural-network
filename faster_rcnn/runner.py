import argparse
from dataloaders import split_dls_coco
from model import get_faster_rcnn_model
from trainer import train
from external.engine import evaluate
import torch


def train_runner(args, model):
    assert args.dataset_type in ["pascal", "coco"], "Invalid dataset type."

    dl_train, dl_valid, _ = split_dls_coco(
        train_ann_path=args.ann_train_path,
        train_img_dir=args.img_train_path,
        test_ann_path=args.ann_test_path,
        test_img_dir=args.img_test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    train(
        model=model,
        dl_train=dl_train,
        dl_validate=dl_valid,
        model_name_prefix=args.dataset_type,
        lr=args.lr,
        n_epochs=args.n_epochs,
        print_freq=args.print_freq,
    )


def test_runner(args, model):
    assert args.model_path is not None, "Model path is required for testing."
    if args.dataset_type == "pascal":
        raise NotImplementedError("Pascal dataset is not implemented yet.")

    elif args.dataset_type == "coco":
        _, _, dl_test = split_dls_coco(
            train_ann_path=args.ann_train_path,
            train_img_dir=args.img_train_path,
            test_ann_path=args.ann_test_path,
            test_img_dir=args.img_test_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            validation_split=args.validation_split,
            seed=args.seed,
        )

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        evaluate(model, dl_test, device=device)


def main(args):
    model = get_faster_rcnn_model(args.n_classes)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

    if args.action == "train":
        train_runner(args, model)

    elif args.action == "test":
        test_runner(args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, choices=["train", "test"])
    parser.add_argument("--ann-train-path", type=str)
    parser.add_argument("--img-train-path", type=str)
    parser.add_argument("--ann-test-path", type=str)
    parser.add_argument("--img-test-path", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset_type", type=str, default="coco", choices=["coco", "pascal"]
    )
    parser.add_argument("--n-classes", type=int)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--model-path", type=str, required=False)

    args = parser.parse_args()
    main(args)
