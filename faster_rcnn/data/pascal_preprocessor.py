import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, "r") as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None) -> List[str]:
    ann_paths = [
        os.path.join(ann_dir_path, aid) for aid in os.listdir(ann_dir_path)
    ]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext("path")
    if path is None:
        filename = annotation_root.findtext("filename")
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.sub(r"[^\d]*", "", img_id))

    size = annotation_root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    image_info = {
        "file_name": filename,
        "height": height,
        "width": width,
        "id": img_id,
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext("name")
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find("bndbox")
    xmin = int(float(bndbox.findtext("xmin"))) - 1
    ymin = int(float(bndbox.findtext("ymin"))) - 1
    xmax = int(float(bndbox.findtext("xmax")))
    ymax = int(float(bndbox.findtext("ymax")))
    assert (
        xmax > xmin and ymax > ymin
    ), f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        "area": o_width * o_height,
        "iscrowd": 0,
        "bbox": [xmin, ymin, o_width, o_height],
        "category_id": category_id,
        "ignore": 0,
        "segmentation": [],  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(
    annotation_paths: List[str],
    label2id: Dict[str, int],
    output_jsonpath: str,
    extract_num_from_imgid: bool = True,
):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [],
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print("Start converting !")
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(
            annotation_root=ann_root,
            extract_num_from_imgid=extract_num_from_imgid,
        )
        img_id = img_info["id"]
        output_json_dict["images"].append(img_info)

        for obj in ann_root.findall("object"):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({"image_id": img_id, "id": bnd_id})
            output_json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {
            "supercategory": "none",
            "id": label_id,
            "name": label,
        }
        output_json_dict["categories"].append(category_info)

    with open(output_jsonpath, "w") as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def move_images_to_test_train(images_dir, test_split, ann_dir):
    print("Splitting images and annotations into train and test sets...")
    np.random.seed(0)
    img_files = os.listdir(images_dir)
    files_ids = np.array([os.path.splitext(f)[0] for f in img_files])
    np.random.shuffle(files_ids)

    images_train_path = os.path.abspath(
        os.path.join(images_dir, os.pardir, "images_train")
    )
    images_val_path = os.path.abspath(
        os.path.join(images_dir, os.pardir, "images_val")
    )
    ann_train_path = os.path.abspath(
        os.path.join(ann_dir, os.pardir, "ann_train")
    )
    ann_val_path = os.path.abspath(os.path.join(ann_dir, os.pardir, "ann_val"))

    os.makedirs(images_train_path, exist_ok=True)
    os.makedirs(images_val_path, exist_ok=True)
    os.makedirs(ann_train_path, exist_ok=True)
    os.makedirs(ann_val_path, exist_ok=True)

    test_size = int(len(files_ids) * test_split)
    files_train = files_ids[:-test_size]
    files_test = files_ids[-test_size:]

    for file_id in files_train:
        shutil.move(
            os.path.join(images_dir, file_id + ".jpg"),
            images_train_path,
        )
        shutil.move(os.path.join(ann_dir, file_id + ".xml"), ann_train_path)
    for file_id in files_test:
        shutil.move(
            os.path.join(images_dir, file_id + ".jpg"),
            images_val_path,
        )
        shutil.move(os.path.join(ann_dir, file_id + ".xml"), ann_val_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "This script support converting voc format xmls to coco format"
            " json"
        )
    )
    parser.add_argument(
        "--action", type=str, choices=["split", "convert"], help="action to do"
    )
    parser.add_argument(
        "--ann-dir",
        type=str,
        default=None,
        help=(
            "path to annotation files directory. It is not need when use"
            " --ann_paths_list"
        ),
    )
    parser.add_argument(
        "--labels", type=str, default=None, help="path to label list."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output.json",
        help="path to output json file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="path to images directory",
    )
    parser.add_argument(
        "--test-split",
        default=0.2,
        type=float,
        help="how many percent of data to be test data",
    )
    args = parser.parse_args()

    if args.action == "split":
        assert (
            args.images_dir is not None
            and args.test_split is not None
            and args.ann_dir is not None
        ), "Error: --images-dir and --test-split must be set !"
        move_images_to_test_train(
            images_dir=args.images_dir,
            test_split=args.test_split,
            ann_dir=args.ann_dir,
        )
    elif args.action == "convert":
        assert (
            args.ann_dir is not None
            and args.labels is not None
            and args.output_json is not None
        ), "Error: --ann_dir, --labels and --output must be set !"
        label2id = get_label2id(labels_path=args.labels)
        ann_paths = get_annpaths(ann_dir_path=args.ann_dir)
        convert_xmls_to_cocojson(
            annotation_paths=ann_paths,
            label2id=label2id,
            output_jsonpath=args.output_json,
            extract_num_from_imgid=True,
        )
    else:
        raise ValueError("Error: --action must be split or convert !")


if __name__ == "__main__":
    main()
