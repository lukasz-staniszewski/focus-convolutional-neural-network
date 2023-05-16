import numpy as np
from tqdm import tqdm
import json
from pycocotools.coco import COCO
import argparse


class CocoPreprocessor:
    def __init__(self, json_in_path, json_out_path, categories_names):
        print("Starting preprocessor initialization...")
        self.coco = COCO(json_in_path)
        self.json_in_path = json_in_path
        self.json_out_path = json_out_path
        self.categories_names = categories_names

        self.cat_ids = self.coco.getCatIds(catNms=self.categories_names)
        self.images = self._get_imgs()
        self.cats = self._get_cats()
        self.annotations = []
        print("Preprocessor initialized!")

    def _get_imgs(self):
        """Returns all images metadata."""
        imgs_ids = self.coco.getImgIds(catIds=[])
        return self.coco.loadImgs(imgs_ids)

    def _get_cats(self):
        return self.coco.loadCats(self.cat_ids)

    def _get_closest_to_center_ann(self, cat_annotations, image_info):
        img_width, img_height = image_info["width"], image_info["height"]
        img_center = np.array([img_width / 2, img_height / 2])
        bboxes = np.array([ann["bbox"] for ann in cat_annotations])
        bbox_centers = np.array(
            [(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2) for bbox in bboxes]
        )
        diffs = np.linalg.norm(bbox_centers - img_center, axis=1)
        min_idx = np.argmin(diffs)
        return cat_annotations[min_idx]

    def _collect_annotations(self):
        # TODO: DEBUG HERE
        for image_info in tqdm(
            self.images, desc="Collecting annotations for images"
        ):
            annotation_ids = self.coco.getAnnIds(
                imgIds=image_info["id"], catIds=self.cat_ids, iscrowd=False
            )
            annotations = self.coco.loadAnns(annotation_ids)
            anns_dict = {
                cat_id: [
                    ann for ann in annotations if ann["category_id"] == cat_id
                ]
                for cat_id in self.cat_ids
            }
            for cat_id, cat_anns in anns_dict.items():
                if len(cat_anns) == 0:
                    continue
                elif len(cat_anns) == 1:
                    self.annotations.extend(cat_anns)
                else:
                    self.annotations.append(
                        self._get_closest_to_center_ann(cat_anns, image_info)
                    )

    def _save_new_annotations(self):
        new_json = {
            "info": {
                "description": (
                    "COCO 2017 dataset with less number of categories"
                ),
            },
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.cats,
        }

        # Write the JSON to a file
        print("Saving new json file...")
        with open(self.json_out_path, "w+") as output_file:
            json.dump(new_json, output_file)

        print("New COCO json saved.")

    def preprocess(self):
        self._collect_annotations()
        self._save_new_annotations()


def main(args):
    coco_preprocessor = CocoPreprocessor(
        args.json_in_path, args.json_out_path, args.categories_names
    )
    coco_preprocessor.preprocess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_in_path", type=str)
    parser.add_argument("--json_out_path", type=str)
    parser.add_argument("--categories_names", type=str, nargs="+")

    args = parser.parse_args()
    main(args)
