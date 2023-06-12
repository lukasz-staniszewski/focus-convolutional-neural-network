import os
from collections import OrderedDict
from math import log

import cv2
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from tqdm import tqdm

from base import BasePreprocessor
from base.base_preprocessor import BasePreprocessor
from data_utils.constants import COCO_2017_LABEL_MAP


class CocoFocusCNNPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_input_size = kwargs["max_input_size"]

        self.logger.info("Starting mapping CocoDetection dataset to memory...")
        self.dataset = torchvision.datasets.CocoDetection(
            root=self.img_in_dir_path, annFile=self.ann_file_path
        )
        self.logger.info(f"Mapping finished.")

        self.cls2idx_map = kwargs["cls_map"]
        self.idx2cls_map = {v: k for k, v in self.cls2idx_map.items()}

        self.labels_per_cls = {
            cls_idx: [] for cls_idx in self.cls2idx_map.values()
        }
        self.trans_x_per_cls = {
            cls_idx: [] for cls_idx in self.cls2idx_map.values()
        }
        self.trans_y_per_cls = {
            cls_idx: [] for cls_idx in self.cls2idx_map.values()
        }
        self.scales_per_cls = {
            cls_idx: [] for cls_idx in self.cls2idx_map.values()
        }

        self.filenames = []
        self.df_out = None
        self.img_counter = 0

        self._prepare_output_img_dir()

    def _add_gauss_reflect_padding(self, img, desired_size):
        # convert to opencv image
        cv2_img = np.array(img)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        # calculate padding sizes
        delta_width = desired_size[0] - img.size[0]
        delta_height = desired_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        # pad image with reflect border
        image = cv2.copyMakeBorder(
            cv2_img,
            top=pad_height,
            bottom=delta_height - pad_height,
            left=pad_width,
            right=delta_width - pad_width,
            borderType=cv2.BORDER_REFLECT,
        )
        # add gauss blur to this image
        image = cv2.GaussianBlur(
            image,
            ksize=(0, 0),
            sigmaX=10,
            sigmaY=10,
            borderType=cv2.BORDER_REFLECT,
        )
        # replace center of image with non-blurred image
        h_in, w_in, _ = cv2_img.shape
        h_bg, w_bg, _ = image.shape
        diff_y, diff_x = round((h_bg - h_in) / 2), round((w_bg - w_in) / 2)
        image[diff_y : diff_y + h_in, diff_x : diff_x + w_in] = cv2_img
        # return as Pillow Image
        img_out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_out)
        return pil_img

    def _move_bboxes(self, pos_bboxes, img_before, img_after):
        pad_width = (img_after.size[0] - img_before.size[0]) // 2
        pad_height = (img_after.size[1] - img_before.size[1]) // 2
        new_bboxes = [
            [bbox[0] + pad_width, bbox[1] + pad_height, bbox[2], bbox[3]]
            for bbox in pos_bboxes
        ]
        return new_bboxes

    def _make_bbox_square(self, bbox):
        center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        radius = max(bbox[2], bbox[3]) / 2
        return [center[0] - radius, center[1] - radius, radius * 2, radius * 2]

    def _get_bbox_closest_to_center(self, bboxes, image):
        centers = [
            (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2) for bbox in bboxes
        ]
        img_center = (image.size[0] / 2, image.size[1] / 2)
        distances_to_center = [
            np.sqrt(
                (center[0] - img_center[0]) ** 2
                + (center[1] - img_center[1]) ** 2
            )
            for center in centers
        ]
        idx_min = distances_to_center.index(min(distances_to_center))
        return bboxes[idx_min]

    def _collect_data(self):
        """Function collects data to create dataset."""
        self.collect_files()

        self.logger.info(
            "Images preprocessed, class distribution for classifier:"
            f" {self.cls_distribution}"
        )

        final_dict = OrderedDict()
        final_dict["filename"] = self.filenames
        for cls_idx in sorted(self.cls2idx_map.values()):
            assert len(self.filenames) == len(self.labels_per_cls[cls_idx])
            assert len(self.filenames) == len(self.trans_x_per_cls[cls_idx])
            assert len(self.filenames) == len(self.trans_y_per_cls[cls_idx])
            assert len(self.filenames) == len(self.scales_per_cls[cls_idx])

            final_dict[f"label_{cls_idx}"] = self.labels_per_cls[cls_idx]
            final_dict[f"tx_{cls_idx}"] = self.trans_x_per_cls[cls_idx]
            final_dict[f"ty_{cls_idx}"] = self.trans_y_per_cls[cls_idx]
            final_dict[f"scale_{cls_idx}"] = self.scales_per_cls[cls_idx]

        self.df_out = pd.DataFrame(final_dict)

    def _get_transform_params_from_bbox(self, bbox, image_padded):
        center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        radius = max(bbox[2], bbox[3]) / 2
        img_padded_center = (
            image_padded.size[0] / 2,
            image_padded.size[1] / 2,
        )

        scale = log(radius * 2 / image_padded.size[0])
        diff_x = 2 * (center[0] - img_padded_center[0]) / image_padded.size[0]
        diff_y = 2 * (center[1] - img_padded_center[1]) / image_padded.size[1]

        return scale, diff_x, diff_y

    def _process_image(self, image, cls_bboxes):
        # first pad image to have all images with same size
        img_padded = self._add_gauss_reflect_padding(
            img=image, desired_size=self.max_input_size
        )
        new_cls_bboxes = {}
        for cls_idx, bboxes in cls_bboxes.items():
            new_cls_bboxes[cls_idx] = self._move_bboxes(
                pos_bboxes=bboxes, img_before=image, img_after=img_padded
            )
        return img_padded, new_cls_bboxes

    def collect_files(self):
        progress_bar = tqdm(
            self.dataset,
            total=len(self.dataset),
            desc="Collecting data",
            colour="green",
        )
        zero_cnts = 0
        cls_cnts = {cls_name: 0 for cls_name in self.cls2idx_map.keys()}
        for image, annotations in progress_bar:
            progress_bar.set_postfix(
                {
                    f"N_{cls_name}": cls_cnt
                    for cls_name, cls_cnt in cls_cnts.items()
                }
            )

            cls_bboxes = {
                cls_idx: [
                    ann["bbox"]
                    for ann in annotations
                    if COCO_2017_LABEL_MAP[ann["category_id"]]
                    == self.idx2cls_map[cls_idx]
                    and ann["iscrowd"] == 0
                ]
                for cls_idx in self.cls2idx_map.values()
            }

            # image modification because of rotations
            image_processed, cls_bboxes = self._process_image(
                image=image, cls_bboxes=cls_bboxes
            )
            # save file
            self.img_counter += 1
            filename = f"{self.img_counter}.jpg"
            image_processed.save(os.path.join(self.img_out_dir_path, filename))
            self.filenames.append(filename)

            # process annotations per each class
            for cls_idx, bboxes in cls_bboxes.items():
                if len(bboxes) == 0:
                    self.labels_per_cls[cls_idx].append(0)
                    self.trans_x_per_cls[cls_idx].append(0)
                    self.trans_y_per_cls[cls_idx].append(0)
                    self.scales_per_cls[cls_idx].append(0)
                    zero_cnts += 1
                else:
                    bbox = self._get_bbox_closest_to_center(
                        bboxes=bboxes, image=image_processed
                    )
                    (
                        scale,
                        diff_x,
                        diff_y,
                    ) = self._get_transform_params_from_bbox(
                        bbox=bbox, image_padded=image_processed
                    )
                    self.labels_per_cls[cls_idx].append(cls_idx)
                    self.trans_x_per_cls[cls_idx].append(diff_x)
                    self.trans_y_per_cls[cls_idx].append(diff_y)
                    self.scales_per_cls[cls_idx].append(scale)
                    cls_cnts[self.idx2cls_map[cls_idx]] += 1

        self.cls_distribution = cls_cnts
        self.cls_distribution["none"] = zero_cnts

    def preprocess(self):
        self._collect_data()
        self._save_data()

    def _save_data(self):
        """Saves data to disk. In this case, saved is csv file with images filenames and labels."""
        assert (
            self.labels_per_cls
            and self.filenames
            and self.trans_x_per_cls
            and self.trans_y_per_cls
            and self.scales_per_cls
            and self.df_out is not None
        ), "Data not collected."

        self.df_out.to_csv(
            os.path.join(self.out_dir_path, "labels.csv"),
            index=False,
            sep=",",
            header=True,
        )
