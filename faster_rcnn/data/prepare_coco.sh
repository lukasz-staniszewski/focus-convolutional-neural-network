#!/bin/bash

# Directory initialization
mkdir COCO
cd COCO

# Downloading images
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip
rm train2017.zip
rm val2017.zip
mv train2017 train_images
mv val2017 val_images

# Downloading annotations
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
mv annotations_trainval2017 

# Converting annotations
cd ..
python ./coco_preprocessor.py --json_in_path ./COCO/annotations/instances_train2017.json --json_out_path ./COCO/annotations/annotations_train.json --categories_names person car bicycle
python ./coco_preprocessor.py --json_in_path ./COCO/annotations/instances_val2017.json --json_out_path ./COCO/annotations/annotations_val.json --categories_names person car bicycle