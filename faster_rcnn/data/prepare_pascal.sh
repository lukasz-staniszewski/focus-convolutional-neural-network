#!/bin/bash

# Directory initialization
mkdir PASCAL
cd PASCAL

# Downloading images
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar
mv VOCdevkit/VOC2012/* . 
rm -r VOCdevkit
rm -r SegmentationClass
rm -r SegmentationObject
cd ..

# Splitting train and test sets
python ./pascal_preprocessor.py --action split --images-dir ./PASCAL/JPEGImages --test-split 0.2 --ann-dir ./PASCAL/Annotations

# Converting annotations to COCO format
python ./pascal_preprocessor.py --action convert --ann-dir ./PASCAL/ann_train --output-json ./PASCAL/ann_train/annotations_train.json --labels ./pascal_labels.txt
python ./pascal_preprocessor.py --action convert --ann-dir ./PASCAL/ann_val --output-json ./PASCAL/ann_val/annotations_val.json --labels ./pascal_labels.txt

# Filtering annotations to specific categories
python ./coco_preprocessor.py --json_in_path ./PASCAL/ann_train/annotations_train.json --json_out_path ./PASCAL/ann_train/annotations_train_processed.json --categories_names person car bicycle
python ./coco_preprocessor.py --json_in_path ./PASCAL/ann_val/annotations_val.json --json_out_path ./PASCAL/ann_val/annotations_val_processed.json --categories_names person car bicycle