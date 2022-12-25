#!/bin/bash

# Remember to use it on COCO-2017 directory
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip

mv train2017 train
mv val2017 validate
mv annotations_trainval2017 annotations

rm train2017.zip
rm val2017.zip
rm annotations_trainval2017.zip
