ds_name="pascal"

for e in 0 1 2 3 4 5
do
    echo "-----Epoch ${ds_name} ${e}"
    echo "Epoch ${ds_name} ${e} ----"
    echo "-----Epoch ${ds_name} ${e}"
    CUDA_VISIBLE_DEVICES=4,5 python ./runner.py --action test --ann-train-path ./data/PASCAL/ann_train/annotations_train_processed.json --img-train-path ./data/PASCAL/images_train/ --ann-test-path ./data/PASCAL/ann_val/annotations_val_processed.json --img-test-path ./data/PASCAL/images_val/ --batch-size 8 --num-workers 32 --dataset_type ${ds_name} --n-classes 4 --model-path ./models/${ds_name}_epoch_${e}.pth
done