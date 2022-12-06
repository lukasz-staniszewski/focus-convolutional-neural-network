#!/bin/sh

ep_start=$1
ep_end=$2
models_path=$3
cfg_file=$4


for idx_epoch in `seq ${ep_start} ${ep_end}`
do

    python ./src/test_runner.py --model_path "${models_path}/checkpoint-epoch${idx_epoch}.pth" --config "${cfg_file}"

done